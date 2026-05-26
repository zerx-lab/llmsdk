//! Streaming state machine: SSE chunks → [`StreamPart`].
//!
//! Mirrors the `TransformStream` body in `xai-responses-language-model.ts`'s
//! `doStream`. Stateful accumulator consuming one chunk at a time.
//!
//! Per-id block tracking:
//! - Text blocks: id = `text-{item_id}` (one per message item)
//! - Reasoning blocks: id = `reasoning-{item_id}`
//! - Tool calls: id = `tool_call.id` (function tools) or `tool_call.call_id`
//!   (provider-executed)
// Rust guideline compliant 2026-05-25

use std::collections::{HashMap, HashSet};

use llmsdk_provider::language_model::{
    FinishReason, FinishReasonKind, ResponseMetadata, Source, StreamPart, ToolCallPart, ToolResult,
    ToolResultOutput,
};
use llmsdk_provider::shared::{ProviderMetadata, Warning};
use llmsdk_provider_utils::time::rfc3339_from_unix_seconds;
use serde_json::{Map, Value, json};

use super::finish_reason::map as map_finish_reason;
use super::parse_response::next_id as next_citation_id;
use super::prepare_tools::ResolvedToolNames;
use super::usage;
use super::wire::{OutputItem, ResponsesChunk, ToolCallItem, WireUsage};

const WEB_SEARCH_SUB_TOOLS: &[&str] = &["web_search", "web_search_with_snippets", "browse_page"];
const X_SEARCH_SUB_TOOLS: &[&str] = &[
    "x_user_search",
    "x_keyword_search",
    "x_semantic_search",
    "x_thread_fetch",
];

/// State machine driving an xAI Responses stream.
#[derive(Debug)]
pub(crate) struct StreamState {
    initial_warnings: Option<Vec<Warning>>,
    finish_reason: FinishReason,
    last_usage: Option<WireUsage>,
    cost_in_usd_ticks: Option<u64>,
    is_first_chunk: bool,
    text_blocks_open: HashSet<String>,
    reasoning_blocks_open: HashSet<String>,
    seen_tool_calls: HashSet<String>,
    /// `output_index -> (tool_call_id, tool_name)` for streaming function-call
    /// argument deltas.
    ongoing_tool_calls: HashMap<u32, (String, String)>,
    citation_seed: u64,
    has_function_call: bool,
    names: ResolvedToolNames,
    include_raw_chunks: bool,
}

impl StreamState {
    /// Construct.
    pub(crate) fn new(
        warnings: Vec<Warning>,
        names: ResolvedToolNames,
        include_raw_chunks: bool,
    ) -> Self {
        Self {
            initial_warnings: Some(warnings),
            finish_reason: FinishReason::new(FinishReasonKind::Other),
            last_usage: None,
            cost_in_usd_ticks: None,
            is_first_chunk: true,
            text_blocks_open: HashSet::new(),
            reasoning_blocks_open: HashSet::new(),
            seen_tool_calls: HashSet::new(),
            ongoing_tool_calls: HashMap::new(),
            citation_seed: 0,
            has_function_call: false,
            names,
            include_raw_chunks,
        }
    }

    /// Initial frame: `StreamStart` with warnings.
    pub(crate) fn start_frames(&mut self) -> Vec<StreamPart> {
        let warnings = self.initial_warnings.take().unwrap_or_default();
        vec![StreamPart::StreamStart { warnings }]
    }

    /// Consume one chunk; returns the frames to forward.
    ///
    /// `raw_value` is the original JSON value of the chunk; forwarded as
    /// `StreamPart::Raw` when `include_raw_chunks=true`.
    pub(crate) fn on_chunk(
        &mut self,
        chunk: ResponsesChunk,
        raw_value: Option<Value>,
    ) -> Vec<StreamPart> {
        let mut out = Vec::new();

        if self.include_raw_chunks
            && let Some(rv) = raw_value
        {
            out.push(StreamPart::Raw { raw_value: rv });
        }

        match chunk {
            ResponsesChunk::ResponseCreated { response }
            | ResponsesChunk::ResponseInProgress { response } => {
                if self.is_first_chunk {
                    self.is_first_chunk = false;
                    out.push(StreamPart::ResponseMetadata(ResponseMetadata {
                        id: response.id,
                        timestamp: response.created_at.map(rfc3339_from_unix_seconds),
                        model_id: response.model,
                        headers: None,
                    }));
                }
            }

            ResponsesChunk::ReasoningSummaryPartAdded { item_id, .. } => {
                let block_id = reasoning_block_id(&item_id);
                if self.reasoning_blocks_open.insert(block_id.clone()) {
                    out.push(StreamPart::ReasoningStart {
                        id: block_id,
                        provider_metadata: Some(xai_item_id_metadata(&item_id)),
                    });
                }
            }

            ResponsesChunk::ReasoningSummaryTextDelta { item_id, delta, .. } => {
                out.push(StreamPart::ReasoningDelta {
                    id: reasoning_block_id(&item_id),
                    delta,
                    provider_metadata: Some(xai_item_id_metadata(&item_id)),
                });
            }
            ResponsesChunk::ReasoningSummaryTextDone { .. }
            | ResponsesChunk::ReasoningTextDone { .. } => {
                // closed by reasoning output_item.done
            }

            ResponsesChunk::ReasoningTextDelta { item_id, delta, .. } => {
                let block_id = reasoning_block_id(&item_id);
                if self.reasoning_blocks_open.insert(block_id.clone()) {
                    out.push(StreamPart::ReasoningStart {
                        id: block_id.clone(),
                        provider_metadata: Some(xai_item_id_metadata(&item_id)),
                    });
                }
                out.push(StreamPart::ReasoningDelta {
                    id: block_id,
                    delta,
                    provider_metadata: Some(xai_item_id_metadata(&item_id)),
                });
            }

            ResponsesChunk::OutputTextDelta { item_id, delta, .. } => {
                let block_id = text_block_id(&item_id);
                if self.text_blocks_open.insert(block_id.clone()) {
                    out.push(StreamPart::TextStart {
                        id: block_id.clone(),
                        provider_metadata: None,
                    });
                }
                out.push(StreamPart::TextDelta {
                    id: block_id,
                    delta,
                    provider_metadata: None,
                });
            }

            ResponsesChunk::OutputTextDone { annotations, .. } => {
                if let Some(anns) = annotations {
                    for ann in anns {
                        if let Some((url, title)) = ann.as_url_citation() {
                            out.push(StreamPart::Source(Source::Url {
                                id: next_citation_id(&mut self.citation_seed),
                                url: url.to_owned(),
                                title: Some(title.unwrap_or(url).to_owned()),
                                provider_metadata: None,
                            }));
                        }
                    }
                }
            }

            ResponsesChunk::OutputTextAnnotationAdded { annotation, .. } => {
                if let Some((url, title)) = annotation.as_url_citation() {
                    out.push(StreamPart::Source(Source::Url {
                        id: next_citation_id(&mut self.citation_seed),
                        url: url.to_owned(),
                        title: Some(title.unwrap_or(url).to_owned()),
                        provider_metadata: None,
                    }));
                }
            }

            ResponsesChunk::FunctionCallArgumentsDelta {
                output_index,
                delta,
                ..
            } => {
                if let Some((id, _name)) = self.ongoing_tool_calls.get(&output_index) {
                    out.push(StreamPart::ToolInputDelta {
                        id: id.clone(),
                        delta,
                        provider_metadata: None,
                    });
                }
            }
            ResponsesChunk::FunctionCallArgumentsDone { .. } => {
                // output_item.done emits the final tool-call.
            }

            ResponsesChunk::CustomToolCallInputDelta { .. }
            | ResponsesChunk::CustomToolCallInputDone { .. } => {
                // output_item events drive these.
            }

            ResponsesChunk::OutputItemAdded { item, output_index } => {
                handle_output_item(self, item, output_index, /* done */ false, &mut out);
            }
            ResponsesChunk::OutputItemDone { item, output_index } => {
                handle_output_item(self, item, output_index, /* done */ true, &mut out);
            }

            ResponsesChunk::ResponseDone { response }
            | ResponsesChunk::ResponseCompleted { response } => {
                if let Some(u) = &response.usage {
                    self.last_usage = Some(u.clone());
                    self.cost_in_usd_ticks = u.cost_in_usd_ticks;
                }
                if let Some(status) = response.status {
                    self.finish_reason = if self.has_function_call {
                        FinishReason::with_raw(FinishReasonKind::ToolCalls, status)
                    } else {
                        map_finish_reason(Some(&status))
                    };
                }
            }
            ResponsesChunk::ResponseIncomplete { response } => {
                if let Some(u) = &response.usage {
                    self.last_usage = Some(u.clone());
                    self.cost_in_usd_ticks = u.cost_in_usd_ticks;
                }
                let reason = response
                    .incomplete_details
                    .as_ref()
                    .and_then(|d| d.reason.as_deref());
                self.finish_reason = FinishReason {
                    unified: reason
                        .map(|r| map_finish_reason(Some(r)).unified)
                        .unwrap_or(FinishReasonKind::Other),
                    raw: Some(reason.unwrap_or("incomplete").to_owned()),
                };
            }
            ResponsesChunk::ResponseFailed { response } => {
                let reason = response
                    .incomplete_details
                    .as_ref()
                    .and_then(|d| d.reason.as_deref());
                self.finish_reason = FinishReason {
                    unified: reason
                        .map(|r| map_finish_reason(Some(r)).unified)
                        .unwrap_or(FinishReasonKind::Error),
                    raw: Some(reason.unwrap_or("error").to_owned()),
                };
                if let Some(u) = &response.usage {
                    self.last_usage = Some(u.clone());
                    self.cost_in_usd_ticks = u.cost_in_usd_ticks;
                }
            }

            ResponsesChunk::Error {
                code,
                message,
                param,
            } => {
                let mut payload = Map::new();
                payload.insert("type".into(), json!("error"));
                payload.insert("message".into(), json!(message));
                if let Some(c) = code {
                    payload.insert("code".into(), json!(c));
                }
                if let Some(p) = param {
                    payload.insert("param".into(), json!(p));
                }
                out.push(StreamPart::Error {
                    error: Value::Object(payload),
                });
            }

            ResponsesChunk::Other => {}

            // Status-only frames for the tool-call lifecycle. ai-sdk's
            // xai-responses-language-model.ts ignores these — surface state
            // exclusively via `response.output_item.added` / `.done`.
            ResponsesChunk::ContentPartAdded { .. }
            | ResponsesChunk::ContentPartDone { .. }
            | ResponsesChunk::ReasoningSummaryPartDone { .. }
            | ResponsesChunk::WebSearchCallInProgress { .. }
            | ResponsesChunk::WebSearchCallSearching { .. }
            | ResponsesChunk::WebSearchCallCompleted { .. }
            | ResponsesChunk::XSearchCallInProgress { .. }
            | ResponsesChunk::XSearchCallSearching { .. }
            | ResponsesChunk::XSearchCallCompleted { .. }
            | ResponsesChunk::FileSearchCallInProgress { .. }
            | ResponsesChunk::FileSearchCallSearching { .. }
            | ResponsesChunk::FileSearchCallCompleted { .. }
            | ResponsesChunk::CodeExecutionCallInProgress { .. }
            | ResponsesChunk::CodeExecutionCallExecuting { .. }
            | ResponsesChunk::CodeExecutionCallCompleted { .. }
            | ResponsesChunk::CodeInterpreterCallInProgress { .. }
            | ResponsesChunk::CodeInterpreterCallExecuting { .. }
            | ResponsesChunk::CodeInterpreterCallInterpreting { .. }
            | ResponsesChunk::CodeInterpreterCallCompleted { .. }
            | ResponsesChunk::CodeInterpreterCallCodeDelta { .. }
            | ResponsesChunk::CodeInterpreterCallCodeDone { .. }
            | ResponsesChunk::McpCallInProgress { .. }
            | ResponsesChunk::McpCallExecuting { .. }
            | ResponsesChunk::McpCallCompleted { .. }
            | ResponsesChunk::McpCallFailed { .. }
            | ResponsesChunk::McpCallArgumentsDelta { .. }
            | ResponsesChunk::McpCallArgumentsDone { .. }
            | ResponsesChunk::McpCallOutputDelta { .. }
            | ResponsesChunk::McpCallOutputDone { .. } => {}
        }

        out
    }

    /// Surface a JSON parse failure as an in-stream error.
    pub(crate) fn on_parse_error(&mut self, raw: &str, message: &str) -> Vec<StreamPart> {
        self.finish_reason = FinishReason::new(FinishReasonKind::Error);
        vec![StreamPart::Error {
            error: json!({ "message": message, "raw": raw }),
        }]
    }

    /// Final flush: emit pending `*End` frames, then `Finish`.
    pub(crate) fn flush(self) -> Vec<StreamPart> {
        let mut out: Vec<StreamPart> = Vec::new();

        for id in self.text_blocks_open {
            out.push(StreamPart::TextEnd {
                id,
                provider_metadata: None,
            });
        }

        let usage_value = self
            .last_usage
            .as_ref()
            .map_or_else(usage::zero, usage::convert);

        let provider_metadata = self.cost_in_usd_ticks.map(|ticks| {
            let mut xai = Map::new();
            xai.insert("costInUsdTicks".into(), json!(ticks));
            let mut outer = ProviderMetadata::new();
            outer.insert("xai".into(), xai);
            outer
        });

        out.push(StreamPart::Finish {
            usage: usage_value,
            finish_reason: self.finish_reason,
            provider_metadata,
        });
        out
    }
}

fn handle_output_item(
    state: &mut StreamState,
    item: OutputItem,
    output_index: u32,
    done: bool,
    out: &mut Vec<StreamPart>,
) {
    match item {
        OutputItem::Reasoning(r) => {
            if !done {
                return;
            }
            let block_id = reasoning_block_id(&r.id);
            if state.reasoning_blocks_open.remove(&block_id) {
                // already opened by reasoning_summary_part.added / reasoning_text.delta
            } else {
                // emit start even if no delta arrived (encrypted-only case)
                out.push(StreamPart::ReasoningStart {
                    id: block_id.clone(),
                    provider_metadata: Some(xai_item_id_metadata(&r.id)),
                });
            }
            let mut xai = Map::new();
            if let Some(enc) = &r.encrypted_content {
                xai.insert("reasoningEncryptedContent".into(), json!(enc));
            }
            if !r.id.is_empty() {
                xai.insert("itemId".into(), json!(r.id.clone()));
            }
            let mut po = ProviderMetadata::new();
            po.insert("xai".into(), xai);
            out.push(StreamPart::ReasoningEnd {
                id: block_id,
                provider_metadata: Some(po),
            });
        }

        OutputItem::Message(m) => {
            if !done {
                return;
            }
            for part in m.content {
                if let Some(text) = &part.text
                    && !text.is_empty()
                {
                    let block_id = text_block_id(&m.id);
                    if state.text_blocks_open.insert(block_id.clone()) {
                        out.push(StreamPart::TextStart {
                            id: block_id.clone(),
                            provider_metadata: None,
                        });
                        out.push(StreamPart::TextDelta {
                            id: block_id,
                            delta: text.clone(),
                            provider_metadata: None,
                        });
                    }
                }
                if let Some(anns) = part.annotations {
                    for ann in anns {
                        if let Some((url, title)) = ann.as_url_citation() {
                            out.push(StreamPart::Source(Source::Url {
                                id: next_citation_id(&mut state.citation_seed),
                                url: url.to_owned(),
                                title: Some(title.unwrap_or(url).to_owned()),
                                provider_metadata: None,
                            }));
                        }
                    }
                }
            }
        }

        OutputItem::FunctionCall(f) => {
            if !done {
                state
                    .ongoing_tool_calls
                    .insert(output_index, (f.call_id.clone(), f.name.clone()));
                out.push(StreamPart::ToolInputStart {
                    id: f.call_id,
                    tool_name: f.name,
                    provider_executed: None,
                    dynamic: None,
                    title: None,
                    provider_metadata: None,
                });
            } else {
                state.has_function_call = true;
                state.ongoing_tool_calls.remove(&output_index);
                out.push(StreamPart::ToolInputEnd {
                    id: f.call_id.clone(),
                    provider_metadata: None,
                });
                let input = serde_json::from_str::<Value>(&f.arguments)
                    .unwrap_or(Value::String(f.arguments.clone()));
                out.push(StreamPart::ToolCall(ToolCallPart {
                    tool_call_id: f.call_id,
                    tool_name: f.name,
                    input,
                    provider_executed: None,
                    dynamic: None,
                    provider_options: None,
                }));
            }
        }

        OutputItem::FileSearchCall(fs) => {
            let tool_name = state
                .names
                .file_search
                .clone()
                .unwrap_or_else(|| "file_search".to_owned());
            if state.seen_tool_calls.insert(fs.id.clone()) {
                out.push(StreamPart::ToolInputStart {
                    id: fs.id.clone(),
                    tool_name: tool_name.clone(),
                    provider_executed: Some(true),
                    dynamic: None,
                    title: None,
                    provider_metadata: None,
                });
                out.push(StreamPart::ToolInputDelta {
                    id: fs.id.clone(),
                    delta: String::new(),
                    provider_metadata: None,
                });
                out.push(StreamPart::ToolInputEnd {
                    id: fs.id.clone(),
                    provider_metadata: None,
                });
                out.push(StreamPart::ToolCall(ToolCallPart {
                    tool_call_id: fs.id.clone(),
                    tool_name: tool_name.clone(),
                    input: Value::String(String::new()),
                    provider_executed: Some(true),
                    dynamic: None,
                    provider_options: None,
                }));
            }
            if done {
                let queries = fs.queries.clone().unwrap_or_default();
                let results_value = fs
                    .results
                    .clone()
                    .map(|rs| {
                        Value::Array(
                            rs.into_iter()
                                .map(|r| {
                                    json!({
                                        "fileId": r.file_id,
                                        "filename": r.filename,
                                        "score": r.score,
                                        "text": r.text,
                                    })
                                })
                                .collect(),
                        )
                    })
                    .unwrap_or(Value::Null);
                let output_value = json!({
                    "queries": queries,
                    "results": results_value,
                });
                out.push(StreamPart::ToolResult(ToolResult {
                    tool_call_id: fs.id,
                    tool_name,
                    output: ToolResultOutput::Json {
                        value: output_value,
                        provider_options: None,
                    },
                    provider_metadata: None,
                }));
            }
        }

        OutputItem::WebSearchCall(t) => {
            emit_server_tool(state, t, done, ServerToolKind::WebSearch, out);
        }
        OutputItem::XSearchCall(t) => {
            emit_server_tool(state, t, done, ServerToolKind::XSearch, out);
        }
        OutputItem::CodeInterpreterCall(t) | OutputItem::CodeExecutionCall(t) => {
            emit_server_tool(state, t, done, ServerToolKind::CodeExecution, out);
        }
        OutputItem::ViewImageCall(t) => {
            emit_server_tool(state, t, done, ServerToolKind::ViewImage, out);
        }
        OutputItem::ViewXVideoCall(t) => {
            emit_server_tool(state, t, done, ServerToolKind::ViewXVideo, out);
        }
        OutputItem::CustomToolCall(t) => {
            // custom tools only emit on done (input fully assembled).
            if done {
                emit_server_tool(state, t, true, ServerToolKind::Custom, out);
            }
        }
        OutputItem::McpCall(m) => {
            let tool_name = state
                .names
                .mcp
                .clone()
                .or(m.name.clone())
                .unwrap_or_else(|| "mcp".to_owned());
            if state.seen_tool_calls.insert(m.id.clone()) {
                let input_str = m.arguments.clone().unwrap_or_default();
                out.push(StreamPart::ToolInputStart {
                    id: m.id.clone(),
                    tool_name: tool_name.clone(),
                    provider_executed: Some(true),
                    dynamic: None,
                    title: None,
                    provider_metadata: None,
                });
                out.push(StreamPart::ToolInputDelta {
                    id: m.id.clone(),
                    delta: input_str.clone(),
                    provider_metadata: None,
                });
                out.push(StreamPart::ToolInputEnd {
                    id: m.id.clone(),
                    provider_metadata: None,
                });
                out.push(StreamPart::ToolCall(ToolCallPart {
                    tool_call_id: m.id,
                    tool_name,
                    input: Value::String(input_str),
                    provider_executed: Some(true),
                    dynamic: None,
                    provider_options: None,
                }));
            }
        }

        OutputItem::Other => {}
    }
}

#[derive(Debug, Clone, Copy)]
enum ServerToolKind {
    WebSearch,
    XSearch,
    CodeExecution,
    ViewImage,
    ViewXVideo,
    Custom,
}

fn emit_server_tool(
    state: &mut StreamState,
    item: ToolCallItem,
    _done: bool,
    kind: ServerToolKind,
    out: &mut Vec<StreamPart>,
) {
    if !state.seen_tool_calls.insert(item.id.clone()) {
        return;
    }
    let raw_name = item.name.clone().unwrap_or_default();
    let tool_name = match kind {
        ServerToolKind::WebSearch => state
            .names
            .web_search
            .clone()
            .unwrap_or_else(|| "web_search".to_owned()),
        ServerToolKind::XSearch => state
            .names
            .x_search
            .clone()
            .unwrap_or_else(|| "x_search".to_owned()),
        ServerToolKind::CodeExecution => state
            .names
            .code_execution
            .clone()
            .unwrap_or_else(|| "code_execution".to_owned()),
        ServerToolKind::ViewImage => {
            if raw_name.is_empty() {
                "view_image".to_owned()
            } else {
                raw_name.clone()
            }
        }
        ServerToolKind::ViewXVideo => {
            if raw_name.is_empty() {
                "view_x_video".to_owned()
            } else {
                raw_name.clone()
            }
        }
        ServerToolKind::Custom => {
            if WEB_SEARCH_SUB_TOOLS.iter().any(|s| *s == raw_name.as_str()) {
                state
                    .names
                    .web_search
                    .clone()
                    .unwrap_or_else(|| "web_search".to_owned())
            } else if X_SEARCH_SUB_TOOLS.iter().any(|s| *s == raw_name.as_str()) {
                state
                    .names
                    .x_search
                    .clone()
                    .unwrap_or_else(|| "x_search".to_owned())
            } else if raw_name == "code_execution" {
                state
                    .names
                    .code_execution
                    .clone()
                    .unwrap_or_else(|| "code_execution".to_owned())
            } else if raw_name.is_empty() {
                "custom_tool".to_owned()
            } else {
                raw_name.clone()
            }
        }
    };

    let input_str = match kind {
        ServerToolKind::Custom => item.input.unwrap_or_default(),
        _ => item.arguments.unwrap_or_default(),
    };

    out.push(StreamPart::ToolInputStart {
        id: item.id.clone(),
        tool_name: tool_name.clone(),
        provider_executed: Some(true),
        dynamic: None,
        title: None,
        provider_metadata: None,
    });
    out.push(StreamPart::ToolInputDelta {
        id: item.id.clone(),
        delta: input_str.clone(),
        provider_metadata: None,
    });
    out.push(StreamPart::ToolInputEnd {
        id: item.id.clone(),
        provider_metadata: None,
    });
    out.push(StreamPart::ToolCall(ToolCallPart {
        tool_call_id: item.id,
        tool_name,
        input: Value::String(input_str),
        provider_executed: Some(true),
        dynamic: None,
        provider_options: None,
    }));
}

fn text_block_id(item_id: &str) -> String {
    format!("text-{item_id}")
}

fn reasoning_block_id(item_id: &str) -> String {
    format!("reasoning-{item_id}")
}

fn xai_item_id_metadata(item_id: &str) -> ProviderMetadata {
    let mut xai = Map::new();
    xai.insert("itemId".into(), json!(item_id));
    let mut outer = ProviderMetadata::new();
    outer.insert("xai".into(), xai);
    outer
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses::wire::{
        FunctionCallItem, MessageContentPart, MessageItem, ResponsesResponse,
    };

    fn state() -> StreamState {
        StreamState::new(vec![], ResolvedToolNames::default(), false)
    }

    #[test]
    fn first_metadata_chunk_emits_response_metadata() {
        let mut s = state();
        let _ = s.start_frames();
        let r = s.on_chunk(
            ResponsesChunk::ResponseCreated {
                response: ResponsesResponse {
                    id: Some("resp_1".into()),
                    model: Some("grok-4.3".into()),
                    ..Default::default()
                },
            },
            None,
        );
        assert!(matches!(r[0], StreamPart::ResponseMetadata(_)));
    }

    #[test]
    fn output_text_delta_opens_text_block() {
        let mut s = state();
        let _ = s.start_frames();
        let r = s.on_chunk(
            ResponsesChunk::OutputTextDelta {
                item_id: "msg_1".into(),
                output_index: 0,
                content_index: 0,
                delta: "hi".into(),
            },
            None,
        );
        assert!(matches!(&r[0], StreamPart::TextStart { id, .. } if id == "text-msg_1"));
        assert!(matches!(&r[1], StreamPart::TextDelta { delta, .. } if delta == "hi"));
    }

    #[test]
    fn function_call_emits_input_start_then_end_with_tool_call() {
        let mut s = state();
        let _ = s.start_frames();
        let added = s.on_chunk(
            ResponsesChunk::OutputItemAdded {
                item: OutputItem::FunctionCall(FunctionCallItem {
                    name: "weather".into(),
                    arguments: String::new(),
                    call_id: "call_1".into(),
                    id: "fc_1".into(),
                }),
                output_index: 0,
            },
            None,
        );
        assert!(matches!(&added[0], StreamPart::ToolInputStart { id, .. } if id == "call_1"));
        let delta = s.on_chunk(
            ResponsesChunk::FunctionCallArgumentsDelta {
                item_id: "fc_1".into(),
                output_index: 0,
                delta: "{".into(),
            },
            None,
        );
        assert!(matches!(&delta[0], StreamPart::ToolInputDelta { delta, .. } if delta == "{"));
        let done = s.on_chunk(
            ResponsesChunk::OutputItemDone {
                item: OutputItem::FunctionCall(FunctionCallItem {
                    name: "weather".into(),
                    arguments: r#"{"city":"NYC"}"#.into(),
                    call_id: "call_1".into(),
                    id: "fc_1".into(),
                }),
                output_index: 0,
            },
            None,
        );
        assert!(matches!(&done[0], StreamPart::ToolInputEnd { .. }));
        assert!(matches!(&done[1], StreamPart::ToolCall(_)));
    }

    #[test]
    fn message_done_emits_text_block_when_no_deltas_seen() {
        let mut s = state();
        let _ = s.start_frames();
        let done = s.on_chunk(
            ResponsesChunk::OutputItemDone {
                item: OutputItem::Message(MessageItem {
                    id: "msg_2".into(),
                    role: Some("assistant".into()),
                    status: Some("completed".into()),
                    content: vec![MessageContentPart {
                        kind: Some("output_text".into()),
                        text: Some("answer".into()),
                        annotations: None,
                    }],
                }),
                output_index: 0,
            },
            None,
        );
        // text-start + text-delta
        assert!(matches!(&done[0], StreamPart::TextStart { .. }));
        assert!(matches!(&done[1], StreamPart::TextDelta { delta, .. } if delta == "answer"));
    }

    #[test]
    fn reasoning_text_delta_opens_reasoning_block_with_metadata() {
        let mut s = state();
        let _ = s.start_frames();
        let frames = s.on_chunk(
            ResponsesChunk::ReasoningTextDelta {
                item_id: "rs_1".into(),
                content_index: 0,
                delta: "think".into(),
            },
            None,
        );
        let StreamPart::ReasoningStart {
            id,
            provider_metadata,
        } = &frames[0]
        else {
            panic!("expected reasoning-start");
        };
        assert_eq!(id, "reasoning-rs_1");
        let pm = provider_metadata.as_ref().unwrap();
        assert_eq!(pm["xai"]["itemId"], "rs_1");
        assert!(matches!(&frames[1], StreamPart::ReasoningDelta { delta, .. } if delta == "think"));
    }

    #[test]
    fn response_done_with_function_call_sets_tool_calls_finish() {
        let mut s = state();
        let _ = s.start_frames();
        s.has_function_call = true;
        let _ = s.on_chunk(
            ResponsesChunk::ResponseCompleted {
                response: ResponsesResponse {
                    status: Some("completed".into()),
                    ..Default::default()
                },
            },
            None,
        );
        let tail = s.flush();
        let StreamPart::Finish { finish_reason, .. } = tail.last().unwrap() else {
            panic!("expected Finish");
        };
        assert_eq!(finish_reason.unified, FinishReasonKind::ToolCalls);
    }

    #[test]
    fn raw_chunks_forwarded_when_enabled() {
        let mut s = StreamState::new(vec![], ResolvedToolNames::default(), true);
        let _ = s.start_frames();
        let r = s.on_chunk(
            ResponsesChunk::ResponseCreated {
                response: ResponsesResponse {
                    id: Some("resp_1".into()),
                    ..Default::default()
                },
            },
            Some(json!({"type": "response.created"})),
        );
        assert!(matches!(&r[0], StreamPart::Raw { .. }));
    }

    #[test]
    fn error_chunk_emits_error_frame() {
        let mut s = state();
        let _ = s.start_frames();
        let r = s.on_chunk(
            ResponsesChunk::Error {
                code: Some("e1".into()),
                message: "boom".into(),
                param: None,
            },
            None,
        );
        assert!(matches!(&r[0], StreamPart::Error { .. }));
    }
}
