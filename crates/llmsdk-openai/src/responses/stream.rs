//! SSE state machine for `POST /v1/responses` (`stream: true`).
//!
//! Mirrors the `doStream` half of
//! `@ai-sdk/openai/src/responses/openai-responses-language-model.ts`.
// Rust guideline compliant 2026-02-21

use std::collections::{BTreeMap, HashMap};

use llmsdk_provider::json::JsonValue;
use llmsdk_provider::language_model::{
    FinishReason, FinishReasonKind, ResponseMetadata, Source, StreamPart, ToolCallPart, ToolResult,
    ToolResultOutput,
};
use llmsdk_provider::shared::{ProviderMetadata, Warning};
use serde_json::{Map, json};

use super::finish_reason::map_finish_reason;
use super::tools::ids;
use super::usage::{ResponsesUsage, convert_usage};
use super::wire::chunk::{AddedItem, ResponsesChunk};
use super::wire::response::{
    Annotation, ApplyPatchCallStatus, MessagePhase, OutputItem, ShellCallStatus,
};

/// State driving the conversion of one Responses-API SSE stream into
/// llmsdk [`StreamPart`]s.
pub struct StreamState {
    initial_warnings: Option<Vec<Warning>>,
    provider_options_name: String,
    store: bool,
    include_raw_chunks: bool,
    web_search_tool_name: Option<String>,
    is_shell_provider_executed: bool,

    has_function_call: bool,
    finish_reason: FinishReason,
    last_usage: Option<ResponsesUsage>,
    service_tier: Option<String>,
    response_id: Option<String>,
    logprobs: Vec<JsonValue>,

    ongoing_tool_calls: BTreeMap<u32, ToolCallAccum>,
    ongoing_annotations: Vec<Annotation>,
    active_message_phase: Option<MessagePhase>,
    active_reasoning: HashMap<String, ReasoningAccum>,
    hosted_tool_search_call_ids: Vec<String>,
    approval_request_id_to_call_id: HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct ToolCallAccum {
    tool_name: String,
    tool_call_id: String,
    kind: ToolCallKind,
}

#[derive(Debug, Clone)]
enum ToolCallKind {
    Function,
    Custom,
    WebSearch,
    Computer,
    CodeInterpreter { container_id: String },
    ImageGeneration,
    FileSearch,
    LocalShell,
    Shell,
    ApplyPatch { has_diff: bool, end_emitted: bool },
    Mcp,
    ToolSearch { hosted: bool },
}

#[derive(Debug, Clone, Default)]
struct ReasoningAccum {
    encrypted_content: Option<String>,
    summary_parts: HashMap<u32, SummaryStatus>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SummaryStatus {
    Active,
    CanConclude,
    Concluded,
}

/// Parameters captured from the caller's request before opening the stream.
pub struct StreamSetup<'a> {
    pub warnings: Vec<Warning>,
    pub provider_options_name: &'a str,
    pub store: bool,
    pub include_raw_chunks: bool,
    pub web_search_tool_name: Option<String>,
    pub is_shell_provider_executed: bool,
}

impl StreamState {
    /// Build new state from request-time setup.
    pub fn new(setup: StreamSetup<'_>) -> Self {
        Self {
            initial_warnings: Some(setup.warnings),
            provider_options_name: setup.provider_options_name.to_owned(),
            store: setup.store,
            include_raw_chunks: setup.include_raw_chunks,
            web_search_tool_name: setup.web_search_tool_name,
            is_shell_provider_executed: setup.is_shell_provider_executed,

            has_function_call: false,
            finish_reason: FinishReason::new(FinishReasonKind::Other),
            last_usage: None,
            service_tier: None,
            response_id: None,
            logprobs: Vec::new(),

            ongoing_tool_calls: BTreeMap::new(),
            ongoing_annotations: Vec::new(),
            active_message_phase: None,
            active_reasoning: HashMap::new(),
            hosted_tool_search_call_ids: Vec::new(),
            approval_request_id_to_call_id: HashMap::new(),
        }
    }

    /// Frames emitted before any chunk is processed (`stream-start`).
    pub fn start_frames(&mut self) -> Vec<StreamPart> {
        let warnings = self.initial_warnings.take().unwrap_or_default();
        vec![StreamPart::StreamStart { warnings }]
    }

    /// Process one SSE chunk; return the parts to emit immediately.
    #[allow(
        clippy::too_many_lines,
        reason = "single switch over 18 chunk variants"
    )]
    pub fn on_chunk(&mut self, chunk: ResponsesChunk, raw: Option<JsonValue>) -> Vec<StreamPart> {
        let mut out = Vec::new();
        if self.include_raw_chunks
            && let Some(raw) = raw
        {
            out.push(StreamPart::Raw { raw_value: raw });
        }

        match chunk {
            ResponsesChunk::Created { response } => {
                self.response_id = Some(response.id.clone());
                if let Some(t) = response.service_tier {
                    self.service_tier = Some(t);
                }
                out.push(StreamPart::ResponseMetadata(ResponseMetadata {
                    id: Some(response.id),
                    timestamp: None,
                    model_id: Some(response.model),
                    headers: None,
                }));
            }
            ResponsesChunk::Completed { response } => {
                self.collect_finish(
                    response
                        .incomplete_details
                        .as_ref()
                        .map(|d| d.reason.as_str()),
                    Some(response.usage),
                    response.service_tier,
                );
            }
            ResponsesChunk::Incomplete { response } => {
                self.collect_finish(
                    response
                        .incomplete_details
                        .as_ref()
                        .map(|d| d.reason.as_str()),
                    Some(response.usage),
                    response.service_tier,
                );
            }
            ResponsesChunk::Failed { response } => {
                let reason = response
                    .incomplete_details
                    .as_ref()
                    .map(|d| d.reason.as_str());
                self.collect_finish(reason, response.usage, response.service_tier);
                if reason.is_none() {
                    self.finish_reason = FinishReason::with_raw(FinishReasonKind::Error, "error");
                }
            }
            ResponsesChunk::OutputItemAdded { output_index, item } => {
                self.on_item_added(output_index, item, &mut out);
            }
            ResponsesChunk::OutputItemDone { output_index, item } => {
                self.on_item_done(output_index, item, &mut out);
            }
            ResponsesChunk::OutputTextDelta {
                item_id,
                delta,
                logprobs,
            } => {
                if let Some(lp) = logprobs {
                    for entry in lp {
                        self.logprobs
                            .push(serde_json::to_value(entry).unwrap_or(JsonValue::Null));
                    }
                }
                out.push(StreamPart::TextDelta {
                    id: item_id,
                    delta,
                    provider_metadata: None,
                });
            }
            ResponsesChunk::AnnotationAdded { annotation } => {
                self.ongoing_annotations.push(annotation.clone());
                self.push_annotation_source(&annotation, &mut out);
            }
            ResponsesChunk::ReasoningSummaryPartAdded {
                item_id,
                summary_index,
            } => {
                self.on_reasoning_summary_added(&item_id, summary_index, &mut out);
            }
            ResponsesChunk::ReasoningSummaryTextDelta {
                item_id,
                summary_index,
                delta,
            } => {
                out.push(StreamPart::ReasoningDelta {
                    id: format!("{item_id}:{summary_index}"),
                    delta,
                    provider_metadata: Some(self.make_pm(json!({"itemId": item_id}))),
                });
            }
            ResponsesChunk::ReasoningSummaryPartDone {
                item_id,
                summary_index,
            } => {
                self.on_reasoning_summary_done(&item_id, summary_index, &mut out);
            }
            ResponsesChunk::FunctionCallArgumentsDelta {
                output_index,
                delta,
                ..
            }
            | ResponsesChunk::CustomToolCallInputDelta {
                output_index,
                delta,
                ..
            } => {
                if let Some(call) = self.ongoing_tool_calls.get(&output_index) {
                    out.push(StreamPart::ToolInputDelta {
                        id: call.tool_call_id.clone(),
                        delta,
                        provider_metadata: None,
                    });
                }
            }
            ResponsesChunk::ImageGenerationPartialImage {
                item_id,
                partial_image_b64,
                ..
            } => {
                // ai-sdk marks these with `preliminary: true`. llmsdk trait
                // has no field for that yet — surface it on provider_metadata.
                let pm = self.make_pm(json!({"preliminary": true}));
                out.push(StreamPart::ToolResult(ToolResult {
                    tool_call_id: item_id,
                    tool_name: ids::IMAGE_GENERATION.into(),
                    output: ToolResultOutput::Json {
                        value: json!({"result": partial_image_b64}),
                        provider_options: None,
                    },
                    provider_metadata: Some(pm),
                }));
            }
            ResponsesChunk::CodeInterpreterCodeDelta {
                output_index,
                delta,
                ..
            } => {
                if let Some(call) = self.ongoing_tool_calls.get(&output_index) {
                    out.push(StreamPart::ToolInputDelta {
                        id: call.tool_call_id.clone(),
                        delta: escape_json_delta(&delta),
                        provider_metadata: None,
                    });
                }
            }
            ResponsesChunk::CodeInterpreterCodeDone {
                output_index, code, ..
            } => {
                if let Some(call) = self.ongoing_tool_calls.get(&output_index) {
                    let id = call.tool_call_id.clone();
                    let container_id = match &call.kind {
                        ToolCallKind::CodeInterpreter { container_id } => container_id.clone(),
                        _ => String::new(),
                    };
                    out.push(StreamPart::ToolInputDelta {
                        id: id.clone(),
                        delta: "\"}".into(),
                        provider_metadata: None,
                    });
                    out.push(StreamPart::ToolInputEnd {
                        id: id.clone(),
                        provider_metadata: None,
                    });
                    out.push(StreamPart::ToolCall(ToolCallPart {
                        tool_call_id: id,
                        tool_name: ids::CODE_INTERPRETER.into(),
                        input: json!({"code": code, "containerId": container_id}),
                        provider_executed: Some(true),
                        dynamic: None,
                        provider_options: None,
                    }));
                }
            }
            ResponsesChunk::ApplyPatchOpDiffDelta {
                output_index,
                delta,
                ..
            } => {
                if let Some(call) = self.ongoing_tool_calls.get_mut(&output_index)
                    && let ToolCallKind::ApplyPatch { has_diff, .. } = &mut call.kind
                {
                    *has_diff = true;
                    let id = call.tool_call_id.clone();
                    out.push(StreamPart::ToolInputDelta {
                        id,
                        delta: escape_json_delta(&delta),
                        provider_metadata: None,
                    });
                }
            }
            ResponsesChunk::ApplyPatchOpDiffDone {
                output_index, diff, ..
            } => {
                if let Some(call) = self.ongoing_tool_calls.get_mut(&output_index)
                    && let ToolCallKind::ApplyPatch {
                        has_diff,
                        end_emitted,
                    } = &mut call.kind
                    && !*end_emitted
                {
                    let id = call.tool_call_id.clone();
                    if !*has_diff {
                        out.push(StreamPart::ToolInputDelta {
                            id: id.clone(),
                            delta: escape_json_delta(&diff),
                            provider_metadata: None,
                        });
                        *has_diff = true;
                    }
                    out.push(StreamPart::ToolInputDelta {
                        id: id.clone(),
                        delta: "\"}}".into(),
                        provider_metadata: None,
                    });
                    out.push(StreamPart::ToolInputEnd {
                        id,
                        provider_metadata: None,
                    });
                    *end_emitted = true;
                }
            }
            ResponsesChunk::Error { error, .. } => {
                self.finish_reason = FinishReason::with_raw(FinishReasonKind::Error, &error.code);
                out.push(StreamPart::Error {
                    error: json!({
                        "type": error.kind,
                        "code": error.code,
                        "message": error.message,
                        "param": error.param,
                    }),
                });
            }
            ResponsesChunk::Unknown => {
                // Already optionally surfaced as `Raw` above.
            }
        }

        out
    }

    /// Frame emitted after the SSE stream closes (`finish`).
    pub fn finish_frame(&self) -> StreamPart {
        let usage = convert_usage(self.last_usage.as_ref());
        let mut openai_meta = Map::new();
        openai_meta.insert(
            "responseId".into(),
            self.response_id
                .as_ref()
                .map_or(JsonValue::Null, |s| json!(s)),
        );
        if !self.logprobs.is_empty() {
            openai_meta.insert("logprobs".into(), json!(self.logprobs));
        }
        if let Some(t) = &self.service_tier {
            openai_meta.insert("serviceTier".into(), json!(t));
        }
        let mut pm = ProviderMetadata::new();
        pm.insert(self.provider_options_name.clone(), openai_meta);
        StreamPart::Finish {
            usage,
            finish_reason: self.finish_reason.clone(),
            provider_metadata: Some(pm),
        }
    }

    fn collect_finish(
        &mut self,
        reason: Option<&str>,
        usage: Option<ResponsesUsage>,
        service_tier: Option<String>,
    ) {
        self.finish_reason = map_finish_reason(reason, self.has_function_call);
        if let Some(u) = usage {
            self.last_usage = Some(u);
        }
        if let Some(t) = service_tier {
            self.service_tier = Some(t);
        }
    }

    fn make_pm(&self, body: JsonValue) -> ProviderMetadata {
        let mut pm = ProviderMetadata::new();
        let obj = body.as_object().cloned().unwrap_or_default();
        pm.insert(self.provider_options_name.clone(), obj);
        pm
    }

    fn push_annotation_source(&self, annotation: &Annotation, out: &mut Vec<StreamPart>) {
        match annotation {
            Annotation::UrlCitation { url, title, .. } => {
                out.push(StreamPart::Source(Source::Url {
                    id: generated_id("src"),
                    url: url.clone(),
                    title: Some(title.clone()),
                    provider_metadata: None,
                }))
            }
            Annotation::FileCitation {
                file_id,
                filename,
                index,
            } => {
                out.push(StreamPart::Source(Source::Document {
                    id: generated_id("src"),
                    media_type: "text/plain".into(),
                    title: filename.clone(),
                    filename: Some(filename.clone()),
                    provider_metadata: Some(self.make_pm(
                        json!({"type": "file_citation", "fileId": file_id, "index": index}),
                    )),
                }))
            }
            Annotation::ContainerFileCitation {
                container_id,
                file_id,
                filename,
                ..
            } => out.push(StreamPart::Source(Source::Document {
                id: generated_id("src"),
                media_type: "text/plain".into(),
                title: filename.clone(),
                filename: Some(filename.clone()),
                provider_metadata: Some(self.make_pm(json!({
                    "type": "container_file_citation",
                    "fileId": file_id,
                    "containerId": container_id
                }))),
            })),
            Annotation::FilePath { file_id, index } => {
                out.push(StreamPart::Source(Source::Document {
                    id: generated_id("src"),
                    media_type: "application/octet-stream".into(),
                    title: file_id.clone(),
                    filename: Some(file_id.clone()),
                    provider_metadata: Some(
                        self.make_pm(
                            json!({"type": "file_path", "fileId": file_id, "index": index}),
                        ),
                    ),
                }))
            }
        }
    }

    fn on_reasoning_summary_added(
        &mut self,
        item_id: &str,
        summary_index: u32,
        out: &mut Vec<StreamPart>,
    ) {
        if summary_index == 0 {
            // Already emitted by output_item.added.
            return;
        }
        // Snapshot what we need from the active reasoning state before
        // calling `make_pm` (which borrows &self immutably).
        let (to_conclude, encrypted): (Vec<u32>, Option<String>) = {
            let part = self
                .active_reasoning
                .entry(item_id.to_string())
                .or_default();
            let to_conclude: Vec<u32> = part
                .summary_parts
                .iter()
                .filter_map(|(k, v)| (*v == SummaryStatus::CanConclude).then_some(*k))
                .collect();
            for idx in &to_conclude {
                part.summary_parts.insert(*idx, SummaryStatus::Concluded);
            }
            part.summary_parts
                .insert(summary_index, SummaryStatus::Active);
            (to_conclude, part.encrypted_content.clone())
        };
        for idx in to_conclude {
            let pm = self.make_pm(json!({"itemId": item_id}));
            out.push(StreamPart::ReasoningEnd {
                id: format!("{item_id}:{idx}"),
                provider_metadata: Some(pm),
            });
        }
        let pm = self.make_pm(json!({
            "itemId": item_id,
            "reasoningEncryptedContent": encrypted,
        }));
        out.push(StreamPart::ReasoningStart {
            id: format!("{item_id}:{summary_index}"),
            provider_metadata: Some(pm),
        });
    }

    fn on_reasoning_summary_done(
        &mut self,
        item_id: &str,
        summary_index: u32,
        out: &mut Vec<StreamPart>,
    ) {
        let status = if self.store {
            SummaryStatus::Concluded
        } else {
            SummaryStatus::CanConclude
        };
        self.active_reasoning
            .entry(item_id.to_string())
            .or_default()
            .summary_parts
            .insert(summary_index, status);
        if self.store {
            let pm = self.make_pm(json!({"itemId": item_id}));
            out.push(StreamPart::ReasoningEnd {
                id: format!("{item_id}:{summary_index}"),
                provider_metadata: Some(pm),
            });
        }
    }

    #[allow(clippy::too_many_lines, reason = "added-item switch over 15 variants")]
    fn on_item_added(&mut self, output_index: u32, item: AddedItem, out: &mut Vec<StreamPart>) {
        match item {
            AddedItem::Message { id, phase } => {
                self.ongoing_annotations.clear();
                self.active_message_phase = phase;
                let body = if let Some(p) = phase {
                    json!({ "itemId": id, "phase": p })
                } else {
                    json!({ "itemId": id })
                };
                out.push(StreamPart::TextStart {
                    id,
                    provider_metadata: Some(self.make_pm(body)),
                });
            }
            AddedItem::Reasoning {
                id,
                encrypted_content,
            } => {
                let mut accum = ReasoningAccum::default();
                accum.encrypted_content = encrypted_content.clone();
                accum.summary_parts.insert(0, SummaryStatus::Active);
                self.active_reasoning.insert(id.clone(), accum);
                let pm = self.make_pm(json!({
                    "itemId": id,
                    "reasoningEncryptedContent": encrypted_content,
                }));
                out.push(StreamPart::ReasoningStart {
                    id: format!("{id}:0"),
                    provider_metadata: Some(pm),
                });
            }
            AddedItem::FunctionCall { call_id, name, .. } => {
                self.ongoing_tool_calls.insert(
                    output_index,
                    ToolCallAccum {
                        tool_name: name.clone(),
                        tool_call_id: call_id.clone(),
                        kind: ToolCallKind::Function,
                    },
                );
                out.push(StreamPart::ToolInputStart {
                    id: call_id,
                    tool_name: name,
                    provider_executed: None,
                    dynamic: None,
                    title: None,
                    provider_metadata: None,
                });
            }
            AddedItem::CustomToolCall { call_id, name, .. } => {
                self.ongoing_tool_calls.insert(
                    output_index,
                    ToolCallAccum {
                        tool_name: name.clone(),
                        tool_call_id: call_id.clone(),
                        kind: ToolCallKind::Custom,
                    },
                );
                out.push(StreamPart::ToolInputStart {
                    id: call_id,
                    tool_name: name,
                    provider_executed: None,
                    dynamic: None,
                    title: None,
                    provider_metadata: None,
                });
            }
            AddedItem::WebSearchCall { id, .. } => {
                let tool_name = self
                    .web_search_tool_name
                    .clone()
                    .unwrap_or_else(|| "web_search".into());
                self.ongoing_tool_calls.insert(
                    output_index,
                    ToolCallAccum {
                        tool_name: tool_name.clone(),
                        tool_call_id: id.clone(),
                        kind: ToolCallKind::WebSearch,
                    },
                );
                out.push(StreamPart::ToolInputStart {
                    id: id.clone(),
                    tool_name: tool_name.clone(),
                    provider_executed: Some(true),
                    dynamic: None,
                    title: None,
                    provider_metadata: None,
                });
                out.push(StreamPart::ToolInputEnd {
                    id: id.clone(),
                    provider_metadata: None,
                });
                out.push(StreamPart::ToolCall(ToolCallPart {
                    tool_call_id: id,
                    tool_name,
                    input: json!({}),
                    provider_executed: Some(true),
                    dynamic: None,
                    provider_options: None,
                }));
            }
            AddedItem::ComputerCall { id, .. } => {
                self.ongoing_tool_calls.insert(
                    output_index,
                    ToolCallAccum {
                        tool_name: "computer_use".into(),
                        tool_call_id: id.clone(),
                        kind: ToolCallKind::Computer,
                    },
                );
                out.push(StreamPart::ToolInputStart {
                    id,
                    tool_name: "computer_use".into(),
                    provider_executed: Some(true),
                    dynamic: None,
                    title: None,
                    provider_metadata: None,
                });
            }
            AddedItem::FileSearchCall { id } => {
                self.ongoing_tool_calls.insert(
                    output_index,
                    ToolCallAccum {
                        tool_name: ids::FILE_SEARCH.into(),
                        tool_call_id: id.clone(),
                        kind: ToolCallKind::FileSearch,
                    },
                );
                out.push(StreamPart::ToolCall(ToolCallPart {
                    tool_call_id: id,
                    tool_name: ids::FILE_SEARCH.into(),
                    input: json!({}),
                    provider_executed: Some(true),
                    dynamic: None,
                    provider_options: None,
                }));
            }
            AddedItem::ImageGenerationCall { id } => {
                self.ongoing_tool_calls.insert(
                    output_index,
                    ToolCallAccum {
                        tool_name: ids::IMAGE_GENERATION.into(),
                        tool_call_id: id.clone(),
                        kind: ToolCallKind::ImageGeneration,
                    },
                );
                out.push(StreamPart::ToolCall(ToolCallPart {
                    tool_call_id: id,
                    tool_name: ids::IMAGE_GENERATION.into(),
                    input: json!({}),
                    provider_executed: Some(true),
                    dynamic: None,
                    provider_options: None,
                }));
            }
            AddedItem::CodeInterpreterCall {
                id, container_id, ..
            } => {
                self.ongoing_tool_calls.insert(
                    output_index,
                    ToolCallAccum {
                        tool_name: ids::CODE_INTERPRETER.into(),
                        tool_call_id: id.clone(),
                        kind: ToolCallKind::CodeInterpreter {
                            container_id: container_id.clone(),
                        },
                    },
                );
                out.push(StreamPart::ToolInputStart {
                    id: id.clone(),
                    tool_name: ids::CODE_INTERPRETER.into(),
                    provider_executed: Some(true),
                    dynamic: None,
                    title: None,
                    provider_metadata: None,
                });
                out.push(StreamPart::ToolInputDelta {
                    id,
                    delta: format!("{{\"containerId\":\"{container_id}\",\"code\":\""),
                    provider_metadata: None,
                });
            }
            AddedItem::ApplyPatchCall {
                call_id, operation, ..
            } => {
                let is_delete = matches!(
                    operation,
                    super::tools::apply_patch::Operation::DeleteFile { .. }
                );
                self.ongoing_tool_calls.insert(
                    output_index,
                    ToolCallAccum {
                        tool_name: ids::APPLY_PATCH.into(),
                        tool_call_id: call_id.clone(),
                        kind: ToolCallKind::ApplyPatch {
                            has_diff: is_delete,
                            end_emitted: is_delete,
                        },
                    },
                );
                out.push(StreamPart::ToolInputStart {
                    id: call_id.clone(),
                    tool_name: ids::APPLY_PATCH.into(),
                    provider_executed: None,
                    dynamic: None,
                    title: None,
                    provider_metadata: None,
                });
                if is_delete {
                    let input = json!({"callId": call_id, "operation": operation});
                    out.push(StreamPart::ToolInputDelta {
                        id: call_id.clone(),
                        delta: serde_json::to_string(&input).unwrap_or_default(),
                        provider_metadata: None,
                    });
                    out.push(StreamPart::ToolInputEnd {
                        id: call_id,
                        provider_metadata: None,
                    });
                } else {
                    let op_type = match &operation {
                        super::tools::apply_patch::Operation::CreateFile { .. } => "create_file",
                        super::tools::apply_patch::Operation::UpdateFile { .. } => "update_file",
                        super::tools::apply_patch::Operation::DeleteFile { .. } => "delete_file",
                    };
                    let path = match &operation {
                        super::tools::apply_patch::Operation::CreateFile { path, .. } => path,
                        super::tools::apply_patch::Operation::UpdateFile { path, .. } => path,
                        super::tools::apply_patch::Operation::DeleteFile { path } => path,
                    };
                    out.push(StreamPart::ToolInputDelta {
                        id: call_id,
                        delta: format!(
                            "{{\"callId\":\"{}\",\"operation\":{{\"type\":\"{}\",\"path\":\"{}\",\"diff\":\"",
                            escape_json_delta(
                                self.ongoing_tool_calls
                                    .values()
                                    .last()
                                    .map(|c| c.tool_call_id.as_str())
                                    .unwrap_or(""),
                            ),
                            escape_json_delta(op_type),
                            escape_json_delta(path),
                        ),
                        provider_metadata: None,
                    });
                }
            }
            AddedItem::ShellCall { call_id, .. } => {
                self.ongoing_tool_calls.insert(
                    output_index,
                    ToolCallAccum {
                        tool_name: ids::SHELL.into(),
                        tool_call_id: call_id,
                        kind: ToolCallKind::Shell,
                    },
                );
            }
            AddedItem::ToolSearchCall { id, execution, .. } => {
                let hosted = matches!(execution, super::tools::tool_search::Execution::Server);
                self.ongoing_tool_calls.insert(
                    output_index,
                    ToolCallAccum {
                        tool_name: ids::TOOL_SEARCH.into(),
                        tool_call_id: id.clone(),
                        kind: ToolCallKind::ToolSearch { hosted },
                    },
                );
                if hosted {
                    out.push(StreamPart::ToolInputStart {
                        id,
                        tool_name: ids::TOOL_SEARCH.into(),
                        provider_executed: Some(true),
                        dynamic: None,
                        title: None,
                        provider_metadata: None,
                    });
                }
            }
            AddedItem::McpCall {
                id,
                approval_request_id,
                ..
            } => {
                self.ongoing_tool_calls.insert(
                    output_index,
                    ToolCallAccum {
                        tool_name: "mcp.unknown".into(),
                        tool_call_id: id.clone(),
                        kind: ToolCallKind::Mcp,
                    },
                );
                if let Some(req_id) = approval_request_id {
                    self.approval_request_id_to_call_id.insert(req_id, id);
                }
            }
            // The following added-item types are handled entirely on done.
            AddedItem::McpListTools { .. }
            | AddedItem::McpApprovalRequest { .. }
            | AddedItem::Compaction { .. }
            | AddedItem::ShellCallOutput { .. }
            | AddedItem::ToolSearchOutput { .. }
            | AddedItem::Unknown => {}
        }
    }

    #[allow(clippy::too_many_lines, reason = "done-item switch over 19 variants")]
    fn on_item_done(&mut self, output_index: u32, item: OutputItem, out: &mut Vec<StreamPart>) {
        match item {
            OutputItem::Message(m) => {
                let phase = m.phase.or(self.active_message_phase);
                self.active_message_phase = None;
                let mut body = Map::new();
                body.insert("itemId".into(), json!(m.id));
                if let Some(p) = phase {
                    body.insert("phase".into(), json!(p));
                }
                if !self.ongoing_annotations.is_empty() {
                    body.insert(
                        "annotations".into(),
                        serde_json::to_value(&self.ongoing_annotations).unwrap_or(JsonValue::Null),
                    );
                }
                out.push(StreamPart::TextEnd {
                    id: m.id,
                    provider_metadata: Some(self.make_pm(JsonValue::Object(body))),
                });
            }
            OutputItem::FunctionCall(f) => {
                self.ongoing_tool_calls.remove(&output_index);
                self.has_function_call = true;
                out.push(StreamPart::ToolInputEnd {
                    id: f.call_id.clone(),
                    provider_metadata: f
                        .namespace
                        .as_ref()
                        .map(|ns| self.make_pm(json!({"namespace": ns}))),
                });
                let mut po_body = Map::new();
                po_body.insert("itemId".into(), json!(f.id));
                if let Some(ns) = f.namespace {
                    po_body.insert("namespace".into(), json!(ns));
                }
                let mut po = std::collections::HashMap::new();
                po.insert(self.provider_options_name.clone(), po_body);
                out.push(StreamPart::ToolCall(ToolCallPart {
                    tool_call_id: f.call_id,
                    tool_name: f.name,
                    input: super::parse_response::parse_args(&f.arguments),
                    provider_executed: None,
                    dynamic: None,
                    provider_options: Some(po),
                }));
            }
            OutputItem::CustomToolCall(c) => {
                self.ongoing_tool_calls.remove(&output_index);
                self.has_function_call = true;
                out.push(StreamPart::ToolInputEnd {
                    id: c.call_id.clone(),
                    provider_metadata: None,
                });
                let mut po = std::collections::HashMap::new();
                let mut body = Map::new();
                body.insert("itemId".into(), json!(c.id));
                po.insert(self.provider_options_name.clone(), body);
                out.push(StreamPart::ToolCall(ToolCallPart {
                    tool_call_id: c.call_id,
                    tool_name: c.name,
                    input: JsonValue::String(c.input),
                    provider_executed: None,
                    dynamic: None,
                    provider_options: Some(po),
                }));
            }
            OutputItem::WebSearchCall(w) => {
                self.ongoing_tool_calls.remove(&output_index);
                let tool_name = self
                    .web_search_tool_name
                    .clone()
                    .unwrap_or_else(|| "web_search".into());
                let value = w
                    .action
                    .as_ref()
                    .map(|a| serde_json::to_value(a).unwrap_or(JsonValue::Null))
                    .unwrap_or(json!({}));
                out.push(StreamPart::ToolResult(ToolResult {
                    tool_call_id: w.id,
                    tool_name,
                    output: ToolResultOutput::Json {
                        value,
                        provider_options: None,
                    },
                    provider_metadata: None,
                }));
            }
            OutputItem::ComputerCall(c) => {
                self.ongoing_tool_calls.remove(&output_index);
                out.push(StreamPart::ToolInputEnd {
                    id: c.id.clone(),
                    provider_metadata: None,
                });
                out.push(StreamPart::ToolCall(ToolCallPart {
                    tool_call_id: c.id.clone(),
                    tool_name: "computer_use".into(),
                    input: JsonValue::String(String::new()),
                    provider_executed: Some(true),
                    dynamic: None,
                    provider_options: None,
                }));
                out.push(StreamPart::ToolResult(ToolResult {
                    tool_call_id: c.id,
                    tool_name: "computer_use".into(),
                    output: ToolResultOutput::Json {
                        value: json!({
                            "type": "computer_use_tool_result",
                            "status": c.status.unwrap_or_else(|| "completed".into()),
                        }),
                        provider_options: None,
                    },
                    provider_metadata: None,
                }));
            }
            OutputItem::FileSearchCall(f) => {
                self.ongoing_tool_calls.remove(&output_index);
                out.push(StreamPart::ToolResult(ToolResult {
                    tool_call_id: f.id,
                    tool_name: ids::FILE_SEARCH.into(),
                    output: ToolResultOutput::Json {
                        value: json!({
                            "queries": f.queries,
                            "results": f.results,
                        }),
                        provider_options: None,
                    },
                    provider_metadata: None,
                }));
            }
            OutputItem::CodeInterpreterCall(c) => {
                self.ongoing_tool_calls.remove(&output_index);
                out.push(StreamPart::ToolResult(ToolResult {
                    tool_call_id: c.id,
                    tool_name: ids::CODE_INTERPRETER.into(),
                    output: ToolResultOutput::Json {
                        value: json!({ "outputs": c.outputs }),
                        provider_options: None,
                    },
                    provider_metadata: None,
                }));
            }
            OutputItem::ImageGenerationCall(i) => {
                out.push(StreamPart::ToolResult(ToolResult {
                    tool_call_id: i.id,
                    tool_name: ids::IMAGE_GENERATION.into(),
                    output: ToolResultOutput::Json {
                        value: json!({"result": i.result}),
                        provider_options: None,
                    },
                    provider_metadata: None,
                }));
            }
            OutputItem::LocalShellCall(l) => {
                self.ongoing_tool_calls.remove(&output_index);
                let mut body = Map::new();
                body.insert("itemId".into(), json!(l.id));
                let mut po = std::collections::HashMap::new();
                po.insert(self.provider_options_name.clone(), body);
                out.push(StreamPart::ToolCall(ToolCallPart {
                    tool_call_id: l.call_id,
                    tool_name: ids::LOCAL_SHELL.into(),
                    input: json!({"action": l.action}),
                    provider_executed: None,
                    dynamic: None,
                    provider_options: Some(po),
                }));
            }
            OutputItem::ShellCall(s) => {
                self.ongoing_tool_calls.remove(&output_index);
                let mut body = Map::new();
                body.insert("itemId".into(), json!(s.id));
                let mut po = std::collections::HashMap::new();
                po.insert(self.provider_options_name.clone(), body);
                out.push(StreamPart::ToolCall(ToolCallPart {
                    tool_call_id: s.call_id,
                    tool_name: ids::SHELL.into(),
                    input: json!({"action": {"commands": s.action.commands}}),
                    provider_executed: self.is_shell_provider_executed.then_some(true),
                    dynamic: None,
                    provider_options: Some(po),
                }));
            }
            OutputItem::ShellCallOutput(o) => {
                out.push(StreamPart::ToolResult(ToolResult {
                    tool_call_id: o.call_id,
                    tool_name: ids::SHELL.into(),
                    output: ToolResultOutput::Json {
                        value: serde_json::to_value(&o.output).unwrap_or(JsonValue::Null),
                        provider_options: None,
                    },
                    provider_metadata: None,
                }));
            }
            OutputItem::McpCall(m) => {
                self.ongoing_tool_calls.remove(&output_index);
                let tool_name = format!("mcp.{}", m.name);
                let call_id = m
                    .approval_request_id
                    .as_deref()
                    .and_then(|r| self.approval_request_id_to_call_id.get(r))
                    .cloned()
                    .unwrap_or_else(|| m.id.clone());
                out.push(StreamPart::ToolCall(ToolCallPart {
                    tool_call_id: call_id.clone(),
                    tool_name: tool_name.clone(),
                    input: JsonValue::String(m.arguments.clone()),
                    provider_executed: Some(true),
                    dynamic: Some(true),
                    provider_options: None,
                }));
                let mut payload = serde_json::Map::new();
                payload.insert("type".into(), json!("call"));
                payload.insert("serverLabel".into(), json!(m.server_label));
                payload.insert("name".into(), json!(m.name));
                payload.insert("arguments".into(), json!(m.arguments));
                if let Some(o) = m.output {
                    payload.insert("output".into(), json!(o));
                }
                if let Some(e) = m.error {
                    payload.insert("error".into(), e);
                }
                let mut body = Map::new();
                body.insert("itemId".into(), json!(m.id));
                let mut pm = ProviderMetadata::new();
                pm.insert(self.provider_options_name.clone(), body);
                out.push(StreamPart::ToolResult(ToolResult {
                    tool_call_id: call_id,
                    tool_name,
                    output: ToolResultOutput::Json {
                        value: JsonValue::Object(payload),
                        provider_options: None,
                    },
                    provider_metadata: Some(pm),
                }));
            }
            OutputItem::McpListTools(_) => {
                self.ongoing_tool_calls.remove(&output_index);
            }
            OutputItem::McpApprovalRequest(a) => {
                self.ongoing_tool_calls.remove(&output_index);
                let req_id = a
                    .approval_request_id
                    .clone()
                    .unwrap_or_else(|| a.id.clone());
                let dummy_id = generated_id("mcp_call");
                self.approval_request_id_to_call_id
                    .insert(req_id.clone(), dummy_id.clone());
                let tool_name = format!("mcp.{}", a.name);
                out.push(StreamPart::ToolCall(ToolCallPart {
                    tool_call_id: dummy_id.clone(),
                    tool_name: tool_name.clone(),
                    input: JsonValue::String(a.arguments),
                    provider_executed: Some(true),
                    dynamic: Some(true),
                    provider_options: None,
                }));
                out.push(StreamPart::ToolApprovalRequest(
                    llmsdk_provider::language_model::ToolApprovalRequest {
                        approval_id: req_id,
                        tool_call: ToolCallPart {
                            tool_call_id: dummy_id,
                            tool_name,
                            input: JsonValue::String(String::new()),
                            provider_executed: Some(true),
                            dynamic: Some(true),
                            provider_options: None,
                        },
                        provider_metadata: None,
                    },
                ));
            }
            OutputItem::ApplyPatchCall(a) => {
                if let Some(call) = self.ongoing_tool_calls.get_mut(&output_index)
                    && let ToolCallKind::ApplyPatch {
                        has_diff,
                        end_emitted,
                    } = &mut call.kind
                    && !*end_emitted
                    && !matches!(
                        a.operation,
                        super::tools::apply_patch::Operation::DeleteFile { .. }
                    )
                {
                    let id = call.tool_call_id.clone();
                    if !*has_diff {
                        out.push(StreamPart::ToolInputDelta {
                            id: id.clone(),
                            delta: escape_json_delta(match &a.operation {
                                super::tools::apply_patch::Operation::CreateFile {
                                    diff, ..
                                }
                                | super::tools::apply_patch::Operation::UpdateFile {
                                    diff, ..
                                } => diff,
                                super::tools::apply_patch::Operation::DeleteFile { .. } => "",
                            }),
                            provider_metadata: None,
                        });
                    }
                    out.push(StreamPart::ToolInputDelta {
                        id: id.clone(),
                        delta: "\"}}".into(),
                        provider_metadata: None,
                    });
                    out.push(StreamPart::ToolInputEnd {
                        id,
                        provider_metadata: None,
                    });
                    *end_emitted = true;
                }
                if a.status == ApplyPatchCallStatus::Completed
                    && let Some(call) = self.ongoing_tool_calls.remove(&output_index)
                {
                    let mut body = Map::new();
                    body.insert("itemId".into(), json!(a.id));
                    let mut po = std::collections::HashMap::new();
                    po.insert(self.provider_options_name.clone(), body);
                    out.push(StreamPart::ToolCall(ToolCallPart {
                        tool_call_id: call.tool_call_id,
                        tool_name: ids::APPLY_PATCH.into(),
                        input: json!({
                            "callId": a.call_id,
                            "operation": a.operation,
                        }),
                        provider_executed: None,
                        dynamic: None,
                        provider_options: Some(po),
                    }));
                } else {
                    self.ongoing_tool_calls.remove(&output_index);
                }
            }
            OutputItem::Compaction(c) => {
                let mut body = Map::new();
                body.insert("type".into(), json!("compaction"));
                body.insert("itemId".into(), json!(c.id));
                body.insert("encryptedContent".into(), json!(c.encrypted_content));
                out.push(StreamPart::Custom {
                    kind: "openai.compaction".into(),
                    provider_metadata: Some(self.make_pm(JsonValue::Object(body))),
                });
            }
            OutputItem::ToolSearchCall(t) => {
                let call = self.ongoing_tool_calls.remove(&output_index);
                let hosted = matches!(t.execution, super::tools::tool_search::Execution::Server);
                let tool_call_id = if hosted {
                    call.as_ref()
                        .map(|c| c.tool_call_id.clone())
                        .unwrap_or_else(|| t.id.clone())
                } else {
                    t.call_id.clone().unwrap_or_else(|| t.id.clone())
                };
                if hosted {
                    self.hosted_tool_search_call_ids.push(tool_call_id.clone());
                } else {
                    out.push(StreamPart::ToolInputStart {
                        id: tool_call_id.clone(),
                        tool_name: ids::TOOL_SEARCH.into(),
                        provider_executed: None,
                        dynamic: None,
                        title: None,
                        provider_metadata: None,
                    });
                }
                out.push(StreamPart::ToolInputEnd {
                    id: tool_call_id.clone(),
                    provider_metadata: None,
                });
                let mut po_body = Map::new();
                po_body.insert("itemId".into(), json!(t.id));
                let mut po = std::collections::HashMap::new();
                po.insert(self.provider_options_name.clone(), po_body);
                out.push(StreamPart::ToolCall(ToolCallPart {
                    tool_call_id,
                    tool_name: ids::TOOL_SEARCH.into(),
                    input: json!({
                        "arguments": t.arguments,
                        "call_id": if hosted { JsonValue::Null } else { json!(t.call_id) },
                    }),
                    provider_executed: hosted.then_some(true),
                    dynamic: None,
                    provider_options: Some(po),
                }));
            }
            OutputItem::ToolSearchOutput(t) => {
                let tool_call_id = t
                    .call_id
                    .clone()
                    .or_else(|| {
                        (!self.hosted_tool_search_call_ids.is_empty())
                            .then(|| self.hosted_tool_search_call_ids.remove(0))
                    })
                    .unwrap_or_else(|| t.id.clone());
                let mut po_body = Map::new();
                po_body.insert("itemId".into(), json!(t.id));
                let mut pm = ProviderMetadata::new();
                pm.insert(self.provider_options_name.clone(), po_body);
                out.push(StreamPart::ToolResult(ToolResult {
                    tool_call_id,
                    tool_name: ids::TOOL_SEARCH.into(),
                    output: ToolResultOutput::Json {
                        value: json!({"tools": t.tools}),
                        provider_options: None,
                    },
                    provider_metadata: Some(pm),
                }));
            }
            OutputItem::Reasoning(r) => {
                // Conclude all active/can-conclude parts.
                let entry = self.active_reasoning.remove(&r.id);
                if let Some(part) = entry {
                    for (idx, status) in part.summary_parts {
                        if status == SummaryStatus::Active || status == SummaryStatus::CanConclude {
                            out.push(StreamPart::ReasoningEnd {
                                id: format!("{}:{}", r.id, idx),
                                provider_metadata: Some(self.make_pm(json!({
                                    "itemId": r.id,
                                    "reasoningEncryptedContent": r.encrypted_content,
                                }))),
                            });
                        }
                    }
                }
            }
            OutputItem::Unknown => {}
        }
    }
}

/// Escape a string for embedding inside a JSON string literal (no surrounding quotes).
fn escape_json_delta(s: &str) -> String {
    let quoted = serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into());
    quoted
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .map(str::to_owned)
        .unwrap_or_default()
}

fn generated_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    format!("{prefix}_{nanos}_{n}")
}

// Suppress unused import warning when no chunk uses ShellCallStatus directly.
#[allow(dead_code)]
fn _shell_status_kept(_: ShellCallStatus) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> StreamState {
        StreamState::new(StreamSetup {
            warnings: vec![],
            provider_options_name: "openai",
            store: true,
            include_raw_chunks: false,
            web_search_tool_name: None,
            is_shell_provider_executed: false,
        })
    }

    #[test]
    fn text_delta_passes_through() {
        let mut s = setup();
        let _ = s.start_frames();
        let chunk = ResponsesChunk::OutputTextDelta {
            item_id: "msg_1".into(),
            delta: "hi".into(),
            logprobs: None,
        };
        let out = s.on_chunk(chunk, None);
        assert!(matches!(&out[0], StreamPart::TextDelta { delta, .. } if delta == "hi"));
    }

    #[test]
    fn created_emits_response_metadata_and_captures_id() {
        let mut s = setup();
        let chunk = ResponsesChunk::Created {
            response: super::super::wire::chunk::CreatedSnapshot {
                id: "resp_1".into(),
                created_at: 1.0,
                model: "gpt-5".into(),
                service_tier: Some("flex".into()),
            },
        };
        let out = s.on_chunk(chunk, None);
        assert!(matches!(out[0], StreamPart::ResponseMetadata(_)));
        assert_eq!(s.response_id.as_deref(), Some("resp_1"));
        assert_eq!(s.service_tier.as_deref(), Some("flex"));
    }

    #[test]
    fn function_call_arguments_delta_flows_through_ongoing_map() {
        let mut s = setup();
        s.on_chunk(
            ResponsesChunk::OutputItemAdded {
                output_index: 0,
                item: AddedItem::FunctionCall {
                    id: "fc_1".into(),
                    call_id: "call_x".into(),
                    name: "weather".into(),
                    arguments: String::new(),
                    namespace: None,
                },
            },
            None,
        );
        let out = s.on_chunk(
            ResponsesChunk::FunctionCallArgumentsDelta {
                item_id: "fc_1".into(),
                output_index: 0,
                delta: "{\"city".into(),
            },
            None,
        );
        assert!(
            matches!(&out[0], StreamPart::ToolInputDelta { id, delta, .. } if id == "call_x" && delta == "{\"city")
        );
    }

    #[test]
    fn output_item_done_function_call_emits_tool_call_and_end() {
        let mut s = setup();
        s.on_chunk(
            ResponsesChunk::OutputItemAdded {
                output_index: 0,
                item: AddedItem::FunctionCall {
                    id: "fc_1".into(),
                    call_id: "call_x".into(),
                    name: "weather".into(),
                    arguments: String::new(),
                    namespace: None,
                },
            },
            None,
        );
        let item: OutputItem = serde_json::from_value(serde_json::json!({
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_x",
            "name": "weather",
            "arguments": "{\"city\":\"NYC\"}"
        }))
        .unwrap();
        let out = s.on_chunk(
            ResponsesChunk::OutputItemDone {
                output_index: 0,
                item,
            },
            None,
        );
        assert!(matches!(out[0], StreamPart::ToolInputEnd { .. }));
        assert!(matches!(out[1], StreamPart::ToolCall(_)));
        assert!(s.has_function_call);
    }

    #[test]
    fn completed_chunk_collects_finish_reason_and_usage() {
        let mut s = setup();
        let chunk = ResponsesChunk::Completed {
            response: super::super::wire::chunk::FinishedSnapshot {
                incomplete_details: None,
                usage: ResponsesUsage {
                    input_tokens: 7,
                    output_tokens: 3,
                    ..Default::default()
                },
                service_tier: None,
            },
        };
        let _ = s.on_chunk(chunk, None);
        let finish = s.finish_frame();
        let StreamPart::Finish {
            usage,
            finish_reason,
            ..
        } = finish
        else {
            panic!("expected finish");
        };
        assert_eq!(finish_reason.unified, FinishReasonKind::Stop);
        assert_eq!(usage.input_tokens.total, Some(7));
    }

    #[test]
    fn error_chunk_emits_error_part_and_sets_error_reason() {
        let mut s = setup();
        let chunk = ResponsesChunk::Error {
            sequence_number: 1,
            error: super::super::wire::chunk::ErrorChunk {
                kind: "rate_limit".into(),
                code: "rate_limit_exceeded".into(),
                message: "slow".into(),
                param: None,
            },
        };
        let out = s.on_chunk(chunk, None);
        assert!(matches!(&out[0], StreamPart::Error { .. }));
        assert_eq!(s.finish_reason.unified, FinishReasonKind::Error);
    }

    #[test]
    fn annotation_added_url_citation_emits_source() {
        let mut s = setup();
        let chunk = ResponsesChunk::AnnotationAdded {
            annotation: Annotation::UrlCitation {
                start_index: 0,
                end_index: 1,
                url: "https://x".into(),
                title: "X".into(),
            },
        };
        let out = s.on_chunk(chunk, None);
        assert!(matches!(&out[0], StreamPart::Source(Source::Url { .. })));
        assert_eq!(s.ongoing_annotations.len(), 1);
    }

    #[test]
    fn raw_chunks_emitted_when_enabled() {
        let mut s = StreamState::new(StreamSetup {
            warnings: vec![],
            provider_options_name: "openai",
            store: true,
            include_raw_chunks: true,
            web_search_tool_name: None,
            is_shell_provider_executed: false,
        });
        let raw = serde_json::json!({"type": "response.future_event"});
        let out = s.on_chunk(ResponsesChunk::Unknown, Some(raw));
        assert!(matches!(out[0], StreamPart::Raw { .. }));
    }
}
