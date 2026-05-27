//! Streaming state machine: `Anthropic` SSE events → [`StreamPart`]s.
//!
//! Mirrors the `TransformStream` body in `anthropic-language-model.ts`'s
//! `doStream`. Unlike `OpenAI`, `Anthropic`'s wire protocol is **event-typed**
//! with explicit `content_block_start` / `content_block_stop` markers, so
//! there is no inference to do at block boundaries.
//!
//! # Block model
//!
//! `Anthropic` indexes content blocks by an integer. A block is either a
//! `text` block (streamed via `text_delta`) or a `tool_use` block
//! (streamed via `input_json_delta`). Each block opens with
//! `content_block_start`, deltas arrive as `content_block_delta`, and
//! closes with `content_block_stop`. We map these 1:1 onto
//! `StreamPart::TextStart/Delta/End` and
//! `StreamPart::ToolInputStart/Delta/End` + `ToolCall`.
//!
//! # Usage
//!
//! Input tokens arrive in `message_start.message.usage`; output tokens
//! arrive in `message_delta.usage`. We accumulate both and emit one
//! `Finish` frame at the end.
// Rust guideline compliant 2026-02-21

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use llmsdk_provider::language_model::{
    FinishReason, FinishReasonKind, Source, StreamPart, ToolCallPart, ToolResult, ToolResultOutput,
};
use llmsdk_provider::shared::{ProviderMetadata, ProviderOptions, Warning};
use serde_json::{Map, Value as JsonValue};

use crate::config::GenerateIdFn;

use super::finish_reason::{map as map_finish_reason, map_with_json_tool};
use super::stream_event::{BlockDelta, BlockStart, StreamEvent};
use super::usage::convert as convert_usage;
use super::wire::ResponseUsage;

#[derive(Debug)]
enum BlockKind {
    Text {
        /// Logical text-block id we emit downstream (matches `Anthropic`'s
        /// numeric index, formatted as decimal string).
        id: String,
    },
    ToolUse {
        id: String,
        name: String,
        arguments: String,
        /// Normalized `caller` (`snake_case` `tool_id` → `camelCase` `toolId`)
        /// pre-staged at `content_block_start` so the closing `tool-call`
        /// frame can attach it via `provider_metadata.anthropic.caller`,
        /// mirroring upstream `anthropic-language-model.ts:1659`.
        caller: Option<JsonValue>,
        /// `true` until the first non-empty `input_json_delta` arrives. Used
        /// to inject `"type": "programmatic-tool-call"` into the *first*
        /// streaming delta for the unified `code_execution` provider tool.
        /// Mirrors upstream `anthropic-language-model.ts:2241-2249`'s
        /// `firstDelta` flag. Initialized `true` at `content_block_start`,
        /// flipped to `false` once any delta has been buffered (including
        /// inline input that arrives with `server_tool_use`'s opening
        /// frame), per the upstream rule "only set firstDelta: true when no
        /// input has been buffered yet".
        first_delta: bool,
    },
    /// Extended-thinking block; tracks the latest signature observed via
    /// `signature_delta`.
    Reasoning { id: String },
    /// `tool_use` block synthesized by `jsonResponseTool`: `input_json` deltas
    /// are forwarded as text deltas and the block closes as text-end, not
    /// tool-call. Mirrors upstream `anthropic-language-model.ts:2229-2266`.
    JsonText { id: String },
}

#[expect(
    clippy::struct_excessive_bools,
    reason = "mirrors ai-sdk anthropic-language-model.ts stream state flags 1:1; \
              each flag is independently observable on the wire and cannot be \
              collapsed without losing the json-response-tool semantics"
)]
pub(crate) struct StreamState {
    initial_warnings: Option<Vec<Warning>>,
    finish_reason: FinishReason,
    /// Accumulated usage for the trailing `Finish` frame.
    usage: ResponseUsage,
    /// Captured `message_start` metadata; emitted once.
    metadata: Option<llmsdk_provider::language_model::ResponseMetadata>,
    metadata_emitted: bool,
    /// Open blocks keyed by `Anthropic`'s `index`.
    blocks: BTreeMap<u32, BlockKind>,
    /// Monotonic source id counter used as a fallback when no
    /// `generate_id` callback is configured. Bumped for each
    /// `citations_delta` block.
    source_seq: u64,
    /// Optional id generator (mirrors `config.generateId` upstream).
    generate_id: Option<Arc<GenerateIdFn>>,
    /// When true, `code_execution` server-tool uses are emitted with
    /// `dynamic: true` to bypass strict tool validation. See
    /// `model::has_web_tool_20260209_without_code_execution`.
    mark_code_execution_dynamic: bool,
    /// When true, the request synthesized a `name = "json"` tool to
    /// fall back to a tool-call-based JSON response (see
    /// `model::build_request` jsonResponseTool path). At parse time this
    /// flag flips a `tool_use(name='json')` block into a text-only frame
    /// and remaps `tool_use` finish reason to `stop`, mirroring upstream
    /// `anthropic-language-model.ts:1620-1632` + `:2229-2266`.
    uses_json_response_tool: bool,
    /// Set to true once a `name='json'` `tool_use` block is encountered so
    /// the final `Finish` frame can collapse `tool_use → stop`.
    is_json_response_from_tool: bool,
    /// Cache of MCP tool-call entries so the matching `mcp_tool_result`
    /// can inherit `toolName` + `providerMetadata`. Mirrors `mcpToolCalls`
    /// in upstream `anthropic-language-model.ts:2026-2057`.
    mcp_tool_calls: HashMap<String, (String, Option<ProviderMetadata>)>,
}

impl std::fmt::Debug for StreamState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamState")
            .field("initial_warnings", &self.initial_warnings)
            .field("finish_reason", &self.finish_reason)
            .field("usage", &self.usage)
            .field("metadata", &self.metadata)
            .field("metadata_emitted", &self.metadata_emitted)
            .field("blocks", &self.blocks)
            .field("source_seq", &self.source_seq)
            .field("generate_id", &self.generate_id.is_some())
            .field(
                "mark_code_execution_dynamic",
                &self.mark_code_execution_dynamic,
            )
            .field("uses_json_response_tool", &self.uses_json_response_tool)
            .field(
                "is_json_response_from_tool",
                &self.is_json_response_from_tool,
            )
            .field("mcp_tool_calls", &self.mcp_tool_calls.keys())
            .finish()
    }
}

impl StreamState {
    #[cfg(test)]
    pub(crate) fn new(warnings: Vec<Warning>) -> Self {
        Self::with_generate_id(warnings, None, false, false)
    }

    pub(crate) fn with_generate_id(
        warnings: Vec<Warning>,
        generate_id: Option<Arc<GenerateIdFn>>,
        mark_code_execution_dynamic: bool,
        uses_json_response_tool: bool,
    ) -> Self {
        Self {
            initial_warnings: Some(warnings),
            finish_reason: FinishReason::new(FinishReasonKind::Other),
            usage: ResponseUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                iterations: None,
            },
            metadata: None,
            metadata_emitted: false,
            blocks: BTreeMap::new(),
            source_seq: 0,
            generate_id,
            mark_code_execution_dynamic,
            uses_json_response_tool,
            is_json_response_from_tool: false,
            mcp_tool_calls: HashMap::new(),
        }
    }

    pub(crate) fn start_frames(&mut self) -> Vec<StreamPart> {
        let warnings = self.initial_warnings.take().unwrap_or_default();
        vec![StreamPart::StreamStart { warnings }]
    }

    pub(crate) fn on_event(&mut self, event: StreamEvent) -> Vec<StreamPart> {
        match event {
            StreamEvent::MessageStart { message } => {
                self.usage.input_tokens = message.usage.input_tokens;
                self.usage.cache_creation_input_tokens = message.usage.cache_creation_input_tokens;
                self.usage.cache_read_input_tokens = message.usage.cache_read_input_tokens;
                self.metadata = Some(llmsdk_provider::language_model::ResponseMetadata {
                    id: message.id,
                    timestamp: None,
                    model_id: message.model,
                    headers: None,
                });
                self.emit_metadata_once()
            }
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => self.on_block_start(index, content_block),
            StreamEvent::ContentBlockDelta { index, delta } => self.on_block_delta(index, delta),
            StreamEvent::ContentBlockStop { index } => self.on_block_stop(index),
            StreamEvent::MessageDelta { delta, usage } => {
                if let Some(reason) = delta.stop_reason.as_deref()
                    && !matches!(self.finish_reason.unified, FinishReasonKind::Error)
                {
                    self.finish_reason = map_finish_reason(Some(reason));
                }
                // Mirror upstream `anthropic-language-model.ts:2382-2401`:
                // accept the delta's input_tokens/output_tokens (overwrite
                // when newer) and propagate cache + iterations updates.
                if let Some(u) = usage {
                    if let Some(in_tokens) = u.input_tokens
                        && self.usage.input_tokens != in_tokens
                    {
                        self.usage.input_tokens = in_tokens;
                    }
                    if let Some(out_tokens) = u.output_tokens {
                        self.usage.output_tokens = out_tokens;
                    }
                    if let Some(cache_read) = u.cache_read_input_tokens {
                        self.usage.cache_read_input_tokens = Some(cache_read);
                    }
                    if let Some(cache_creation) = u.cache_creation_input_tokens {
                        self.usage.cache_creation_input_tokens = Some(cache_creation);
                    }
                    if let Some(iterations) = u.iterations {
                        self.usage.iterations = Some(iterations);
                    }
                }
                Vec::new()
            }
            StreamEvent::Error { error } => {
                self.finish_reason = FinishReason::new(FinishReasonKind::Error);
                vec![StreamPart::Error {
                    error: serde_json::json!({ "message": error.message }),
                }]
            }
            StreamEvent::MessageStop | StreamEvent::Ping | StreamEvent::Other => Vec::new(),
        }
    }

    pub(crate) fn on_parse_error(&mut self, raw: &str, message: &str) -> Vec<StreamPart> {
        self.finish_reason = FinishReason::new(FinishReasonKind::Error);
        vec![StreamPart::Error {
            error: serde_json::json!({ "message": message, "raw": raw }),
        }]
    }

    /// Emit `TextEnd` / `ToolInputEnd` + `ToolCall` for any blocks the
    /// server left open, then `Finish`.
    pub(crate) fn flush(self) -> Vec<StreamPart> {
        let mut out = Vec::new();
        for (_idx, kind) in self.blocks {
            match kind {
                BlockKind::Text { id } | BlockKind::JsonText { id } => {
                    out.push(StreamPart::TextEnd {
                        id,
                        provider_metadata: None,
                    });
                }
                BlockKind::ToolUse {
                    id,
                    name,
                    arguments,
                    caller,
                    ..
                } => {
                    let dynamic = (self.mark_code_execution_dynamic && name == "code_execution")
                        .then_some(true);
                    out.push(StreamPart::ToolInputEnd {
                        id: id.clone(),
                        provider_metadata: None,
                    });
                    out.push(StreamPart::ToolCall(build_tool_call(
                        id, name, arguments, caller, dynamic,
                    )));
                }
                BlockKind::Reasoning { id } => out.push(StreamPart::ReasoningEnd {
                    id,
                    provider_metadata: None,
                }),
            }
        }
        // When jsonResponseTool fired, the `tool_use` raw stop reason must
        // collapse to `stop`; everything else flows through unchanged so we
        // don't override an upstream `Error` or unmapped raw value.
        let finish = if self.is_json_response_from_tool
            && self.finish_reason.unified == FinishReasonKind::ToolCalls
        {
            map_with_json_tool(self.finish_reason.raw.as_deref(), true)
        } else {
            self.finish_reason
        };
        out.push(StreamPart::Finish {
            usage: convert_usage(&self.usage),
            finish_reason: finish,
            provider_metadata: None,
        });
        out
    }

    fn emit_metadata_once(&mut self) -> Vec<StreamPart> {
        if self.metadata_emitted {
            return Vec::new();
        }
        let Some(meta) = self.metadata.clone() else {
            return Vec::new();
        };
        self.metadata_emitted = true;
        vec![StreamPart::ResponseMetadata(meta)]
    }

    #[allow(
        clippy::too_many_lines,
        reason = "dispatch over BlockStart variants; each branch is short but the function is long"
    )]
    fn on_block_start(&mut self, index: u32, block: BlockStart) -> Vec<StreamPart> {
        match block {
            BlockStart::Text { text } => {
                let id = index.to_string();
                self.blocks
                    .insert(index, BlockKind::Text { id: id.clone() });
                let mut out = vec![StreamPart::TextStart {
                    id: id.clone(),
                    provider_metadata: None,
                }];
                if !text.is_empty() {
                    out.push(StreamPart::TextDelta {
                        id,
                        delta: text,
                        provider_metadata: None,
                    });
                }
                out
            }
            BlockStart::ToolUse {
                id,
                name,
                input,
                caller,
            } => {
                // jsonResponseTool fallback: when the request synthesized a
                // `name="json"` tool, route this block as text rather than
                // a tool call so consumers see only the assembled JSON. The
                // flag also flips `tool_use → stop` at finish time.
                // Mirrors upstream anthropic-language-model.ts:2229-2266.
                if self.uses_json_response_tool && name == "json" {
                    self.is_json_response_from_tool = true;
                    let block_id = index.to_string();
                    self.blocks.insert(
                        index,
                        BlockKind::JsonText {
                            id: block_id.clone(),
                        },
                    );
                    let mut out = vec![StreamPart::TextStart {
                        id: block_id.clone(),
                        provider_metadata: None,
                    }];
                    // `content_block_start` may carry an inline non-empty
                    // input object (programmatic-tool-call short path). When
                    // present, surface it as the opening text delta.
                    if let Some(v) = input
                        && !v.is_null()
                    {
                        let text = serde_json::to_string(&v).unwrap_or_default();
                        if !text.is_empty() && text != "{}" {
                            out.push(StreamPart::TextDelta {
                                id: block_id,
                                delta: text,
                                provider_metadata: None,
                            });
                        }
                    }
                    let _ = id;
                    let _ = caller;
                    return out;
                }
                let arguments = match input {
                    Some(v) if !v.is_null() => serde_json::to_string(&v).unwrap_or_default(),
                    _ => String::new(),
                };
                let normalized_caller = normalize_caller(caller.as_ref());
                self.blocks.insert(
                    index,
                    BlockKind::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                        caller: normalized_caller,
                        // Inline input already buffered? Then `firstDelta`
                        // must start `false` so we don't double-prefix later.
                        first_delta: arguments.is_empty(),
                    },
                );
                // Mark code_execution invocations as dynamic when the request
                // enabled web_*_20260209 without an explicit code_execution
                // tool. Mirrors upstream anthropic-language-model.ts:1714-1735.
                let dynamic =
                    (self.mark_code_execution_dynamic && name == "code_execution").then_some(true);
                let mut out = vec![StreamPart::ToolInputStart {
                    id: id.clone(),
                    tool_name: name,
                    provider_executed: None,
                    dynamic,
                    title: None,
                    provider_metadata: None,
                }];
                if !arguments.is_empty() {
                    out.push(StreamPart::ToolInputDelta {
                        id,
                        delta: arguments,
                        provider_metadata: None,
                    });
                }
                out
            }
            BlockStart::Thinking {
                thinking,
                signature,
            } => {
                let id = index.to_string();
                self.blocks
                    .insert(index, BlockKind::Reasoning { id: id.clone() });
                let mut out = vec![StreamPart::ReasoningStart {
                    id: id.clone(),
                    provider_metadata: None,
                }];
                if !thinking.is_empty() {
                    out.push(StreamPart::ReasoningDelta {
                        id: id.clone(),
                        delta: thinking,
                        provider_metadata: None,
                    });
                }
                if let Some(sig) = signature {
                    out.push(StreamPart::ReasoningDelta {
                        id,
                        delta: String::new(),
                        provider_metadata: Some(signature_metadata(&sig)),
                    });
                }
                out
            }
            BlockStart::RedactedThinking { data } => {
                let id = index.to_string();
                self.blocks
                    .insert(index, BlockKind::Reasoning { id: id.clone() });
                vec![StreamPart::ReasoningStart {
                    id,
                    provider_metadata: Some(redacted_metadata(&data)),
                }]
            }
            BlockStart::Compaction { content } => {
                // Open a regular text block tagged with
                // `anthropic.type = "compaction"`. Any inline `content`
                // is forwarded as the first text delta. Mirrors upstream
                // anthropic-language-model.ts:1606-1618.
                let id = index.to_string();
                self.blocks
                    .insert(index, BlockKind::Text { id: id.clone() });
                let mut out = vec![StreamPart::TextStart {
                    id: id.clone(),
                    provider_metadata: Some(compaction_metadata()),
                }];
                if let Some(text) = content
                    && !text.is_empty()
                {
                    out.push(StreamPart::TextDelta {
                        id,
                        delta: text,
                        provider_metadata: None,
                    });
                }
                out
            }
            BlockStart::ServerToolUse { id, name, input } => {
                // Streaming variant of `parse_response.rs::ResponseContent::ServerToolUse`.
                // Anthropic emits server-side tool invocations (web_search /
                // code_execution / web_fetch / tool_search / bash / text_editor
                // family) as `content_block_start` of type `server_tool_use`
                // with the full input inline. Mirrors upstream
                // `anthropic-language-model.ts:1671-1735`.
                //
                // Apply `code_execution_20250825` sub-tool collapsing here:
                // `bash_code_execution`/`text_editor_code_execution` →
                // `code_execution` (with `type` injected into input).
                let raw_input = input.unwrap_or(JsonValue::Null);
                let (mapped_name, mapped_input) =
                    crate::messages::parse_response::remap_code_execution_subtool(&name, raw_input);
                let dynamic = (self.mark_code_execution_dynamic && mapped_name == "code_execution")
                    .then_some(true);
                let input_json = if mapped_input.is_null() {
                    String::new()
                } else {
                    serde_json::to_string(&mapped_input).unwrap_or_default()
                };
                self.blocks.insert(
                    index,
                    BlockKind::ToolUse {
                        id: id.clone(),
                        name: mapped_name.clone(),
                        arguments: input_json.clone(),
                        caller: None,
                        first_delta: input_json.is_empty() || input_json == "{}",
                    },
                );
                let mut out = vec![StreamPart::ToolInputStart {
                    id: id.clone(),
                    tool_name: mapped_name,
                    provider_executed: Some(true),
                    dynamic,
                    title: None,
                    provider_metadata: None,
                }];
                if !input_json.is_empty() && input_json != "{}" {
                    out.push(StreamPart::ToolInputDelta {
                        id,
                        delta: input_json,
                        provider_metadata: None,
                    });
                }
                out
            }
            BlockStart::McpToolUse(v) => {
                // Mirror non-stream `parse_response.rs:McpToolUse` arm:
                // emit a `ToolCall` tagged with `serverName` and cache the
                // metadata for the trailing `mcp_tool_result` event.
                let (id, name, server_name) = mcp_call_meta_from_value(&v);
                let input = v
                    .as_object()
                    .and_then(|m| m.get("input").cloned())
                    .unwrap_or(JsonValue::Null);
                let provider_metadata = mcp_tool_use_metadata(server_name.as_deref());
                self.mcp_tool_calls
                    .insert(id.clone(), (name.clone(), Some(provider_metadata.clone())));
                let input_json = serde_json::to_string(&input).unwrap_or_default();
                let mut out = vec![
                    StreamPart::ToolInputStart {
                        id: id.clone(),
                        tool_name: name.clone(),
                        dynamic: Some(true),
                        provider_executed: Some(true),
                        title: None,
                        provider_metadata: Some(provider_metadata.clone()),
                    },
                    StreamPart::ToolCall(ToolCallPart {
                        tool_call_id: id.clone(),
                        tool_name: name,
                        input,
                        provider_executed: Some(true),
                        dynamic: Some(true),
                        provider_options: Some(provider_metadata_to_options(&provider_metadata)),
                    }),
                ];
                if !input_json.is_empty() && input_json != "{}" {
                    out.push(StreamPart::ToolInputDelta {
                        id,
                        delta: input_json,
                        provider_metadata: None,
                    });
                }
                out
            }
            BlockStart::McpToolResult(v) => {
                // Look up the cached tool-call entry to inherit toolName +
                // providerMetadata. Mirrors upstream `2045-2057`.
                let tool_use_id = v
                    .as_object()
                    .and_then(|m| m.get("tool_use_id").and_then(JsonValue::as_str))
                    .unwrap_or_default()
                    .to_owned();
                let is_error = v
                    .as_object()
                    .and_then(|m| m.get("is_error").and_then(JsonValue::as_bool))
                    .unwrap_or(false);
                let result_value = v
                    .as_object()
                    .and_then(|m| m.get("content").cloned())
                    .unwrap_or(JsonValue::Null);
                let (tool_name, provider_metadata) = self
                    .mcp_tool_calls
                    .get(&tool_use_id)
                    .cloned()
                    .unwrap_or_else(|| (String::new(), None));
                let output = if is_error {
                    ToolResultOutput::ErrorJson {
                        value: result_value,
                        provider_options: None,
                    }
                } else {
                    ToolResultOutput::Json {
                        value: result_value,
                        provider_options: None,
                    }
                };
                vec![StreamPart::ToolResult(ToolResult {
                    tool_call_id: tool_use_id,
                    tool_name,
                    output,
                    preliminary: None,
                    provider_metadata,
                })]
            }
            BlockStart::WebSearchToolResult(v)
            | BlockStart::WebFetchToolResult(v)
            | BlockStart::CodeExecutionToolResult(v)
            | BlockStart::BashCodeExecutionToolResult(v)
            | BlockStart::TextEditorCodeExecutionToolResult(v)
            | BlockStart::ToolSearchToolResult(v)
            | BlockStart::AdvisorToolResult(v) => {
                // Streaming variant of the 9 server-tool result content blocks
                // handled non-streaming in
                // `parse_response.rs:166-209`. Each block already contains
                // the full result on `content_block_start`; emit a single
                // `ToolResult` part (no later deltas / stop frames to
                // forward).
                let (tool_call_id, tool_name) =
                    super::parse_response::extract_tool_call_id_and_name(&v);
                let is_error = super::parse_response::is_tool_result_error(&v);
                let mut anthropic_meta = Map::new();
                if is_error {
                    anthropic_meta.insert("isError".into(), JsonValue::Bool(true));
                }
                let provider_metadata = if anthropic_meta.is_empty() {
                    None
                } else {
                    let mut pm = ProviderMetadata::new();
                    pm.insert("anthropic".into(), anthropic_meta);
                    Some(pm)
                };
                let output = if is_error {
                    ToolResultOutput::ErrorJson {
                        value: v,
                        provider_options: None,
                    }
                } else {
                    ToolResultOutput::Json {
                        value: v,
                        provider_options: None,
                    }
                };
                vec![StreamPart::ToolResult(ToolResult {
                    tool_call_id,
                    tool_name,
                    output,
                    preliminary: None,
                    provider_metadata,
                })]
            }
            BlockStart::Other => Vec::new(),
        }
    }

    fn on_block_delta(&mut self, index: u32, delta: BlockDelta) -> Vec<StreamPart> {
        let Some(block) = self.blocks.get_mut(&index) else {
            return Vec::new();
        };
        match (block, delta) {
            (BlockKind::Text { id }, BlockDelta::TextDelta { text }) => {
                if text.is_empty() {
                    return Vec::new();
                }
                vec![StreamPart::TextDelta {
                    id: id.clone(),
                    delta: text,
                    provider_metadata: None,
                }]
            }
            (
                BlockKind::ToolUse {
                    id,
                    name,
                    arguments,
                    first_delta,
                    ..
                },
                BlockDelta::InputJsonDelta { partial_json },
            ) => {
                if partial_json.is_empty() {
                    return Vec::new();
                }
                // Mirror upstream anthropic-language-model.ts:2241-2249:
                // when this is the very first delta for a unified
                // `code_execution` tool (which already covers bash /
                // text_editor sub-tools after remapping), and the delta
                // opens a JSON object (`{...`), inject
                // `"type":"programmatic-tool-call",` so the assembled
                // input has a stable `type` field downstream.
                let injected =
                    if *first_delta && name == "code_execution" && partial_json.starts_with('{') {
                        format!(
                            "{{\"type\":\"programmatic-tool-call\",{}",
                            &partial_json[1..]
                        )
                    } else {
                        partial_json.clone()
                    };
                *first_delta = false;
                arguments.push_str(&injected);
                vec![StreamPart::ToolInputDelta {
                    id: id.clone(),
                    delta: injected,
                    provider_metadata: None,
                }]
            }
            (BlockKind::JsonText { id }, BlockDelta::InputJsonDelta { partial_json }) => {
                // jsonResponseTool: forward `input_json_delta` fragments as
                // raw text deltas — they already form a valid JSON suffix
                // when concatenated. Mirrors upstream
                // anthropic-language-model.ts:2253-2261.
                if partial_json.is_empty() {
                    return Vec::new();
                }
                vec![StreamPart::TextDelta {
                    id: id.clone(),
                    delta: partial_json,
                    provider_metadata: None,
                }]
            }
            (BlockKind::Reasoning { id }, BlockDelta::ThinkingDelta { thinking }) => {
                if thinking.is_empty() {
                    return Vec::new();
                }
                vec![StreamPart::ReasoningDelta {
                    id: id.clone(),
                    delta: thinking,
                    provider_metadata: None,
                }]
            }
            (BlockKind::Reasoning { id }, BlockDelta::SignatureDelta { signature }) => {
                vec![StreamPart::ReasoningDelta {
                    id: id.clone(),
                    delta: String::new(),
                    provider_metadata: Some(signature_metadata(&signature)),
                }]
            }
            (BlockKind::Text { id }, BlockDelta::CompactionDelta { content }) => {
                // Forward compaction-block deltas as plain text deltas.
                // Mirrors upstream anthropic-language-model.ts:2207-2218.
                let Some(text) = content else {
                    return Vec::new();
                };
                if text.is_empty() {
                    return Vec::new();
                }
                vec![StreamPart::TextDelta {
                    id: id.clone(),
                    delta: text,
                    provider_metadata: None,
                }]
            }
            (BlockKind::Text { .. }, BlockDelta::CitationsDelta { citation }) => {
                let id = if let Some(gen_fn) = &self.generate_id {
                    gen_fn()
                } else {
                    self.source_seq = self.source_seq.saturating_add(1);
                    format!("anthropic-cite-{}", self.source_seq)
                };
                match build_citation_source(&citation, id) {
                    Some(source) => vec![StreamPart::Source(source)],
                    None => Vec::new(),
                }
            }
            _ => Vec::new(),
        }
    }

    fn on_block_stop(&mut self, index: u32) -> Vec<StreamPart> {
        let Some(kind) = self.blocks.remove(&index) else {
            return Vec::new();
        };
        match kind {
            BlockKind::Text { id } | BlockKind::JsonText { id } => vec![StreamPart::TextEnd {
                id,
                provider_metadata: None,
            }],
            BlockKind::ToolUse {
                id,
                name,
                arguments,
                caller,
                ..
            } => {
                // Mirror upstream `anthropic-language-model.ts:2107-2138`:
                // mark final tool-call as `dynamic: true` when the request
                // enabled `markCodeExecutionDynamic` and the (already
                // remapped) tool name is `code_execution`. Without this the
                // streaming finalization would emit `dynamic: None` even for
                // code_execution invocations triggered implicitly by the
                // newer web_*_20260209 tools, causing strict tool validation
                // to reject the response.
                let dynamic =
                    (self.mark_code_execution_dynamic && name == "code_execution").then_some(true);
                vec![
                    StreamPart::ToolInputEnd {
                        id: id.clone(),
                        provider_metadata: None,
                    },
                    StreamPart::ToolCall(build_tool_call(id, name, arguments, caller, dynamic)),
                ]
            }
            BlockKind::Reasoning { id } => vec![StreamPart::ReasoningEnd {
                id,
                provider_metadata: None,
            }],
        }
    }
}

fn signature_metadata(signature: &str) -> ProviderMetadata {
    let mut anthropic = Map::new();
    anthropic.insert(
        "signature".to_owned(),
        serde_json::Value::String(signature.to_owned()),
    );
    let mut pm = ProviderMetadata::new();
    pm.insert("anthropic".to_owned(), anthropic);
    pm
}

fn redacted_metadata(data: &str) -> ProviderMetadata {
    let mut anthropic = Map::new();
    anthropic.insert(
        "redactedData".to_owned(),
        serde_json::Value::String(data.to_owned()),
    );
    let mut pm = ProviderMetadata::new();
    pm.insert("anthropic".to_owned(), anthropic);
    pm
}

fn compaction_metadata() -> ProviderMetadata {
    let mut anthropic = Map::new();
    anthropic.insert(
        "type".to_owned(),
        serde_json::Value::String("compaction".to_owned()),
    );
    let mut pm = ProviderMetadata::new();
    pm.insert("anthropic".to_owned(), anthropic);
    pm
}

/// Extract `id`, `name`, `server_name` from a raw `mcp_tool_use` block.
fn mcp_call_meta_from_value(v: &JsonValue) -> (String, String, Option<String>) {
    let Some(map) = v.as_object() else {
        return (String::new(), String::new(), None);
    };
    let id = map
        .get("id")
        .and_then(JsonValue::as_str)
        .unwrap_or_default()
        .to_owned();
    let name = map
        .get("name")
        .and_then(JsonValue::as_str)
        .unwrap_or_default()
        .to_owned();
    let server_name = map
        .get("server_name")
        .and_then(JsonValue::as_str)
        .map(str::to_owned);
    (id, name, server_name)
}

/// Build the `anthropic` slot of an MCP tool-call's provider metadata.
fn mcp_tool_use_metadata(server_name: Option<&str>) -> ProviderMetadata {
    let mut anthropic = Map::new();
    anthropic.insert("type".into(), JsonValue::String("mcp-tool-use".into()));
    if let Some(sn) = server_name {
        anthropic.insert("serverName".into(), JsonValue::String(sn.to_owned()));
    }
    let mut pm = ProviderMetadata::new();
    pm.insert("anthropic".into(), anthropic);
    pm
}

/// Echo the provider metadata into a [`ProviderOptions`] map so the
/// streaming tool-call carries the same hints when consumers serialize it.
fn provider_metadata_to_options(pm: &ProviderMetadata) -> ProviderOptions {
    let mut opts = ProviderOptions::new();
    for (key, value) in pm {
        opts.insert(key.clone(), value.clone());
    }
    opts
}

/// Mirrors `createCitationSource` in `anthropic-language-model.ts`.
///
/// `citation` is the raw `citations_delta.citation` payload. Returns
/// `None` for unknown citation shapes (matches ai-sdk's silent drop).
fn build_citation_source(citation: &serde_json::Value, id: String) -> Option<Source> {
    let kind = citation.get("type").and_then(|v| v.as_str())?;
    let cited_text = citation
        .get("cited_text")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let mut anthropic = Map::new();
    if let Some(ct) = &cited_text {
        anthropic.insert(
            "citedText".to_owned(),
            serde_json::Value::String(ct.clone()),
        );
    }
    match kind {
        "web_search_result_location" => {
            let url = citation.get("url").and_then(|v| v.as_str())?.to_owned();
            let title = citation
                .get("title")
                .and_then(|v| v.as_str())
                .map(str::to_owned);
            if let Some(idx) = citation.get("encrypted_index").and_then(|v| v.as_str()) {
                anthropic.insert(
                    "encryptedIndex".to_owned(),
                    serde_json::Value::String(idx.to_owned()),
                );
            }
            let mut pm = ProviderMetadata::new();
            pm.insert("anthropic".to_owned(), anthropic);
            Some(Source::Url {
                id,
                url,
                title,
                provider_metadata: Some(pm),
            })
        }
        "page_location" => {
            if let Some(n) = citation.get("start_page_number") {
                anthropic.insert("startPageNumber".to_owned(), n.clone());
            }
            if let Some(n) = citation.get("end_page_number") {
                anthropic.insert("endPageNumber".to_owned(), n.clone());
            }
            let mut pm = ProviderMetadata::new();
            pm.insert("anthropic".to_owned(), anthropic);
            let title = citation
                .get("document_title")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned();
            Some(Source::Document {
                id,
                media_type: String::new(),
                title,
                filename: None,
                provider_metadata: Some(pm),
            })
        }
        "char_location" => {
            if let Some(n) = citation.get("start_char_index") {
                anthropic.insert("startCharIndex".to_owned(), n.clone());
            }
            if let Some(n) = citation.get("end_char_index") {
                anthropic.insert("endCharIndex".to_owned(), n.clone());
            }
            let mut pm = ProviderMetadata::new();
            pm.insert("anthropic".to_owned(), anthropic);
            let title = citation
                .get("document_title")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned();
            Some(Source::Document {
                id,
                media_type: String::new(),
                title,
                filename: None,
                provider_metadata: Some(pm),
            })
        }
        _ => None,
    }
}

fn build_tool_call(
    id: String,
    name: String,
    arguments: String,
    caller: Option<JsonValue>,
    dynamic: Option<bool>,
) -> ToolCallPart {
    let input = if arguments.is_empty() {
        serde_json::Value::Object(serde_json::Map::new())
    } else {
        serde_json::from_str::<serde_json::Value>(&arguments)
            .unwrap_or(serde_json::Value::String(arguments))
    };
    // Mirror upstream anthropic-language-model.ts:2139-2145 — when a caller
    // is present, attach it under provider_metadata.anthropic.caller so
    // downstream multi-language consumers see the same providerMetadata
    // shape as the non-streaming path (parse_response normalizes the same
    // way).
    let provider_options = caller.map(|c| {
        let mut anthropic = Map::new();
        anthropic.insert("caller".into(), c);
        let mut po = ProviderOptions::new();
        po.insert("anthropic".into(), anthropic);
        po
    });
    // `provider_executed` follows the upstream contract: only the
    // server-executed branch sets it (see `BlockStart::ServerToolUse` /
    // `BlockStart::McpToolUse`). The function-tool finalization path here
    // intentionally leaves it `None`.
    ToolCallPart {
        tool_call_id: id,
        tool_name: name,
        input,
        provider_executed: None,
        dynamic,
        provider_options,
    }
}

/// Normalize an Anthropic `caller` payload from wire `snake_case`
/// (`tool_id`) into the provider-metadata camelCase contract (`toolId`).
///
/// Mirrors `parse_response.rs`'s `tool_use` caller normalization and the
/// upstream `callerInfo` helper in `anthropic-language-model.ts:984-990`
/// (non-streaming) and `:1635-1642` (streaming). `direct` variants have
/// no `tool_id`; the resulting object omits `toolId` per upstream
/// `toolId: undefined` → JSON.stringify drop behavior.
fn normalize_caller(caller: Option<&JsonValue>) -> Option<JsonValue> {
    let obj = caller?.as_object()?;
    let caller_type = obj.get("type")?.as_str()?.to_owned();
    let mut normalized = Map::new();
    normalized.insert("type".into(), JsonValue::String(caller_type));
    if let Some(tool_id) = obj.get("tool_id").and_then(|v| v.as_str()) {
        normalized.insert("toolId".into(), JsonValue::String(tool_id.to_owned()));
    }
    Some(JsonValue::Object(normalized))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::stream_event::{
        MessageDeltaInner, MessageDeltaUsage, MessageStartUsage, StreamError, StreamMessageMeta,
    };

    fn empty_usage() -> MessageStartUsage {
        MessageStartUsage {
            input_tokens: 5,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        }
    }

    #[test]
    fn text_block_lifecycle() {
        let mut state = StreamState::new(vec![]);
        let s = state.start_frames();
        assert!(matches!(s[0], StreamPart::StreamStart { .. }));

        let f1 = state.on_event(StreamEvent::MessageStart {
            message: StreamMessageMeta {
                id: Some("msg_1".into()),
                model: Some("claude-3-5".into()),
                usage: empty_usage(),
            },
        });
        assert!(matches!(f1[0], StreamPart::ResponseMetadata(_)));

        let f2 = state.on_event(StreamEvent::ContentBlockStart {
            index: 0,
            content_block: BlockStart::Text {
                text: String::new(),
            },
        });
        assert!(matches!(f2[0], StreamPart::TextStart { .. }));

        let f3 = state.on_event(StreamEvent::ContentBlockDelta {
            index: 0,
            delta: BlockDelta::TextDelta {
                text: "hello".into(),
            },
        });
        assert!(matches!(&f3[0], StreamPart::TextDelta { delta, .. } if delta == "hello"));

        let f4 = state.on_event(StreamEvent::ContentBlockStop { index: 0 });
        assert!(matches!(f4[0], StreamPart::TextEnd { .. }));

        let f5 = state.on_event(StreamEvent::MessageDelta {
            delta: MessageDeltaInner {
                stop_reason: Some("end_turn".into()),
            },
            usage: Some(MessageDeltaUsage {
                input_tokens: None,
                output_tokens: Some(2),
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                iterations: None,
            }),
        });
        assert!(f5.is_empty());

        let _ = state.on_event(StreamEvent::MessageStop);
        let tail = state.flush();
        assert_eq!(tail.len(), 1);
        if let StreamPart::Finish {
            finish_reason,
            usage,
            ..
        } = &tail[0]
        {
            assert_eq!(finish_reason.unified, FinishReasonKind::Stop);
            assert_eq!(usage.input_tokens.total, Some(5));
            assert_eq!(usage.output_tokens.total, Some(2));
        } else {
            panic!("expected Finish");
        }
    }

    #[test]
    fn tool_use_block_assembles_input() {
        let mut state = StreamState::new(vec![]);
        let _ = state.start_frames();
        let _ = state.on_event(StreamEvent::MessageStart {
            message: StreamMessageMeta {
                id: None,
                model: None,
                usage: empty_usage(),
            },
        });

        let f1 = state.on_event(StreamEvent::ContentBlockStart {
            index: 0,
            content_block: BlockStart::ToolUse {
                id: "tu_a".into(),
                name: "weather".into(),
                input: None,
                caller: None,
            },
        });
        assert!(matches!(f1[0], StreamPart::ToolInputStart { .. }));

        let _ = state.on_event(StreamEvent::ContentBlockDelta {
            index: 0,
            delta: BlockDelta::InputJsonDelta {
                partial_json: r#"{"ci"#.into(),
            },
        });
        let _ = state.on_event(StreamEvent::ContentBlockDelta {
            index: 0,
            delta: BlockDelta::InputJsonDelta {
                partial_json: r#"ty":"NYC"}"#.into(),
            },
        });

        let stop = state.on_event(StreamEvent::ContentBlockStop { index: 0 });
        // Expect ToolInputEnd then ToolCall.
        assert!(matches!(stop[0], StreamPart::ToolInputEnd { .. }));
        if let StreamPart::ToolCall(tc) = &stop[1] {
            assert_eq!(tc.tool_call_id, "tu_a");
            assert_eq!(tc.input["city"], "NYC");
        } else {
            panic!("expected ToolCall");
        }

        let _ = state.on_event(StreamEvent::MessageDelta {
            delta: MessageDeltaInner {
                stop_reason: Some("tool_use".into()),
            },
            usage: None,
        });
        let tail = state.flush();
        if let StreamPart::Finish { finish_reason, .. } = &tail[0] {
            assert_eq!(finish_reason.unified, FinishReasonKind::ToolCalls);
        }
    }

    #[test]
    fn tool_use_stream_attaches_caller_to_provider_metadata() {
        // Stream-path parity with parse_response: when
        // content_block_start.tool_use ships `caller.tool_id`, the closing
        // `tool-call` frame must carry it through `provider_metadata.anthropic.caller`
        // with the snake_case → camelCase normalization that upstream
        // anthropic-language-model.ts:1635-1642 + :2139-2145 perform.
        let mut state = StreamState::new(vec![]);
        let _ = state.start_frames();
        let _ = state.on_event(StreamEvent::MessageStart {
            message: StreamMessageMeta {
                id: None,
                model: None,
                usage: empty_usage(),
            },
        });

        let _ = state.on_event(StreamEvent::ContentBlockStart {
            index: 0,
            content_block: BlockStart::ToolUse {
                id: "tu_caller".into(),
                name: "query_db".into(),
                input: Some(serde_json::json!({"sql": "SELECT 1"})),
                caller: Some(serde_json::json!({
                    "type": "code_execution_20250825",
                    "tool_id": "srvtoolu_01CodeExec",
                })),
            },
        });

        let stop = state.on_event(StreamEvent::ContentBlockStop { index: 0 });
        let StreamPart::ToolCall(tc) = &stop[1] else {
            panic!("expected ToolCall at index 1");
        };
        let po = tc
            .provider_options
            .as_ref()
            .expect("stream tool-call must carry caller in provider_options");
        let caller = po.get("anthropic").unwrap().get("caller").unwrap();
        assert_eq!(caller["type"], "code_execution_20250825");
        assert_eq!(caller["toolId"], "srvtoolu_01CodeExec");
        assert!(
            caller.get("tool_id").is_none(),
            "wire `snake_case` must be normalized"
        );
    }

    #[test]
    fn error_event_forces_error_finish() {
        let mut state = StreamState::new(vec![]);
        let _ = state.start_frames();
        let f = state.on_event(StreamEvent::Error {
            error: StreamError {
                _kind: Some("overloaded_error".into()),
                message: "overloaded".into(),
            },
        });
        assert!(matches!(f[0], StreamPart::Error { .. }));
        let tail = state.flush();
        if let StreamPart::Finish { finish_reason, .. } = tail.last().unwrap() {
            assert_eq!(finish_reason.unified, FinishReasonKind::Error);
        }
    }

    #[test]
    fn ping_and_other_events_are_inert() {
        let mut state = StreamState::new(vec![]);
        let _ = state.start_frames();
        assert!(state.on_event(StreamEvent::Ping).is_empty());
        assert!(state.on_event(StreamEvent::Other).is_empty());
    }

    #[test]
    fn compaction_block_emits_text_with_anthropic_marker() {
        let mut state = StreamState::new(vec![]);
        let _ = state.start_frames();
        let _ = state.on_event(StreamEvent::MessageStart {
            message: StreamMessageMeta {
                id: None,
                model: None,
                usage: empty_usage(),
            },
        });

        let f1 = state.on_event(StreamEvent::ContentBlockStart {
            index: 0,
            content_block: BlockStart::Compaction {
                content: Some("compacted prefix ".into()),
            },
        });
        // First: TextStart carrying anthropic.type="compaction".
        if let StreamPart::TextStart {
            provider_metadata, ..
        } = &f1[0]
        {
            let pm = provider_metadata.as_ref().expect("metadata set");
            assert_eq!(
                pm.get("anthropic").and_then(|b| b.get("type")),
                Some(&serde_json::Value::String("compaction".into()))
            );
        } else {
            panic!("expected TextStart, got {:?}", f1[0]);
        }
        // Inline content forwarded as TextDelta.
        assert!(
            matches!(&f1[1], StreamPart::TextDelta { delta, .. } if delta == "compacted prefix ")
        );

        // Subsequent compaction_delta → text-delta forwarded.
        let f2 = state.on_event(StreamEvent::ContentBlockDelta {
            index: 0,
            delta: BlockDelta::CompactionDelta {
                content: Some("tail".into()),
            },
        });
        assert!(matches!(&f2[0], StreamPart::TextDelta { delta, .. } if delta == "tail"));

        // null/empty content deltas are inert.
        assert!(
            state
                .on_event(StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: BlockDelta::CompactionDelta { content: None },
                })
                .is_empty()
        );
        assert!(
            state
                .on_event(StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: BlockDelta::CompactionDelta {
                        content: Some(String::new()),
                    },
                })
                .is_empty()
        );

        let stop = state.on_event(StreamEvent::ContentBlockStop { index: 0 });
        assert!(matches!(stop[0], StreamPart::TextEnd { .. }));
    }

    #[test]
    fn json_response_tool_streams_as_text_and_collapses_finish_reason() {
        // Mirrors upstream anthropic-language-model.ts:2229-2266: when
        // `usesJsonResponseTool` is on and the server opens a
        // `tool_use(name="json")` block, the stream emits text-start /
        // text-delta (forwarding input_json_delta as text) / text-end.
        // The closing `tool_use` finish reason collapses to `stop`.
        let mut state = StreamState::with_generate_id(
            vec![],
            None,
            /*mark_dyn=*/ false,
            /*uses_json=*/ true,
        );
        let _ = state.start_frames();
        let _ = state.on_event(StreamEvent::MessageStart {
            message: StreamMessageMeta {
                id: None,
                model: None,
                usage: empty_usage(),
            },
        });

        let open = state.on_event(StreamEvent::ContentBlockStart {
            index: 0,
            content_block: BlockStart::ToolUse {
                id: "tu_json".into(),
                name: "json".into(),
                input: None,
                caller: None,
            },
        });
        assert!(
            matches!(open[0], StreamPart::TextStart { .. }),
            "synthesized json tool must open as text"
        );

        let d1 = state.on_event(StreamEvent::ContentBlockDelta {
            index: 0,
            delta: BlockDelta::InputJsonDelta {
                partial_json: r#"{"city":"#.into(),
            },
        });
        let d2 = state.on_event(StreamEvent::ContentBlockDelta {
            index: 0,
            delta: BlockDelta::InputJsonDelta {
                partial_json: r#""Tokyo"}"#.into(),
            },
        });
        let mut accumulated = String::new();
        for f in d1.iter().chain(d2.iter()) {
            if let StreamPart::TextDelta { delta, .. } = f {
                accumulated.push_str(delta);
            } else {
                panic!("expected text-delta from json tool, got {f:?}");
            }
        }
        assert_eq!(accumulated, r#"{"city":"Tokyo"}"#);

        let stop = state.on_event(StreamEvent::ContentBlockStop { index: 0 });
        assert!(matches!(stop[0], StreamPart::TextEnd { .. }));

        let _ = state.on_event(StreamEvent::MessageDelta {
            delta: MessageDeltaInner {
                stop_reason: Some("tool_use".into()),
            },
            usage: None,
        });
        let tail = state.flush();
        let Some(StreamPart::Finish { finish_reason, .. }) = tail.last() else {
            panic!("expected trailing Finish");
        };
        assert_eq!(finish_reason.unified, FinishReasonKind::Stop);
        assert_eq!(finish_reason.raw.as_deref(), Some("tool_use"));
    }

    /// Verifies that streaming `server_tool_use` content blocks surface as a
    /// `ToolInputStart` (+ inline `ToolInputDelta`) instead of being dropped
    /// by the catch-all `BlockStart::Other`. Mirrors upstream
    /// `anthropic-language-model.ts:1671-1735`.
    #[test]
    fn stream_server_tool_use_emits_tool_input_lifecycle() {
        let mut state = StreamState::new(vec![]);
        let _ = state.start_frames();
        let _ = state.on_event(StreamEvent::MessageStart {
            message: StreamMessageMeta {
                id: Some("msg_1".into()),
                model: Some("claude-3-5".into()),
                usage: empty_usage(),
            },
        });
        let frames = state.on_event(StreamEvent::ContentBlockStart {
            index: 0,
            content_block: BlockStart::ServerToolUse {
                id: "srvtoolu_abc".into(),
                name: "web_search".into(),
                input: Some(serde_json::json!({"query": "rust"})),
            },
        });
        assert!(matches!(
            &frames[0],
            StreamPart::ToolInputStart { id, tool_name, provider_executed: Some(true), .. }
                if id == "srvtoolu_abc" && tool_name == "web_search"
        ));
        assert!(matches!(
            &frames[1],
            StreamPart::ToolInputDelta { id, delta, .. }
                if id == "srvtoolu_abc" && delta.contains("rust")
        ));
    }

    /// Verifies that streaming `*_tool_result` content blocks surface as a
    /// `ToolResult` part instead of being dropped. Mirrors upstream
    /// `anthropic-language-model.ts:1786-2019`.
    #[test]
    fn stream_web_search_tool_result_emits_tool_result_part() {
        let mut state = StreamState::new(vec![]);
        let _ = state.start_frames();
        let _ = state.on_event(StreamEvent::MessageStart {
            message: StreamMessageMeta {
                id: Some("msg_1".into()),
                model: Some("claude-3-5".into()),
                usage: empty_usage(),
            },
        });
        let payload = serde_json::json!({
            "tool_use_id": "srvtoolu_abc",
            "name": "web_search",
            "content": [{"type": "web_search_result", "url": "https://example.com"}]
        });
        let frames = state.on_event(StreamEvent::ContentBlockStart {
            index: 1,
            content_block: BlockStart::WebSearchToolResult(payload.clone()),
        });
        assert_eq!(frames.len(), 1);
        let StreamPart::ToolResult(tr) = &frames[0] else {
            panic!("expected ToolResult, got {:?}", frames[0]);
        };
        assert_eq!(tr.tool_call_id, "srvtoolu_abc");
        assert_eq!(tr.tool_name, "web_search");
        assert!(matches!(tr.output, ToolResultOutput::Json { .. }));
    }
}
