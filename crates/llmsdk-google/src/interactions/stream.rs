//! SSE stream transformer for the Gemini Interactions API.
//!
//! Mirrors `@ai-sdk/google/src/interactions/build-google-interactions-stream-transform.ts`.
//! Drives a wire `event_type` → `StreamPart` state machine that surfaces all
//! 9 upstream event kinds:
//!
//! - `interaction.created` → `ResponseMetadata`
//! - `step.start` → opens a per-index OpenBlock + emits `*-start`
//! - `step.delta` → routes by inner `delta.type` (text / image /
//!   text_annotation / thought_summary / thought_signature / arguments_delta /
//!   builtin tool call|result) to the matching open block
//! - `step.stop` → emits `*-end` (and `tool-call` / `tool-result` for tool
//!   blocks) then drops the slot
//! - `interaction.status_update` / `interaction.in_progress` /
//!   `interaction.requires_action` → updates `finish_status`
//! - `interaction.completed` → updates id / usage / serviceTier / status
//! - `error` → emits `Error` part + sets `finish_status = "failed"`
//!
//! Concurrent steps are framed by `index`; the slot map keeps them isolated
//! so a text delta at index `N` does not collide with a thought delta at `M`.
// Rust guideline compliant 2026-05-25

use std::collections::{HashMap, HashSet};

use futures::StreamExt;
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{ResponseMetadata, StreamPart, ToolCallPart};
use llmsdk_provider::shared::{ProviderMetadata, Warning};
use llmsdk_provider_utils::sse::SseEvent;
use serde_json::{Map as JsonMap, Value as JsonValue, json};

use super::extract_sources::{annotations_to_sources, builtin_tool_result_to_sources, source_key};

const BUILTIN_TOOL_CALL_TYPES: &[&str] = &[
    "google_search_call",
    "code_execution_call",
    "url_context_call",
    "file_search_call",
    "google_maps_call",
    "mcp_server_tool_call",
];

const BUILTIN_TOOL_RESULT_TYPES: &[&str] = &[
    "google_search_result",
    "code_execution_result",
    "url_context_result",
    "file_search_result",
    "google_maps_result",
    "mcp_server_tool_result",
];

fn is_builtin_tool_call_type(t: &str) -> bool {
    BUILTIN_TOOL_CALL_TYPES.contains(&t)
}
fn is_builtin_tool_result_type(t: &str) -> bool {
    BUILTIN_TOOL_RESULT_TYPES.contains(&t)
}

fn builtin_tool_name_from_call_type(t: &str) -> &str {
    t.strip_suffix("_call").unwrap_or(t)
}
fn builtin_tool_name_from_result_type(t: &str) -> &str {
    t.strip_suffix("_result").unwrap_or(t)
}

/// Per-slot OpenBlock state. Mirrors upstream's `OpenBlockState` union.
///
/// `id` fields are kept for parity with upstream's per-block diagnostic id
/// (returned in some traces) — Rust side reads them only via Debug formatter.
#[allow(
    dead_code,
    reason = "id fields are surfaced through Debug for diagnostic parity with upstream"
)]
#[derive(Debug)]
enum OpenBlock {
    Text {
        id: String,
    },
    Reasoning {
        id: String,
        signature: Option<String>,
    },
    Image {
        id: String,
        data: Option<String>,
        mime_type: Option<String>,
        uri: Option<String>,
    },
    FunctionCall {
        id: String,
        tool_call_id: String,
        tool_name: String,
        arguments_accum: String,
        signature: Option<String>,
    },
    BuiltinToolCall {
        id: String,
        block_type: String,
        tool_call_id: String,
        tool_name: String,
        arguments: JsonValue,
        emitted: bool,
    },
    BuiltinToolResult {
        id: String,
        block_type: String,
        call_id: String,
        tool_name: String,
        result: JsonValue,
        is_error: Option<bool>,
        emitted: bool,
    },
    /// `model_output` step opened before the first delta reveals whether
    /// its content is text or image. Promoted in-place on the first delta.
    PendingModelOutput {
        id: String,
    },
    Unknown {
        id: String,
    },
}

/// Drive a transformer from `event_stream` (raw Interactions SSE) to
/// `StreamPart`s. Mirrors the upstream `buildGoogleInteractionsStreamTransform`
/// shape with `stream-start` first, then per-event emits, and a single
/// `finish` part at the end.
///
/// `header_service_tier` is the value read from the `x-gemini-service-tier`
/// HTTP response header (defensive fallback — the Interactions API surfaces
/// the applied tier on `interaction.completed.interaction.service_tier`).
pub(crate) fn drive_stream<S>(
    warnings: Vec<Warning>,
    model_id: String,
    header_service_tier: Option<String>,
    events: S,
) -> impl futures::Stream<Item = Result<StreamPart, ProviderError>> + Send
where
    S: futures::Stream<Item = Result<SseEvent<JsonValue>, ProviderError>> + Send + 'static,
{
    async_stream::stream! {
        yield Ok(StreamPart::StreamStart { warnings });

        let mut interaction_id: Option<String> = None;
        let mut finish_status: Option<String> = None;
        let mut last_usage: Option<JsonValue> = None;
        let mut service_tier: Option<String> = header_service_tier;
        let mut has_function_call = false;

        let mut open_blocks: HashMap<i64, OpenBlock> = HashMap::new();
        let mut emitted_source_keys: HashSet<String> = HashSet::new();

        let mut events = Box::pin(events);
        while let Some(event) = events.next().await {
            match event {
                Err(e) => {
                    yield Err(e);
                    return;
                }
                Ok(SseEvent::ParseError { raw, message }) => {
                    yield Ok(StreamPart::Error {
                        error: json!({ "message": message, "raw": raw }),
                    });
                }
                Ok(SseEvent::Data(value)) => {
                    for part in handle_event(
                        &value,
                        &mut interaction_id,
                        &mut finish_status,
                        &mut last_usage,
                        &mut service_tier,
                        &mut has_function_call,
                        &mut open_blocks,
                        &mut emitted_source_keys,
                        &model_id,
                    ) {
                        yield Ok(part);
                    }
                }
            }
        }

        // Drain any still-open blocks defensively (mirrors upstream's
        // single-finish guarantee — we never leak an open block past the
        // terminal `finish` part).
        let still_open: Vec<i64> = open_blocks.keys().copied().collect();
        for idx in still_open {
            if let Some(block) = open_blocks.remove(&idx) {
                for part in close_open_block(block, &interaction_id, &mut emitted_source_keys) {
                    yield Ok(part);
                }
            }
        }

        let finish_reason = super::model::map_finish_reason_from_status(finish_status.as_deref());
        let usage = super::model::parse_usage(last_usage.as_ref());
        let provider_metadata = build_finish_provider_metadata(
            interaction_id.as_deref(),
            service_tier.as_deref(),
        );

        yield Ok(StreamPart::Finish {
            finish_reason,
            usage,
            provider_metadata,
        });
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "state machine threads the full stream context through one call"
)]
fn handle_event(
    value: &JsonValue,
    interaction_id: &mut Option<String>,
    finish_status: &mut Option<String>,
    last_usage: &mut Option<JsonValue>,
    service_tier: &mut Option<String>,
    has_function_call: &mut bool,
    open_blocks: &mut HashMap<i64, OpenBlock>,
    emitted_source_keys: &mut HashSet<String>,
    model_id: &str,
) -> Vec<StreamPart> {
    let mut out = Vec::new();
    let event_type = value
        .get("event_type")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    match event_type {
        "interaction.created" => {
            let interaction = value.get("interaction");
            if let Some(id) = interaction
                .and_then(|i| i.get("id"))
                .and_then(JsonValue::as_str)
                .filter(|s| !s.is_empty())
            {
                *interaction_id = Some(id.to_owned());
            }
            let timestamp = interaction
                .and_then(|i| i.get("created"))
                .and_then(JsonValue::as_str)
                .map(str::to_owned);
            let id_for_event = interaction_id.clone();
            out.push(StreamPart::ResponseMetadata(ResponseMetadata {
                id: id_for_event,
                timestamp,
                model_id: Some(model_id.to_owned()),
                headers: None,
            }));
        }
        "step.start" => {
            let Some(index) = value.get("index").and_then(JsonValue::as_i64) else {
                return out;
            };
            let step = value.get("step").cloned().unwrap_or(JsonValue::Null);
            let block_id = format!(
                "{}:{}",
                interaction_id.as_deref().unwrap_or("interaction"),
                index
            );
            let step_type = step.get("type").and_then(JsonValue::as_str).unwrap_or("");

            match step_type {
                "model_output" => {
                    let initial = step
                        .get("content")
                        .and_then(JsonValue::as_array)
                        .and_then(|a| a.first());
                    match initial
                        .and_then(|b| b.get("type"))
                        .and_then(JsonValue::as_str)
                    {
                        Some("text") => {
                            open_blocks.insert(
                                index,
                                OpenBlock::Text {
                                    id: block_id.clone(),
                                },
                            );
                            out.push(StreamPart::TextStart {
                                id: block_id.clone(),
                                provider_metadata: None,
                            });
                            if let Some(b) = initial {
                                let mut id_gen = block_id_gen(&block_id);
                                let sources = annotations_to_sources(
                                    b.get("annotations"),
                                    &mut id_gen,
                                    emitted_source_keys,
                                );
                                for src in sources {
                                    out.push(StreamPart::Source(src));
                                }
                            }
                        }
                        Some("image") => {
                            open_blocks.insert(
                                index,
                                OpenBlock::Image {
                                    id: block_id,
                                    data: initial
                                        .and_then(|b| b.get("data"))
                                        .and_then(JsonValue::as_str)
                                        .map(str::to_owned),
                                    mime_type: initial
                                        .and_then(|b| b.get("mime_type"))
                                        .and_then(JsonValue::as_str)
                                        .map(str::to_owned),
                                    uri: initial
                                        .and_then(|b| b.get("uri"))
                                        .and_then(JsonValue::as_str)
                                        .map(str::to_owned),
                                },
                            );
                        }
                        _ => {
                            open_blocks
                                .insert(index, OpenBlock::PendingModelOutput { id: block_id });
                        }
                    }
                }
                "thought" => {
                    let signature = step
                        .get("signature")
                        .and_then(JsonValue::as_str)
                        .map(str::to_owned);
                    open_blocks.insert(
                        index,
                        OpenBlock::Reasoning {
                            id: block_id.clone(),
                            signature,
                        },
                    );
                    out.push(StreamPart::ReasoningStart {
                        id: block_id.clone(),
                        provider_metadata: None,
                    });
                    if let Some(summary) = step.get("summary").and_then(JsonValue::as_array) {
                        for item in summary {
                            if item.get("type").and_then(JsonValue::as_str) == Some("text") {
                                if let Some(t) = item.get("text").and_then(JsonValue::as_str) {
                                    out.push(StreamPart::ReasoningDelta {
                                        id: block_id.clone(),
                                        delta: t.to_owned(),
                                        provider_metadata: None,
                                    });
                                }
                            }
                        }
                    }
                }
                "function_call" => {
                    let tool_call_id = step
                        .get("id")
                        .and_then(JsonValue::as_str)
                        .unwrap_or(&block_id)
                        .to_owned();
                    let tool_name = step
                        .get("name")
                        .and_then(JsonValue::as_str)
                        .unwrap_or("unknown")
                        .to_owned();
                    let signature = step
                        .get("signature")
                        .and_then(JsonValue::as_str)
                        .map(str::to_owned);
                    *has_function_call = true;
                    open_blocks.insert(
                        index,
                        OpenBlock::FunctionCall {
                            id: block_id,
                            tool_call_id: tool_call_id.clone(),
                            tool_name: tool_name.clone(),
                            arguments_accum: String::new(),
                            signature,
                        },
                    );
                    out.push(StreamPart::ToolInputStart {
                        id: tool_call_id,
                        tool_name,
                        provider_executed: None,
                        dynamic: None,
                        title: None,
                        provider_metadata: None,
                    });
                }
                t if is_builtin_tool_call_type(t) => {
                    let tool_name = if t == "mcp_server_tool_call" {
                        step.get("name")
                            .and_then(JsonValue::as_str)
                            .unwrap_or("mcp_server_tool")
                            .to_owned()
                    } else {
                        builtin_tool_name_from_call_type(t).to_owned()
                    };
                    let tool_call_id = step
                        .get("id")
                        .and_then(JsonValue::as_str)
                        .unwrap_or(&block_id)
                        .to_owned();
                    open_blocks.insert(
                        index,
                        OpenBlock::BuiltinToolCall {
                            id: block_id,
                            block_type: t.to_owned(),
                            tool_call_id,
                            tool_name,
                            arguments: step.get("arguments").cloned().unwrap_or(json!({})),
                            emitted: false,
                        },
                    );
                }
                t if is_builtin_tool_result_type(t) => {
                    let tool_name = if t == "mcp_server_tool_result" {
                        step.get("name")
                            .and_then(JsonValue::as_str)
                            .unwrap_or("mcp_server_tool")
                            .to_owned()
                    } else {
                        builtin_tool_name_from_result_type(t).to_owned()
                    };
                    let call_id = step
                        .get("call_id")
                        .and_then(JsonValue::as_str)
                        .unwrap_or(&block_id)
                        .to_owned();
                    open_blocks.insert(
                        index,
                        OpenBlock::BuiltinToolResult {
                            id: block_id,
                            block_type: t.to_owned(),
                            call_id,
                            tool_name,
                            result: step.get("result").cloned().unwrap_or(JsonValue::Null),
                            is_error: step.get("is_error").and_then(JsonValue::as_bool),
                            emitted: false,
                        },
                    );
                }
                _ => {
                    open_blocks.insert(index, OpenBlock::Unknown { id: block_id });
                }
            }
        }
        "step.delta" => {
            let Some(index) = value.get("index").and_then(JsonValue::as_i64) else {
                return out;
            };
            let Some(delta) = value.get("delta") else {
                return out;
            };
            let dtype = delta.get("type").and_then(JsonValue::as_str).unwrap_or("");

            // Promote pending model_output → text on first text-shaped delta.
            if let Some(OpenBlock::PendingModelOutput { id }) = open_blocks.get(&index) {
                if matches!(dtype, "text" | "text_annotation" | "text_annotation_delta") {
                    let new_id = id.clone();
                    open_blocks.insert(index, OpenBlock::Text { id: new_id.clone() });
                    out.push(StreamPart::TextStart {
                        id: new_id,
                        provider_metadata: None,
                    });
                }
            }

            // Image deltas can fire on Pending or Text slots: emit inline.
            if dtype == "image" {
                let media_type = delta
                    .get("mime_type")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("image/png")
                    .to_owned();
                let provider_metadata =
                    build_block_provider_metadata(interaction_id.as_deref(), None);
                if let Some(data) = delta
                    .get("data")
                    .and_then(JsonValue::as_str)
                    .filter(|s| !s.is_empty())
                {
                    out.push(StreamPart::File(
                        llmsdk_provider::language_model::FilePart {
                            media_type,
                            data: llmsdk_provider::shared::FileData::Data {
                                data: llmsdk_provider::shared::FileBytes::Base64(data.to_owned()),
                            },
                            filename: None,
                            provider_options: provider_metadata.map(provider_options_from_map),
                        },
                    ));
                } else if let Some(uri) = delta
                    .get("uri")
                    .and_then(JsonValue::as_str)
                    .filter(|s| !s.is_empty())
                {
                    out.push(StreamPart::File(
                        llmsdk_provider::language_model::FilePart {
                            media_type,
                            data: llmsdk_provider::shared::FileData::Url {
                                url: uri.to_owned(),
                            },
                            filename: None,
                            provider_options: provider_metadata.map(provider_options_from_map),
                        },
                    ));
                }
                // Image slot already-eager: clear data so step.stop won't duplicate.
                if let Some(OpenBlock::Image { data, uri, .. }) = open_blocks.get_mut(&index) {
                    *data = None;
                    *uri = None;
                }
                return out;
            }

            // Route remaining delta types by open block kind.
            let Some(open) = open_blocks.get_mut(&index) else {
                return out;
            };
            match open {
                OpenBlock::Text { id } => {
                    if dtype == "text" {
                        if let Some(text) = delta
                            .get("text")
                            .and_then(JsonValue::as_str)
                            .filter(|s| !s.is_empty())
                        {
                            out.push(StreamPart::TextDelta {
                                id: id.clone(),
                                delta: text.to_owned(),
                                provider_metadata: None,
                            });
                        }
                    } else if dtype == "text_annotation" || dtype == "text_annotation_delta" {
                        let block_id_for_gen = id.clone();
                        let mut id_gen = block_id_gen(&block_id_for_gen);
                        let sources = annotations_to_sources(
                            delta.get("annotations"),
                            &mut id_gen,
                            emitted_source_keys,
                        );
                        for src in sources {
                            out.push(StreamPart::Source(src));
                        }
                    }
                }
                OpenBlock::Reasoning { id, signature } => match dtype {
                    "thought_summary" => {
                        let item = delta.get("content");
                        if let Some(item) = item {
                            if item.get("type").and_then(JsonValue::as_str) == Some("text") {
                                if let Some(t) = item.get("text").and_then(JsonValue::as_str) {
                                    out.push(StreamPart::ReasoningDelta {
                                        id: id.clone(),
                                        delta: t.to_owned(),
                                        provider_metadata: None,
                                    });
                                }
                            }
                        }
                    }
                    "thought_signature" => {
                        if let Some(sig) = delta.get("signature").and_then(JsonValue::as_str) {
                            *signature = Some(sig.to_owned());
                        }
                    }
                    _ => {}
                },
                OpenBlock::FunctionCall {
                    tool_call_id,
                    arguments_accum,
                    signature,
                    ..
                } if dtype == "arguments_delta" => {
                    let slice = delta
                        .get("arguments")
                        .and_then(JsonValue::as_str)
                        .unwrap_or("");
                    if !slice.is_empty() {
                        arguments_accum.push_str(slice);
                        out.push(StreamPart::ToolInputDelta {
                            id: tool_call_id.clone(),
                            delta: slice.to_owned(),
                            provider_metadata: None,
                        });
                    }
                    if let Some(new_id) = delta.get("id").and_then(JsonValue::as_str) {
                        new_id.clone_into(tool_call_id);
                    }
                    if let Some(sig) = delta.get("signature").and_then(JsonValue::as_str) {
                        *signature = Some(sig.to_owned());
                    }
                    *has_function_call = true;
                }
                OpenBlock::BuiltinToolCall {
                    block_type,
                    tool_call_id,
                    tool_name,
                    arguments,
                    ..
                } if dtype == block_type.as_str() => {
                    if let Some(new_id) = delta.get("id").and_then(JsonValue::as_str) {
                        new_id.clone_into(tool_call_id);
                    }
                    if let Some(args) = delta.get("arguments") {
                        if args.is_object() {
                            *arguments = args.clone();
                        }
                    }
                    if block_type == "mcp_server_tool_call" {
                        if let Some(name) = delta.get("name").and_then(JsonValue::as_str) {
                            name.clone_into(tool_name);
                        }
                    }
                }
                OpenBlock::BuiltinToolResult {
                    block_type,
                    call_id,
                    tool_name,
                    result,
                    is_error,
                    ..
                } if dtype == block_type.as_str() => {
                    if let Some(new_id) = delta.get("call_id").and_then(JsonValue::as_str) {
                        new_id.clone_into(call_id);
                    }
                    if let Some(r) = delta.get("result") {
                        *result = r.clone();
                    }
                    if let Some(e) = delta.get("is_error").and_then(JsonValue::as_bool) {
                        *is_error = Some(e);
                    }
                    if block_type == "mcp_server_tool_result" {
                        if let Some(name) = delta.get("name").and_then(JsonValue::as_str) {
                            name.clone_into(tool_name);
                        }
                    }
                }
                _ => {}
            }
        }
        "step.stop" => {
            let Some(index) = value.get("index").and_then(JsonValue::as_i64) else {
                return out;
            };
            if let Some(block) = open_blocks.remove(&index) {
                out.extend(close_open_block(block, interaction_id, emitted_source_keys));
            }
        }
        "interaction.status_update" | "interaction.in_progress" | "interaction.requires_action" => {
            if let Some(s) = value.get("status").and_then(JsonValue::as_str) {
                *finish_status = Some(s.to_owned());
            } else {
                match event_type {
                    "interaction.requires_action" => {
                        *finish_status = Some("requires_action".to_owned());
                    }
                    "interaction.in_progress" => {
                        *finish_status = Some("in_progress".to_owned());
                    }
                    _ => {}
                }
            }
        }
        "interaction.completed" => {
            let interaction = value.get("interaction");
            if let Some(id) = interaction
                .and_then(|i| i.get("id"))
                .and_then(JsonValue::as_str)
                .filter(|s| !s.is_empty())
            {
                *interaction_id = Some(id.to_owned());
            }
            if let Some(s) = interaction
                .and_then(|i| i.get("status"))
                .and_then(JsonValue::as_str)
            {
                *finish_status = Some(s.to_owned());
            }
            if let Some(u) = interaction.and_then(|i| i.get("usage")) {
                *last_usage = Some(u.clone());
            }
            if let Some(t) = interaction
                .and_then(|i| i.get("service_tier"))
                .and_then(JsonValue::as_str)
            {
                *service_tier = Some(t.to_owned());
            }
        }
        "error" => {
            *finish_status = Some("failed".to_owned());
            let error = value
                .get("error")
                .cloned()
                .unwrap_or_else(|| json!({ "message": "Unknown interaction error" }));
            out.push(StreamPart::Error { error });
        }
        _ => {}
    }
    out
}

/// Emit the `*-end` (and tool-call / tool-result) parts for a slot we are
/// closing — either on `step.stop` or as a defensive drain at flush time.
fn close_open_block(
    block: OpenBlock,
    interaction_id: &Option<String>,
    emitted_source_keys: &mut HashSet<String>,
) -> Vec<StreamPart> {
    let mut out = Vec::new();
    match block {
        OpenBlock::Text { id } => {
            out.push(StreamPart::TextEnd {
                id,
                provider_metadata: build_block_provider_metadata(interaction_id.as_deref(), None)
                    .map(provider_metadata_from_map),
            });
        }
        OpenBlock::Reasoning { id, signature } => {
            out.push(StreamPart::ReasoningEnd {
                id,
                provider_metadata: build_block_provider_metadata(
                    interaction_id.as_deref(),
                    signature.as_deref(),
                )
                .map(provider_metadata_from_map),
            });
        }
        OpenBlock::Image {
            data,
            uri,
            mime_type,
            ..
        } => {
            let media_type = mime_type.unwrap_or_else(|| "image/png".to_owned());
            let provider_options = build_block_provider_metadata(interaction_id.as_deref(), None)
                .map(provider_options_from_map);
            if let Some(data) = data.filter(|s| !s.is_empty()) {
                out.push(StreamPart::File(
                    llmsdk_provider::language_model::FilePart {
                        media_type,
                        data: llmsdk_provider::shared::FileData::Data {
                            data: llmsdk_provider::shared::FileBytes::Base64(data),
                        },
                        filename: None,
                        provider_options,
                    },
                ));
            } else if let Some(uri) = uri.filter(|s| !s.is_empty()) {
                out.push(StreamPart::File(
                    llmsdk_provider::language_model::FilePart {
                        media_type,
                        data: llmsdk_provider::shared::FileData::Url { url: uri },
                        filename: None,
                        provider_options,
                    },
                ));
            }
        }
        OpenBlock::FunctionCall {
            tool_call_id,
            tool_name,
            arguments_accum,
            signature,
            ..
        } => {
            let accumulated = if arguments_accum.is_empty() {
                "{}".to_owned()
            } else {
                arguments_accum
            };
            out.push(StreamPart::ToolInputEnd {
                id: tool_call_id.clone(),
                provider_metadata: None,
            });
            let input_value: JsonValue =
                serde_json::from_str(&accumulated).unwrap_or(JsonValue::Null);
            let provider_options =
                build_block_provider_metadata(interaction_id.as_deref(), signature.as_deref())
                    .map(provider_options_from_map);
            out.push(StreamPart::ToolCall(ToolCallPart {
                tool_call_id,
                tool_name,
                input: input_value,
                provider_executed: None,
                dynamic: None,
                provider_options,
            }));
        }
        OpenBlock::BuiltinToolCall {
            tool_call_id,
            tool_name,
            arguments,
            emitted,
            ..
        } => {
            if !emitted {
                out.push(StreamPart::ToolCall(ToolCallPart {
                    tool_call_id,
                    tool_name,
                    input: arguments,
                    provider_executed: Some(true),
                    dynamic: None,
                    provider_options: None,
                }));
            }
        }
        OpenBlock::BuiltinToolResult {
            block_type,
            call_id,
            tool_name,
            result,
            emitted,
            ..
        } => {
            if !emitted {
                use llmsdk_provider::language_model::{ToolResult, ToolResultOutput};
                out.push(StreamPart::ToolResult(ToolResult {
                    tool_call_id: call_id.clone(),
                    tool_name,
                    output: ToolResultOutput::Json {
                        value: result.clone(),
                        provider_options: None,
                    },
                    provider_metadata: None,
                }));
                let mut id_gen = block_id_gen(&call_id);
                let sources =
                    builtin_tool_result_to_sources(&block_type, Some(&result), &mut id_gen);
                for src in sources {
                    let key = source_key(&src);
                    if emitted_source_keys.insert(key) {
                        out.push(StreamPart::Source(src));
                    }
                }
            }
        }
        OpenBlock::PendingModelOutput { .. } | OpenBlock::Unknown { .. } => {}
    }
    out
}

fn block_id_gen(prefix: &str) -> impl FnMut() -> String {
    let prefix = prefix.to_owned();
    let mut n = 0usize;
    move || {
        n += 1;
        format!("{prefix}:src-{n}")
    }
}

fn build_block_provider_metadata(
    interaction_id: Option<&str>,
    signature: Option<&str>,
) -> Option<JsonMap<String, JsonValue>> {
    let mut g = JsonMap::new();
    if let Some(id) = interaction_id {
        g.insert("interactionId".into(), JsonValue::String(id.to_owned()));
    }
    if let Some(sig) = signature {
        g.insert("signature".into(), JsonValue::String(sig.to_owned()));
    }
    if g.is_empty() {
        None
    } else {
        let mut out = JsonMap::new();
        out.insert("google".into(), JsonValue::Object(g));
        Some(out)
    }
}

fn build_finish_provider_metadata(
    interaction_id: Option<&str>,
    service_tier: Option<&str>,
) -> Option<ProviderMetadata> {
    let mut g = JsonMap::new();
    if let Some(id) = interaction_id {
        g.insert("interactionId".into(), JsonValue::String(id.to_owned()));
    }
    if let Some(t) = service_tier {
        g.insert("serviceTier".into(), JsonValue::String(t.to_owned()));
    }
    if g.is_empty() {
        return None;
    }
    let mut pm = ProviderMetadata::new();
    pm.insert("google".to_owned(), g);
    Some(pm)
}

fn provider_metadata_from_map(map: JsonMap<String, JsonValue>) -> ProviderMetadata {
    let mut pm = ProviderMetadata::new();
    for (k, v) in map {
        if let JsonValue::Object(obj) = v {
            pm.insert(k, obj);
        }
    }
    pm
}

fn provider_options_from_map(
    map: JsonMap<String, JsonValue>,
) -> llmsdk_provider::shared::ProviderOptions {
    let mut po = llmsdk_provider::shared::ProviderOptions::new();
    for (k, v) in map {
        if let JsonValue::Object(obj) = v {
            po.insert(k, obj);
        }
    }
    po
}
