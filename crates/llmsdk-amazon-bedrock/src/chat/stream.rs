//! Bedrock ConverseStream decoder + state machine.
//!
//! The wire format is AWS binary EventStream frames; each event JSON is the
//! frame payload (after the `:event-type` header tells us which event it is).
//! We use [`llmsdk_provider_utils::aws_eventstream::decode_event_stream`] to
//! pull frames off the byte stream, parse the payload JSON, and feed it into
//! the same state machine pattern other providers use (`text-start /
//! text-delta / text-end`, tool-use accumulator, reasoning blocks, ...).
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;

use bytes::Bytes;
use futures::Stream;
use futures::stream::StreamExt;
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    FinishReason, FinishReasonKind, ResponseMetadata, StreamPart, ToolCallPart, Usage,
};
use llmsdk_provider::shared::{ProviderMetadata, Warning};
use llmsdk_provider_utils::aws_eventstream::decode_event_stream;
use serde_json::Value;

use super::finish_reason::map_finish_reason;
use super::normalize_tool_call_id::normalize_tool_call_id;
use super::usage::convert_usage;
use super::wire::{BedrockUsage, StreamChunk};
use crate::PROVIDER_ID;

/// Build the unified [`StreamPart`] stream from a Bedrock EventStream byte
/// stream.
#[allow(
    clippy::too_many_arguments,
    reason = "single-call helper mirroring upstream build_stream surface"
)]
pub(crate) fn build_stream<S>(
    bytes: S,
    warnings: Vec<Warning>,
    is_mistral: bool,
    uses_json_response_tool: bool,
    response_headers: HashMap<String, String>,
    model_id: String,
    include_raw: bool,
    generate_id: Option<std::sync::Arc<crate::config::GenerateIdFn>>,
) -> impl Stream<Item = Result<StreamPart, ProviderError>> + Send + 'static
where
    S: Stream<Item = Result<Bytes, ProviderError>> + Send + 'static,
{
    let events = decode_event_stream(bytes);
    let state = State::new(warnings, is_mistral, uses_json_response_tool, generate_id);
    async_stream::stream! {
        // Stream-start + response-metadata frames (mirrors upstream).
        yield Ok(StreamPart::StreamStart { warnings: state.initial_warnings() });
        yield Ok(StreamPart::ResponseMetadata(ResponseMetadata {
            id: response_headers.get("x-amzn-requestid").cloned(),
            timestamp: response_headers.get("date").cloned(),
            model_id: Some(model_id),
            headers: None,
        }));

        let mut state = state;
        let mut events = Box::pin(events);
        while let Some(event_res) = events.next().await {
            match event_res {
                Ok(message) => {
                    if include_raw {
                        let raw_payload = serde_json::from_slice::<Value>(&message.payload)
                            .unwrap_or(Value::Null);
                        yield Ok(StreamPart::Raw { raw_value: raw_payload });
                    }
                    let event_type = message.event_type().map(str::to_owned);
                    let payload_json = serde_json::from_slice::<Value>(&message.payload)
                        .unwrap_or(Value::Null);
                    // Bedrock wraps the inner payload by event-type name (e.g.
                    // `contentBlockDelta`). We mirror that by emitting a chunk
                    // whose single populated field is the event type.
                    let chunk_value = if let Some(name) = event_type {
                        let mut map = serde_json::Map::new();
                        map.insert(name, payload_json);
                        Value::Object(map)
                    } else {
                        payload_json
                    };
                    let chunk: StreamChunk = serde_json::from_value(chunk_value.clone())
                        .unwrap_or_default();
                    for part in state.on_chunk(chunk) {
                        yield Ok(part);
                    }
                }
                Err(e) => {
                    yield Err(e);
                    return;
                }
            }
        }

        for part in state.flush() {
            yield Ok(part);
        }
    }
}

/// In-progress content blocks indexed by Bedrock block id.
#[derive(Debug, Clone)]
enum Block {
    Text,
    Reasoning,
    ToolCall {
        tool_call_id: String,
        tool_name: String,
        json_buffer: String,
        is_json_response_tool: bool,
    },
}

struct State {
    initial_warnings: Vec<Warning>,
    is_mistral: bool,
    uses_json_response_tool: bool,
    finish_reason: FinishReason,
    usage: Option<BedrockUsage>,
    blocks: HashMap<u32, Block>,
    stop_sequence: Option<String>,
    provider_metadata: Option<ProviderMetadata>,
    is_json_response_from_tool: bool,
    start_emitted: bool,
    generate_id: Option<std::sync::Arc<crate::config::GenerateIdFn>>,
}

impl State {
    fn new(
        warnings: Vec<Warning>,
        is_mistral: bool,
        uses_json_response_tool: bool,
        generate_id: Option<std::sync::Arc<crate::config::GenerateIdFn>>,
    ) -> Self {
        Self {
            initial_warnings: warnings,
            is_mistral,
            uses_json_response_tool,
            finish_reason: FinishReason::new(FinishReasonKind::Other),
            usage: None,
            blocks: HashMap::new(),
            stop_sequence: None,
            provider_metadata: None,
            is_json_response_from_tool: false,
            start_emitted: false,
            generate_id,
        }
    }

    fn initial_warnings(&self) -> Vec<Warning> {
        self.initial_warnings.clone()
    }

    fn on_chunk(&mut self, chunk: StreamChunk) -> Vec<StreamPart> {
        let mut out: Vec<StreamPart> = Vec::new();

        // ----- error variants -----
        if let Some(err) = chunk.internal_server_exception {
            self.finish_reason = FinishReason::new(FinishReasonKind::Error);
            out.push(StreamPart::Error { error: err });
            return out;
        }
        if let Some(err) = chunk.model_stream_error_exception {
            self.finish_reason = FinishReason::new(FinishReasonKind::Error);
            out.push(StreamPart::Error { error: err });
            return out;
        }
        if let Some(err) = chunk.throttling_exception {
            self.finish_reason = FinishReason::new(FinishReasonKind::Error);
            out.push(StreamPart::Error { error: err });
            return out;
        }
        if let Some(err) = chunk.validation_exception {
            self.finish_reason = FinishReason::new(FinishReasonKind::Error);
            out.push(StreamPart::Error { error: err });
            return out;
        }

        // ----- contentBlockStart -----
        if let Some(start) = chunk.content_block_start {
            self.start_emitted = true;
            let idx = start.content_block_index.unwrap_or(0);
            if let Some(start_payload) = start.start
                && let Some(tu) = start_payload.tool_use
            {
                // Mirrors upstream `amazon-bedrock-chat-language-model.ts:557` /
                // `:561` — missing `toolUseId` / `name` flow through the
                // user-supplied generator instead of being surfaced blank.
                let tool_use_id = tu
                    .tool_use_id
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| super::parse_response::synth_id(self.generate_id.as_ref()));
                let tool_name = tu.name.filter(|s| !s.is_empty()).unwrap_or_else(|| {
                    format!(
                        "tool-{}",
                        super::parse_response::synth_id(self.generate_id.as_ref())
                    )
                });
                let is_json_response_tool = self.uses_json_response_tool && tool_name == "json";
                let normalized = normalize_tool_call_id(&tool_use_id, self.is_mistral);
                self.blocks.insert(
                    idx,
                    Block::ToolCall {
                        tool_call_id: normalized.clone(),
                        tool_name: tool_name.clone(),
                        json_buffer: String::new(),
                        is_json_response_tool,
                    },
                );
                if !is_json_response_tool {
                    out.push(StreamPart::ToolInputStart {
                        id: normalized,
                        tool_name,
                        provider_executed: None,
                        dynamic: None,
                        title: None,
                        provider_metadata: None,
                    });
                }
            } else {
                // Text block start implicit in upstream — emit text-start
                // only when we know it's text. Defer to first delta.
            }
        }

        // ----- contentBlockDelta -----
        if let Some(delta) = chunk.content_block_delta {
            let idx = delta.content_block_index.unwrap_or(0);
            if let Some(payload) = delta.delta {
                if let Some(text) = payload.text {
                    // Lazy text-start if we haven't seen one for this block yet.
                    if !matches!(self.blocks.get(&idx), Some(Block::Text)) {
                        self.blocks.insert(idx, Block::Text);
                        out.push(StreamPart::TextStart {
                            id: idx.to_string(),
                            provider_metadata: None,
                        });
                    }
                    out.push(StreamPart::TextDelta {
                        id: idx.to_string(),
                        delta: text,
                        provider_metadata: None,
                    });
                } else if let Some(tu) = payload.tool_use {
                    let chunk = tu.input.unwrap_or_default();
                    if let Some(Block::ToolCall {
                        tool_call_id,
                        json_buffer,
                        is_json_response_tool,
                        ..
                    }) = self.blocks.get_mut(&idx)
                    {
                        json_buffer.push_str(&chunk);
                        if !*is_json_response_tool {
                            out.push(StreamPart::ToolInputDelta {
                                id: tool_call_id.clone(),
                                delta: chunk,
                                provider_metadata: None,
                            });
                        }
                    }
                } else if let Some(reasoning) = payload.reasoning_content {
                    if !matches!(self.blocks.get(&idx), Some(Block::Reasoning)) {
                        self.blocks.insert(idx, Block::Reasoning);
                        out.push(StreamPart::ReasoningStart {
                            id: idx.to_string(),
                            provider_metadata: None,
                        });
                    }
                    if let Some(text) = reasoning.text {
                        out.push(StreamPart::ReasoningDelta {
                            id: idx.to_string(),
                            delta: text,
                            provider_metadata: None,
                        });
                    } else if let Some(sig) = reasoning.signature {
                        let mut meta: ProviderMetadata = HashMap::new();
                        let payload = serde_json::json!({ "signature": sig });
                        let map = payload.as_object().cloned().unwrap_or_default();
                        meta.insert(PROVIDER_ID.to_owned(), map.clone());
                        meta.insert("bedrock".to_owned(), map);
                        out.push(StreamPart::ReasoningDelta {
                            id: idx.to_string(),
                            delta: String::new(),
                            provider_metadata: Some(meta),
                        });
                    } else if let Some(data) = reasoning.data {
                        let mut meta: ProviderMetadata = HashMap::new();
                        let payload = serde_json::json!({ "redactedData": data });
                        let map = payload.as_object().cloned().unwrap_or_default();
                        meta.insert(PROVIDER_ID.to_owned(), map.clone());
                        meta.insert("bedrock".to_owned(), map);
                        out.push(StreamPart::ReasoningDelta {
                            id: idx.to_string(),
                            delta: String::new(),
                            provider_metadata: Some(meta),
                        });
                    }
                }
            }
        }

        // ----- contentBlockStop -----
        if let Some(stop) = chunk.content_block_stop {
            let idx = stop.content_block_index.unwrap_or(0);
            if let Some(block) = self.blocks.remove(&idx) {
                match block {
                    Block::Text => {
                        out.push(StreamPart::TextEnd {
                            id: idx.to_string(),
                            provider_metadata: None,
                        });
                    }
                    Block::Reasoning => {
                        out.push(StreamPart::ReasoningEnd {
                            id: idx.to_string(),
                            provider_metadata: None,
                        });
                    }
                    Block::ToolCall {
                        tool_call_id,
                        tool_name,
                        json_buffer,
                        is_json_response_tool,
                    } => {
                        if is_json_response_tool {
                            self.is_json_response_from_tool = true;
                            out.push(StreamPart::TextStart {
                                id: idx.to_string(),
                                provider_metadata: None,
                            });
                            out.push(StreamPart::TextDelta {
                                id: idx.to_string(),
                                delta: json_buffer,
                                provider_metadata: None,
                            });
                            out.push(StreamPart::TextEnd {
                                id: idx.to_string(),
                                provider_metadata: None,
                            });
                        } else {
                            out.push(StreamPart::ToolInputEnd {
                                id: tool_call_id.clone(),
                                provider_metadata: None,
                            });
                            let buf = if json_buffer.is_empty() {
                                "{}".to_owned()
                            } else {
                                json_buffer
                            };
                            let input = serde_json::from_str::<Value>(&buf)
                                .unwrap_or(Value::Object(Default::default()));
                            out.push(StreamPart::ToolCall(ToolCallPart {
                                tool_call_id,
                                tool_name,
                                input,
                                provider_executed: None,
                                dynamic: None,
                                provider_options: None,
                            }));
                        }
                    }
                }
            }
        }

        // ----- messageStop -----
        if let Some(stop) = chunk.message_stop {
            self.finish_reason =
                map_finish_reason(stop.stop_reason.as_deref(), self.is_json_response_from_tool);
            self.stop_sequence = stop
                .additional_model_response_fields
                .as_ref()
                .and_then(|v| v.get("delta"))
                .and_then(|v| v.get("stop_sequence"))
                .and_then(Value::as_str)
                .map(str::to_owned);
        }

        // ----- metadata -----
        if let Some(meta) = chunk.metadata {
            if let Some(usage) = meta.usage {
                self.usage = Some(usage);
            }
            let mut payload = serde_json::Map::new();
            if let Some(usage) = &self.usage
                && (usage.cache_write_input_tokens.is_some() || usage.cache_details.is_some())
            {
                let mut inner = serde_json::Map::new();
                if let Some(v) = usage.cache_write_input_tokens {
                    inner.insert("cacheWriteInputTokens".to_owned(), Value::from(v));
                }
                if let Some(details) = usage.cache_details.clone() {
                    inner.insert("cacheDetails".to_owned(), details);
                }
                payload.insert("usage".to_owned(), Value::Object(inner));
            }
            if let Some(t) = meta.trace {
                payload.insert("trace".to_owned(), t);
            }
            if let Some(p) = meta.performance_config {
                payload.insert("performanceConfig".to_owned(), p);
            }
            if let Some(s) = meta.service_tier {
                payload.insert("serviceTier".to_owned(), s);
            }
            if !payload.is_empty() {
                let mut meta_map: ProviderMetadata = HashMap::new();
                meta_map.insert(PROVIDER_ID.to_owned(), payload.clone());
                meta_map.insert("bedrock".to_owned(), payload);
                self.provider_metadata = Some(meta_map);
            }
        }

        out
    }

    fn flush(mut self) -> Vec<StreamPart> {
        let mut out: Vec<StreamPart> = Vec::new();
        let usage_value = convert_usage(self.usage.take());
        if self.is_json_response_from_tool || self.stop_sequence.is_some() {
            // Mirror upstream: augment provider metadata with the json-response
            // flag and stopSequence on flush.
            let mut payload = self
                .provider_metadata
                .as_ref()
                .and_then(|pm| pm.get(PROVIDER_ID).cloned())
                .unwrap_or_default();
            if self.is_json_response_from_tool {
                payload.insert("isJsonResponseFromTool".to_owned(), Value::Bool(true));
            }
            payload.insert(
                "stopSequence".to_owned(),
                self.stop_sequence
                    .clone()
                    .map_or(Value::Null, Value::String),
            );
            let mut meta_map: ProviderMetadata = HashMap::new();
            meta_map.insert(PROVIDER_ID.to_owned(), payload.clone());
            meta_map.insert("bedrock".to_owned(), payload);
            self.provider_metadata = Some(meta_map);
        }
        out.push(finish_part(
            self.finish_reason,
            usage_value,
            self.provider_metadata,
        ));
        out
    }
}

fn finish_part(
    finish_reason: FinishReason,
    usage: Usage,
    provider_metadata: Option<ProviderMetadata>,
) -> StreamPart {
    StreamPart::Finish {
        usage,
        finish_reason,
        provider_metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_delta_lazily_emits_text_start() {
        let mut state = State::new(vec![], false, false, None);
        let chunk: StreamChunk = serde_json::from_value(serde_json::json!({
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": { "text": "hello" }
            }
        }))
        .unwrap();
        let parts = state.on_chunk(chunk);
        assert!(matches!(parts[0], StreamPart::TextStart { .. }));
        assert!(matches!(parts[1], StreamPart::TextDelta { .. }));
    }

    #[test]
    fn tool_use_assembly_round_trip() {
        let mut state = State::new(vec![], false, false, None);
        let _ = state.on_chunk(
            serde_json::from_value(serde_json::json!({
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": { "toolUse": { "toolUseId": "t1", "name": "weather" } }
                }
            }))
            .unwrap(),
        );
        let _ = state.on_chunk(
            serde_json::from_value(serde_json::json!({
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": { "toolUse": { "input": "{\"city\":" } }
                }
            }))
            .unwrap(),
        );
        let _ = state.on_chunk(
            serde_json::from_value(serde_json::json!({
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": { "toolUse": { "input": "\"NYC\"}" } }
                }
            }))
            .unwrap(),
        );
        let parts = state.on_chunk(
            serde_json::from_value(serde_json::json!({
                "contentBlockStop": { "contentBlockIndex": 0 }
            }))
            .unwrap(),
        );
        let tool_call = parts
            .iter()
            .find(|p| matches!(p, StreamPart::ToolCall(_)))
            .expect("tool call emitted");
        let StreamPart::ToolCall(tc) = tool_call else {
            unreachable!()
        };
        assert_eq!(tc.tool_call_id, "t1");
        assert_eq!(tc.input["city"], "NYC");
    }

    #[test]
    fn flush_emits_finish_with_usage() {
        let mut state = State::new(vec![], false, false, None);
        let _ = state.on_chunk(
            serde_json::from_value(serde_json::json!({
                "messageStop": { "stopReason": "end_turn" }
            }))
            .unwrap(),
        );
        let _ = state.on_chunk(
            serde_json::from_value(serde_json::json!({
                "metadata": {
                    "usage": { "inputTokens": 3, "outputTokens": 1, "totalTokens": 4 }
                }
            }))
            .unwrap(),
        );
        let parts = state.flush();
        assert!(matches!(parts[0], StreamPart::Finish { .. }));
        let StreamPart::Finish {
            finish_reason,
            usage,
            ..
        } = &parts[0]
        else {
            unreachable!()
        };
        assert_eq!(finish_reason.unified, FinishReasonKind::Stop);
        assert_eq!(usage.output_tokens.total, Some(1));
    }
}
