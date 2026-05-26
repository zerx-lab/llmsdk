//! Gemini `streamGenerateContent?alt=sse` parser → `StreamPart` stream.
//!
//! Mirrors the `doStream` branch of
//! `@ai-sdk/google/src/google-language-model.ts`. Each Server-Sent Event is
//! one [`WireChunk`]; the parser keeps a small state machine to:
//!
//! - Group consecutive `text` parts under a single `text-start` / `text-end`
//!   block id; same for `reasoning`.
//! - Open / close `tool-input-start` / `tool-input-end` for streaming
//!   function calls driven by `partialArgs`.
//! - Deduplicate URL sources across chunks.
//! - Forward server-tool calls (`toolCall`) and responses (`toolResponse`)
//!   as `tool-call` / `tool-result` parts with `provider_executed: true`.
//! - Emit a final `finish` part with finish reason + usage + provider
//!   metadata.
// Rust guideline compliant 2026-05-25

use std::collections::HashSet;

use async_stream::stream;
use bytes::Bytes;
use futures::Stream;
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    BoxStream, FilePart, FinishReason, FinishReasonKind, Source, StreamPart, ToolCallPart,
    ToolResult, ToolResultOutput,
};
use llmsdk_provider::shared::{FileBytes, FileData, ProviderMetadata, ProviderOptions, Warning};
use llmsdk_provider_utils::sse::{SseEvent, sse_json_stream};
use serde_json::{Map, Value};

use super::accumulator::GoogleJsonAccumulator;
use super::finish_reason::map_finish_reason;
use super::usage::convert_usage;
use super::wire::{WireChunk, WireGroundingMetadata, WireUsage, usage_to_json_map};

/// Build a [`StreamPart`] stream from the raw byte stream of a Gemini
/// streaming response.
pub(crate) fn make_stream(
    bytes: impl Stream<Item = Result<Bytes, ProviderError>> + Send + 'static,
    warnings: Vec<Warning>,
    include_raw_chunks: bool,
    provider_keys: Vec<String>,
    mut next_id: impl FnMut() -> String + Send + 'static,
) -> BoxStream<Result<StreamPart, ProviderError>> {
    let s = stream! {
        yield Ok(StreamPart::StreamStart { warnings: warnings.clone() });

        let mut events = sse_json_stream::<WireChunk>(bytes);
        let mut state = StreamState::default();
        let provider_keys_ref: Vec<&str> = provider_keys.iter().map(String::as_str).collect();

        use futures::StreamExt;
        while let Some(item) = events.next().await {
            match item {
                Ok(SseEvent::Data(chunk)) => {
                    if include_raw_chunks {
                        yield Ok(StreamPart::Raw {
                            raw_value: serde_json::to_value(&chunk).unwrap_or(Value::Null),
                        });
                    }

                    if let Some(u) = &chunk.usage_metadata {
                        state.usage = Some(u.clone());
                    }

                    let candidates = chunk.candidates.as_ref();
                    let Some(candidates) = candidates else { continue };
                    let Some(candidate) = candidates.first() else { continue };

                    if let Some(g) = &candidate.grounding_metadata {
                        state.last_grounding = Some(g.clone());
                    }
                    if let Some(u) = &candidate.url_context_metadata {
                        state.last_url_context = Some(u.clone());
                    }

                    // Emit sources from grounding metadata (deduped).
                    if let Some(g) = &candidate.grounding_metadata {
                        for src in extract_url_sources(g, &mut next_id) {
                            if let Source::Url { ref url, .. } = src {
                                if state.emitted_sources.insert(url.clone()) {
                                    yield Ok(StreamPart::Source(src));
                                }
                            } else {
                                yield Ok(StreamPart::Source(src));
                            }
                        }
                    }

                    let Some(content) = &candidate.content else {
                        if let Some(raw) = &candidate.finish_reason {
                            state.finish_reason = FinishReason::with_raw(
                                map_finish_reason(Some(raw.as_str()), state.has_tool_calls),
                                raw.clone(),
                            );
                            state.provider_metadata = Some(build_finish_metadata(
                                &chunk,
                                &state,
                                &provider_keys_ref,
                            ));
                        }
                        continue;
                    };
                    let parts = content.parts.as_deref().unwrap_or(&[]);

                    // Pass 1: non-function-call parts in order.
                    for part in parts {
                        if let Some(exe) = &part.executable_code {
                            if !exe.code.is_empty() {
                                let id = next_id();
                                state.last_code_id = Some(id.clone());
                                let input = serde_json::to_string(exe)
                                    .unwrap_or_else(|_| "{}".into());
                                yield Ok(StreamPart::ToolCall(ToolCallPart {
                                    tool_call_id: id,
                                    tool_name: "code_execution".into(),
                                    input: serde_json::from_str(&input).unwrap_or(Value::Null),
                                    provider_executed: Some(true),
                                    dynamic: None,
                                    provider_options: None,
                                }));
                                continue;
                            }
                        }
                        if let Some(res) = &part.code_execution_result {
                            if let Some(id) = state.last_code_id.take() {
                                let mut out = Map::new();
                                out.insert("outcome".into(), Value::String(res.outcome.clone()));
                                out.insert(
                                    "output".into(),
                                    Value::String(res.output.clone().unwrap_or_default()),
                                );
                                yield Ok(StreamPart::ToolResult(ToolResult {
                                    tool_call_id: id,
                                    tool_name: "code_execution".into(),
                                    output: ToolResultOutput::Json {
                                        value: Value::Object(out),
                                        provider_options: None,
                                    },
                                    preliminary: None,
                                    provider_metadata: None,
                                }));
                            }
                            continue;
                        }
                        if let Some(text) = &part.text {
                            let sig_meta = thought_sig_meta(
                                part.thought_signature.as_deref(),
                                &provider_keys_ref,
                            );
                            if text.is_empty() {
                                if let (Some(meta), Some(id)) =
                                    (sig_meta.clone(), state.text_block_id.clone())
                                {
                                    yield Ok(StreamPart::TextDelta {
                                        id,
                                        delta: String::new(),
                                        provider_metadata: Some(meta),
                                    });
                                }
                            } else if part.thought == Some(true) {
                                if let Some(id) = state.text_block_id.take() {
                                    yield Ok(StreamPart::TextEnd {
                                        id,
                                        provider_metadata: None,
                                    });
                                }
                                if state.reasoning_block_id.is_none() {
                                    let id = state.next_block_id();
                                    state.reasoning_block_id = Some(id.clone());
                                    yield Ok(StreamPart::ReasoningStart {
                                        id,
                                        provider_metadata: sig_meta.clone(),
                                    });
                                }
                                let id = state.reasoning_block_id.clone().unwrap();
                                yield Ok(StreamPart::ReasoningDelta {
                                    id,
                                    delta: text.clone(),
                                    provider_metadata: sig_meta.clone(),
                                });
                            } else {
                                if let Some(id) = state.reasoning_block_id.take() {
                                    yield Ok(StreamPart::ReasoningEnd {
                                        id,
                                        provider_metadata: None,
                                    });
                                }
                                if state.text_block_id.is_none() {
                                    let id = state.next_block_id();
                                    state.text_block_id = Some(id.clone());
                                    yield Ok(StreamPart::TextStart {
                                        id,
                                        provider_metadata: sig_meta.clone(),
                                    });
                                }
                                let id = state.text_block_id.clone().unwrap();
                                yield Ok(StreamPart::TextDelta {
                                    id,
                                    delta: text.clone(),
                                    provider_metadata: sig_meta.clone(),
                                });
                            }
                            continue;
                        }
                        if let Some(inline) = &part.inline_data {
                            if let Some(id) = state.text_block_id.take() {
                                yield Ok(StreamPart::TextEnd { id, provider_metadata: None });
                            }
                            if let Some(id) = state.reasoning_block_id.take() {
                                yield Ok(StreamPart::ReasoningEnd { id, provider_metadata: None });
                            }
                            let media = inline.mime_type.clone();
                            let data = FileData::Data {
                                data: FileBytes::Base64(inline.data.clone()),
                            };
                            let pm = thought_sig_meta(
                                part.thought_signature.as_deref(),
                                &provider_keys_ref,
                            );
                            if part.thought == Some(true) {
                                yield Ok(StreamPart::ReasoningFile {
                                    data,
                                    media_type: media,
                                    provider_metadata: pm,
                                });
                            } else {
                                let pm_options = pm.map(metadata_to_options);
                                yield Ok(StreamPart::File(FilePart {
                                    filename: None,
                                    data,
                                    media_type: media,
                                    provider_options: pm_options,
                                }));
                            }
                            continue;
                        }
                        if let Some(tc) = &part.tool_call {
                            let id = if tc.id.is_empty() { next_id() } else { tc.id.clone() };
                            state.last_server_id = Some(id.clone());
                            let mut meta = Map::new();
                            meta.insert("serverToolCallId".into(), Value::String(id.clone()));
                            meta.insert("serverToolType".into(), Value::String(tc.tool_type.clone()));
                            if let Some(sig) = &part.thought_signature {
                                meta.insert("thoughtSignature".into(), Value::String(sig.clone()));
                            }
                            let server_meta = wrap_meta(&provider_keys_ref, meta);
                            yield Ok(StreamPart::ToolCall(ToolCallPart {
                                tool_call_id: id,
                                tool_name: format!("server:{}", tc.tool_type),
                                input: tc.args.clone().unwrap_or_else(|| Value::Object(Map::new())),
                                provider_executed: Some(true),
                                dynamic: Some(true),
                                provider_options: Some(metadata_to_options(server_meta)),
                            }));
                            continue;
                        }
                        if let Some(tr) = &part.tool_response {
                            let id = state.last_server_id.take().unwrap_or_else(|| {
                                if tr.id.is_empty() { next_id() } else { tr.id.clone() }
                            });
                            let mut meta = Map::new();
                            meta.insert("serverToolCallId".into(), Value::String(id.clone()));
                            meta.insert("serverToolType".into(), Value::String(tr.tool_type.clone()));
                            if let Some(sig) = &part.thought_signature {
                                meta.insert("thoughtSignature".into(), Value::String(sig.clone()));
                            }
                            let server_meta = wrap_meta(&provider_keys_ref, meta);
                            yield Ok(StreamPart::ToolResult(ToolResult {
                                tool_call_id: id,
                                tool_name: format!("server:{}", tr.tool_type),
                                output: ToolResultOutput::Json {
                                    value: tr
                                        .response
                                        .clone()
                                        .unwrap_or_else(|| Value::Object(Map::new())),
                                    provider_options: None,
                                },
                                preliminary: None,
                                provider_metadata: Some(server_meta),
                            }));
                        }
                    }

                    // Pass 2: function-call parts (potentially partial).
                    for part in parts {
                        let Some(fc) = &part.function_call else { continue };
                        let provider_meta = thought_sig_meta(
                            part.thought_signature.as_deref(),
                            &provider_keys_ref,
                        );

                        let is_streaming_chunk = fc.partial_args.is_some()
                            || (fc.name.is_some() && fc.will_continue == Some(true));
                        let is_terminal_chunk = fc.name.is_none()
                            && fc.args.is_none()
                            && fc.partial_args.is_none()
                            && fc.will_continue.is_none();
                        let is_complete_call = fc.name.is_some()
                            && fc.args.is_some()
                            && fc.partial_args.is_none();
                        let is_no_args_complete = fc.name.is_some()
                            && fc.args.is_none()
                            && fc.partial_args.is_none()
                            && fc.will_continue != Some(true);

                        if is_streaming_chunk {
                            if let Some(name) = &fc.name {
                                let id = fc.id.clone().unwrap_or_else(&mut next_id);
                                let mut accumulator = GoogleJsonAccumulator::default();
                                state.active_streaming_tools.push(ActiveTool {
                                    tool_call_id: id.clone(),
                                    tool_name: name.clone(),
                                    accumulator: Some(accumulator.clone_for_state()),
                                    provider_metadata: provider_meta.clone(),
                                });
                                yield Ok(StreamPart::ToolInputStart {
                                    id: id.clone(),
                                    tool_name: name.clone(),
                                    provider_executed: None,
                                    dynamic: None,
                                    title: None,
                                    provider_metadata: provider_meta.clone(),
                                });
                                if let Some(pa) = &fc.partial_args {
                                    let delta = accumulator.process_partial_args(pa);
                                    if let Some(active) = state.active_streaming_tools.last_mut() {
                                        active.accumulator = Some(accumulator);
                                    }
                                    if !delta.is_empty() {
                                        yield Ok(StreamPart::ToolInputDelta {
                                            id: id.clone(),
                                            delta,
                                            provider_metadata: provider_meta.clone(),
                                        });
                                    }
                                    if fc.will_continue != Some(true)
                                        && pa.iter().all(|a| a.will_continue != Some(true))
                                    {
                                        for ev in finish_active_streaming_tool(&mut state) {
                                            yield Ok(ev);
                                        }
                                    }
                                }
                            } else if let Some(pa) = &fc.partial_args {
                                let last_idx = state.active_streaming_tools.len();
                                if last_idx > 0 {
                                    let active = &mut state.active_streaming_tools[last_idx - 1];
                                    let mut accumulator = active.accumulator.clone().unwrap_or_default();
                                    let delta = accumulator.process_partial_args(pa);
                                    active.accumulator = Some(accumulator);
                                    let id = active.tool_call_id.clone();
                                    let pm = provider_meta.clone();
                                    if !delta.is_empty() {
                                        yield Ok(StreamPart::ToolInputDelta {
                                            id,
                                            delta,
                                            provider_metadata: pm,
                                        });
                                    }
                                    if fc.will_continue != Some(true)
                                        && pa.iter().all(|a| a.will_continue != Some(true))
                                    {
                                        for ev in finish_active_streaming_tool(&mut state) {
                                            yield Ok(ev);
                                        }
                                    }
                                }
                            }
                        } else if is_terminal_chunk && !state.active_streaming_tools.is_empty() {
                            for ev in finish_active_streaming_tool(&mut state) {
                                yield Ok(ev);
                            }
                        } else if is_complete_call {
                            let id = fc.id.clone().unwrap_or_else(&mut next_id);
                            let name = fc.name.clone().unwrap_or_default();
                            let args_str = match &fc.args {
                                Some(Value::String(s)) => s.clone(),
                                Some(v) => serde_json::to_string(v).unwrap_or_else(|_| "{}".into()),
                                None => "{}".into(),
                            };
                            let pm = provider_meta.clone();
                            yield Ok(StreamPart::ToolInputStart {
                                id: id.clone(),
                                tool_name: name.clone(),
                                provider_executed: None,
                                dynamic: None,
                                title: None,
                                provider_metadata: pm.clone(),
                            });
                            yield Ok(StreamPart::ToolInputDelta {
                                id: id.clone(),
                                delta: args_str.clone(),
                                provider_metadata: pm.clone(),
                            });
                            yield Ok(StreamPart::ToolInputEnd {
                                id: id.clone(),
                                provider_metadata: pm.clone(),
                            });
                            yield Ok(StreamPart::ToolCall(ToolCallPart {
                                tool_call_id: id,
                                tool_name: name,
                                input: serde_json::from_str(&args_str).unwrap_or(Value::Null),
                                provider_executed: None,
                                dynamic: None,
                                provider_options: pm.map(metadata_to_options),
                            }));
                            state.has_tool_calls = true;
                        } else if is_no_args_complete {
                            let id = fc.id.clone().unwrap_or_else(&mut next_id);
                            let name = fc.name.clone().unwrap_or_default();
                            let pm = provider_meta.clone();
                            yield Ok(StreamPart::ToolInputStart {
                                id: id.clone(),
                                tool_name: name.clone(),
                                provider_executed: None,
                                dynamic: None,
                                title: None,
                                provider_metadata: pm.clone(),
                            });
                            yield Ok(StreamPart::ToolInputEnd {
                                id: id.clone(),
                                provider_metadata: pm.clone(),
                            });
                            yield Ok(StreamPart::ToolCall(ToolCallPart {
                                tool_call_id: id,
                                tool_name: name,
                                input: Value::Object(Map::new()),
                                provider_executed: None,
                                dynamic: None,
                                provider_options: pm.map(metadata_to_options),
                            }));
                            state.has_tool_calls = true;
                        }
                    }

                    if let Some(raw) = &candidate.finish_reason {
                        state.finish_reason = FinishReason::with_raw(
                            map_finish_reason(Some(raw.as_str()), state.has_tool_calls),
                            raw.clone(),
                        );
                        state.provider_metadata = Some(build_finish_metadata(
                            &chunk,
                            &state,
                            &provider_keys_ref,
                        ));
                    }
                }
                Ok(SseEvent::ParseError { raw, message }) => {
                    let mut err = Map::new();
                    err.insert("message".into(), Value::String(message));
                    err.insert("raw".into(), Value::String(raw));
                    yield Ok(StreamPart::Error { error: Value::Object(err) });
                }
                Err(e) => {
                    yield Err(e);
                    return;
                }
            }
        }

        if let Some(id) = state.text_block_id.take() {
            yield Ok(StreamPart::TextEnd { id, provider_metadata: None });
        }
        if let Some(id) = state.reasoning_block_id.take() {
            yield Ok(StreamPart::ReasoningEnd { id, provider_metadata: None });
        }

        yield Ok(StreamPart::Finish {
            usage: convert_usage(state.usage.as_ref()),
            finish_reason: state.finish_reason.clone(),
            provider_metadata: state.provider_metadata.clone(),
        });
    };

    Box::pin(s)
}

#[derive(Debug)]
struct StreamState {
    text_block_id: Option<String>,
    reasoning_block_id: Option<String>,
    block_counter: u32,
    emitted_sources: HashSet<String>,
    last_code_id: Option<String>,
    last_server_id: Option<String>,
    usage: Option<WireUsage>,
    last_grounding: Option<WireGroundingMetadata>,
    last_url_context: Option<Value>,
    finish_reason: FinishReason,
    provider_metadata: Option<ProviderMetadata>,
    active_streaming_tools: Vec<ActiveTool>,
    has_tool_calls: bool,
}

impl Default for StreamState {
    fn default() -> Self {
        Self {
            text_block_id: None,
            reasoning_block_id: None,
            block_counter: 0,
            emitted_sources: HashSet::new(),
            last_code_id: None,
            last_server_id: None,
            usage: None,
            last_grounding: None,
            last_url_context: None,
            finish_reason: FinishReason::new(FinishReasonKind::Other),
            provider_metadata: None,
            active_streaming_tools: Vec::new(),
            has_tool_calls: false,
        }
    }
}

impl StreamState {
    fn next_block_id(&mut self) -> String {
        let id = self.block_counter.to_string();
        self.block_counter += 1;
        id
    }
}

#[derive(Debug)]
struct ActiveTool {
    tool_call_id: String,
    tool_name: String,
    accumulator: Option<GoogleJsonAccumulator>,
    provider_metadata: Option<ProviderMetadata>,
}

fn finish_active_streaming_tool(state: &mut StreamState) -> Vec<StreamPart> {
    let Some(active) = state.active_streaming_tools.pop() else {
        return Vec::new();
    };
    let accumulator = active.accumulator.unwrap_or_default();
    let (final_json, closing_delta) = accumulator.finalize();
    let mut out = Vec::new();
    if !closing_delta.is_empty() {
        out.push(StreamPart::ToolInputDelta {
            id: active.tool_call_id.clone(),
            delta: closing_delta,
            provider_metadata: active.provider_metadata.clone(),
        });
    }
    out.push(StreamPart::ToolInputEnd {
        id: active.tool_call_id.clone(),
        provider_metadata: active.provider_metadata.clone(),
    });
    out.push(StreamPart::ToolCall(ToolCallPart {
        tool_call_id: active.tool_call_id,
        tool_name: active.tool_name,
        input: serde_json::from_str(&final_json).unwrap_or(Value::Null),
        provider_executed: None,
        dynamic: None,
        provider_options: active.provider_metadata.map(metadata_to_options),
    }));
    state.has_tool_calls = true;
    out
}

fn build_finish_metadata(
    chunk: &WireChunk,
    state: &StreamState,
    provider_keys: &[&str],
) -> ProviderMetadata {
    let mut payload = Map::new();
    payload.insert(
        "promptFeedback".into(),
        chunk
            .prompt_feedback
            .as_ref()
            .map(|p| serde_json::to_value(p).unwrap_or(Value::Null))
            .unwrap_or(Value::Null),
    );
    payload.insert(
        "groundingMetadata".into(),
        state
            .last_grounding
            .as_ref()
            .map(|g| serde_json::to_value(g).unwrap_or(Value::Null))
            .unwrap_or(Value::Null),
    );
    payload.insert(
        "urlContextMetadata".into(),
        state.last_url_context.clone().unwrap_or(Value::Null),
    );
    payload.insert(
        "safetyRatings".into(),
        chunk
            .candidates
            .as_ref()
            .and_then(|c| c.first())
            .and_then(|c| c.safety_ratings.clone())
            .map(Value::Array)
            .unwrap_or(Value::Null),
    );
    payload.insert(
        "usageMetadata".into(),
        state
            .usage
            .as_ref()
            .map(|u| Value::Object(usage_to_json_map(u)))
            .unwrap_or(Value::Null),
    );
    payload.insert(
        "finishMessage".into(),
        chunk
            .candidates
            .as_ref()
            .and_then(|c| c.first())
            .and_then(|c| c.finish_message.clone())
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    payload.insert(
        "serviceTier".into(),
        state
            .usage
            .as_ref()
            .and_then(|u| u.service_tier.clone())
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    wrap_meta(provider_keys, payload)
}

fn wrap_meta(keys: &[&str], payload: Map<String, Value>) -> ProviderMetadata {
    let mut out = ProviderMetadata::new();
    for k in keys {
        out.insert((*k).to_owned(), payload.clone());
    }
    out
}

fn thought_sig_meta(sig: Option<&str>, keys: &[&str]) -> Option<ProviderMetadata> {
    let sig = sig?;
    let mut p = Map::new();
    p.insert("thoughtSignature".into(), Value::String(sig.to_owned()));
    Some(wrap_meta(keys, p))
}

fn metadata_to_options(m: ProviderMetadata) -> ProviderOptions {
    let mut out = ProviderOptions::new();
    for (k, v) in m {
        out.insert(k, v);
    }
    out
}

fn extract_url_sources(
    grounding: &WireGroundingMetadata,
    next_id: &mut impl FnMut() -> String,
) -> Vec<Source> {
    let mut out = Vec::new();
    let Some(chunks) = grounding.grounding_chunks.as_ref() else {
        return out;
    };
    for c in chunks {
        if let Some(w) = &c.web {
            out.push(Source::Url {
                id: next_id(),
                url: w.uri.clone(),
                title: w.title.clone(),
                provider_metadata: None,
            });
        } else if let Some(img) = &c.image {
            out.push(Source::Url {
                id: next_id(),
                url: img.source_uri.clone(),
                title: img.title.clone(),
                provider_metadata: None,
            });
        } else if let Some(m) = &c.maps {
            if let Some(u) = &m.uri {
                out.push(Source::Url {
                    id: next_id(),
                    url: u.clone(),
                    title: m.title.clone(),
                    provider_metadata: None,
                });
            }
        }
    }
    out
}
