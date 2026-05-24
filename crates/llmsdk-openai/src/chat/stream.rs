//! Streaming state machine: SSE chunks → `Vec<StreamPart>`.
//!
//! Mirrors the `TransformStream` body in `openai-chat-language-model.ts`'s
//! `doStream`. We implement it as a stateful accumulator that consumes one
//! chunk at a time and returns the [`StreamPart`]s to emit immediately,
//! plus a finalizer that flushes the trailing `Finish` frame.
//!
//! # State
//!
//! - `active_text`: whether a `text-start` was emitted but not yet a
//!   `text-end`. `OpenAI` does not signal text-end on its own.
//! - `tool_calls`: per-index accumulator for streaming tool inputs. Each
//!   accumulator tracks the id / name / accumulated JSON arguments.
//! - `finish_reason`, `usage`: captured for the trailing `Finish` frame.
//! - `metadata_emitted`: ensure `ResponseMetadata` is sent at most once.
//!
//! # Mid-stream errors
//!
//! `ChatChunk::Error` (or a JSON parse error from [`crate`]'s SSE layer)
//! becomes a [`StreamPart::Error`]; we also force `finish_reason` to
//! [`FinishReasonKind::Error`] so the trailing `Finish` frame is honest.
// Rust guideline compliant 2026-02-21

use std::collections::BTreeMap;

use llmsdk_provider::language_model::{FinishReason, FinishReasonKind, StreamPart, ToolCallPart};
use llmsdk_provider::shared::Warning;

use super::finish_reason::map as map_finish_reason;
use super::stream_chunk::{ChatChunk, ChatDeltaChunk, ChunkDelta, ToolCallDelta};
use super::usage::convert as convert_usage;
use super::wire::WireUsage;

/// Logical id of the single text block this provider streams.
///
/// `OpenAI` streams "one big text block" per turn, so we hard-code id `"0"`
/// (matches ai-sdk).
const TEXT_BLOCK_ID: &str = "0";

/// Buffered state for one in-flight tool call.
#[derive(Debug, Default)]
struct ToolCallAccum {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
    /// True once we emitted `ToolInputStart` for this index.
    started: bool,
    /// True once we emitted `ToolInputEnd` / `ToolCall` for this index.
    closed: bool,
}

/// State machine driving an `OpenAI` Chat Completions stream.
#[derive(Debug)]
pub(crate) struct StreamState {
    initial_warnings: Option<Vec<Warning>>,
    active_text: bool,
    finish_reason: FinishReason,
    last_usage: Option<WireUsage>,
    metadata_emitted: bool,
    /// Indexed by `OpenAI`'s `tool_calls[].index`.
    tool_calls: BTreeMap<u32, ToolCallAccum>,
}

impl StreamState {
    /// Build with the warnings collected during request building. The
    /// `StreamStart` frame is emitted on the first call to [`Self::on_chunk`]
    /// — actually, we emit it eagerly from [`Self::start_frames`].
    pub(crate) fn new(warnings: Vec<Warning>) -> Self {
        Self {
            initial_warnings: Some(warnings),
            active_text: false,
            finish_reason: FinishReason::new(FinishReasonKind::Other),
            last_usage: None,
            metadata_emitted: false,
            tool_calls: BTreeMap::new(),
        }
    }

    /// Return the very first frame (`StreamStart` with warnings).
    pub(crate) fn start_frames(&mut self) -> Vec<StreamPart> {
        let warnings = self.initial_warnings.take().unwrap_or_default();
        vec![StreamPart::StreamStart { warnings }]
    }

    /// Handle one decoded SSE chunk; returns the frames to forward.
    pub(crate) fn on_chunk(&mut self, chunk: ChatChunk) -> Vec<StreamPart> {
        match chunk {
            ChatChunk::Error(err) => {
                self.finish_reason = FinishReason::new(FinishReasonKind::Error);
                vec![StreamPart::Error {
                    error: serde_json::json!({ "message": err.error.message }),
                }]
            }
            ChatChunk::Delta(d) => self.on_delta(d),
        }
    }

    /// Surface a JSON parse failure as an in-stream error.
    pub(crate) fn on_parse_error(&mut self, raw: &str, message: &str) -> Vec<StreamPart> {
        self.finish_reason = FinishReason::new(FinishReasonKind::Error);
        vec![StreamPart::Error {
            error: serde_json::json!({ "message": message, "raw": raw }),
        }]
    }

    /// Final flush: emit any pending `TextEnd` / `ToolInputEnd` / `ToolCall`
    /// frames, then `Finish`.
    pub(crate) fn flush(mut self) -> Vec<StreamPart> {
        let mut out = Vec::new();

        if self.active_text {
            out.push(StreamPart::TextEnd {
                id: TEXT_BLOCK_ID.to_owned(),
                provider_metadata: None,
            });
            self.active_text = false;
        }

        // Close any tool calls that are still open: emit end + final ToolCall.
        for (_idx, accum) in self.tool_calls {
            if accum.closed {
                continue;
            }
            if accum.started {
                out.push(StreamPart::ToolInputEnd {
                    id: accum_id(&accum),
                    provider_metadata: None,
                });
            }
            if let Some(call) = accum_to_tool_call(accum) {
                out.push(StreamPart::ToolCall(call));
            }
        }

        out.push(StreamPart::Finish {
            usage: convert_usage(self.last_usage.as_ref()),
            finish_reason: self.finish_reason,
            provider_metadata: None,
        });
        out
    }

    fn on_delta(&mut self, chunk: ChatDeltaChunk) -> Vec<StreamPart> {
        let mut out = Vec::new();

        // Emit one ResponseMetadata frame the first time we see useful fields.
        if !self.metadata_emitted
            && (chunk.id.is_some() || chunk.created.is_some() || chunk.model.is_some())
        {
            self.metadata_emitted = true;
            out.push(StreamPart::ResponseMetadata(
                llmsdk_provider::language_model::ResponseMetadata {
                    id: chunk.id.clone(),
                    timestamp: chunk.created.map(|c| c.to_string()),
                    model_id: chunk.model.clone(),
                    headers: None,
                },
            ));
        }

        if let Some(usage) = chunk.usage {
            self.last_usage = Some(usage);
        }

        let Some(choice) = chunk.choices.into_iter().next() else {
            return out;
        };

        if let Some(reason) = choice.finish_reason.as_deref() {
            // Only adopt a non-error reason; never overwrite a previously-set Error.
            if !matches!(self.finish_reason.unified, FinishReasonKind::Error) {
                self.finish_reason = map_finish_reason(Some(reason));
            }
        }

        if let Some(delta) = choice.delta {
            self.process_delta(delta, &mut out);
        }
        out
    }

    fn process_delta(&mut self, delta: ChunkDelta, out: &mut Vec<StreamPart>) {
        if let Some(text) = delta.content
            && !text.is_empty()
        {
            if !self.active_text {
                out.push(StreamPart::TextStart {
                    id: TEXT_BLOCK_ID.to_owned(),
                    provider_metadata: None,
                });
                self.active_text = true;
            }
            out.push(StreamPart::TextDelta {
                id: TEXT_BLOCK_ID.to_owned(),
                delta: text,
                provider_metadata: None,
            });
        }

        if let Some(calls) = delta.tool_calls {
            for call in calls {
                self.process_tool_call(call, out);
            }
        }
    }

    fn process_tool_call(&mut self, delta: ToolCallDelta, out: &mut Vec<StreamPart>) {
        let accum = self.tool_calls.entry(delta.index).or_default();

        if let Some(id) = delta.id
            && accum.id.is_none()
        {
            accum.id = Some(id);
        }
        if let Some(fun) = delta.function {
            if let Some(name) = fun.name
                && accum.name.is_none()
            {
                accum.name = Some(name);
            }
            if let Some(args) = fun.arguments {
                accum.arguments.push_str(&args);
                if !accum.started && accum.name.is_some() && accum.id.is_some() {
                    accum.started = true;
                    out.push(StreamPart::ToolInputStart {
                        id: accum.id.clone().unwrap_or_default(),
                        tool_name: accum.name.clone().unwrap_or_default(),
                        provider_executed: None,
                        dynamic: None,
                        title: None,
                        provider_metadata: None,
                    });
                }
                if accum.started && !args.is_empty() {
                    out.push(StreamPart::ToolInputDelta {
                        id: accum.id.clone().unwrap_or_default(),
                        delta: args,
                        provider_metadata: None,
                    });
                }
            }
        }
    }
}

fn accum_id(accum: &ToolCallAccum) -> String {
    accum.id.clone().unwrap_or_default()
}

fn accum_to_tool_call(accum: ToolCallAccum) -> Option<ToolCallPart> {
    let id = accum.id?;
    let name = accum.name?;
    let input = if accum.arguments.is_empty() {
        serde_json::Value::Object(serde_json::Map::new())
    } else {
        serde_json::from_str::<serde_json::Value>(&accum.arguments)
            .unwrap_or(serde_json::Value::String(accum.arguments))
    };
    Some(ToolCallPart {
        tool_call_id: id,
        tool_name: name,
        input,
        provider_executed: None,
        provider_options: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::stream_chunk::{
        ChatChoiceDelta, ChatErrorChunk, ChatErrorChunkInner, ToolCallFunctionDelta,
    };

    fn delta_chunk(content: Option<&str>, finish: Option<&str>) -> ChatChunk {
        ChatChunk::Delta(ChatDeltaChunk {
            id: None,
            created: None,
            model: None,
            choices: vec![ChatChoiceDelta {
                delta: Some(ChunkDelta {
                    content: content.map(str::to_owned),
                    tool_calls: None,
                }),
                finish_reason: finish.map(str::to_owned),
            }],
            usage: None,
        })
    }

    #[test]
    fn start_then_text_then_finish() {
        let mut state = StreamState::new(vec![]);
        let starts = state.start_frames();
        assert_eq!(starts.len(), 1);
        assert!(matches!(starts[0], StreamPart::StreamStart { .. }));

        let f1 = state.on_chunk(delta_chunk(Some("hel"), None));
        let f2 = state.on_chunk(delta_chunk(Some("lo"), None));
        let f3 = state.on_chunk(delta_chunk(None, Some("stop")));

        // f1: text-start + text-delta
        assert!(matches!(f1[0], StreamPart::TextStart { .. }));
        assert!(matches!(&f1[1], StreamPart::TextDelta { delta, .. } if delta == "hel"));
        // f2: only text-delta
        assert!(matches!(&f2[0], StreamPart::TextDelta { delta, .. } if delta == "lo"));
        // f3: finish_reason was captured but no immediate frame (text still active)
        assert!(f3.is_empty());

        let tail = state.flush();
        assert!(matches!(tail[0], StreamPart::TextEnd { .. }));
        if let StreamPart::Finish {
            finish_reason: fr, ..
        } = &tail[1]
        {
            assert_eq!(fr.unified, FinishReasonKind::Stop);
        } else {
            panic!("expected Finish");
        }
    }

    #[test]
    fn response_metadata_emitted_once() {
        let mut state = StreamState::new(vec![]);
        let _ = state.start_frames();

        let with_id = ChatChunk::Delta(ChatDeltaChunk {
            id: Some("chatcmpl-1".into()),
            created: Some(42),
            model: Some("gpt-4o-mini".into()),
            choices: vec![],
            usage: None,
        });
        let f1 = state.on_chunk(with_id.clone());
        let f2 = state.on_chunk(with_id);
        assert!(matches!(f1[0], StreamPart::ResponseMetadata(_)));
        assert!(f2.is_empty(), "metadata must not repeat");
    }

    #[test]
    fn tool_call_assembly() {
        let mut state = StreamState::new(vec![]);
        let _ = state.start_frames();

        // First chunk: id + name + partial args
        state.on_chunk(ChatChunk::Delta(ChatDeltaChunk {
            id: None,
            created: None,
            model: None,
            choices: vec![ChatChoiceDelta {
                delta: Some(ChunkDelta {
                    content: None,
                    tool_calls: Some(vec![ToolCallDelta {
                        index: 0,
                        id: Some("call_w".into()),
                        function: Some(ToolCallFunctionDelta {
                            name: Some("weather".into()),
                            arguments: Some(r#"{"ci"#.into()),
                        }),
                    }]),
                }),
                finish_reason: None,
            }],
            usage: None,
        }));

        state.on_chunk(ChatChunk::Delta(ChatDeltaChunk {
            id: None,
            created: None,
            model: None,
            choices: vec![ChatChoiceDelta {
                delta: Some(ChunkDelta {
                    content: None,
                    tool_calls: Some(vec![ToolCallDelta {
                        index: 0,
                        id: None,
                        function: Some(ToolCallFunctionDelta {
                            name: None,
                            arguments: Some(r#"ty":"NYC"}"#.into()),
                        }),
                    }]),
                }),
                finish_reason: Some("tool_calls".into()),
            }],
            usage: None,
        }));

        let tail = state.flush();
        // Expected tail order: ToolInputEnd, ToolCall, Finish
        assert!(matches!(tail[0], StreamPart::ToolInputEnd { .. }));
        if let StreamPart::ToolCall(tc) = &tail[1] {
            assert_eq!(tc.tool_call_id, "call_w");
            assert_eq!(tc.tool_name, "weather");
            assert_eq!(tc.input["city"], "NYC");
        } else {
            panic!("expected ToolCall, got {:?}", tail[1]);
        }
        if let StreamPart::Finish { finish_reason, .. } = &tail[2] {
            assert_eq!(finish_reason.unified, FinishReasonKind::ToolCalls);
        }
    }

    #[test]
    fn mid_stream_error_forces_error_finish() {
        let mut state = StreamState::new(vec![]);
        let _ = state.start_frames();

        let _ = state.on_chunk(delta_chunk(Some("partial"), None));
        let err = state.on_chunk(ChatChunk::Error(ChatErrorChunk {
            error: ChatErrorChunkInner {
                message: "boom".into(),
                _kind: None,
                _code: None,
            },
        }));
        assert!(matches!(err[0], StreamPart::Error { .. }));

        let tail = state.flush();
        // text-end + finish
        let last = tail.last().unwrap();
        if let StreamPart::Finish { finish_reason, .. } = last {
            assert_eq!(finish_reason.unified, FinishReasonKind::Error);
        } else {
            panic!("expected Finish at end");
        }
    }

    #[test]
    fn parse_error_inline() {
        let mut state = StreamState::new(vec![]);
        let _ = state.start_frames();
        let frames = state.on_parse_error("not-json", "expected value");
        assert!(matches!(frames[0], StreamPart::Error { .. }));
        let tail = state.flush();
        if let StreamPart::Finish { finish_reason, .. } = tail.last().unwrap() {
            assert_eq!(finish_reason.unified, FinishReasonKind::Error);
        }
    }
}
