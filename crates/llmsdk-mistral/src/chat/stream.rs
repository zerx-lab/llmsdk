//! Streaming state machine: Mistral SSE chunks → [`StreamPart`].
//!
//! Mirrors the `TransformStream` body in `mistral-chat-language-model.ts`'s
//! `doStream`. Mistral content arrives as either a string or a list of
//! `text` / `thinking` typed parts. We map `text` deltas to a single text
//! block with id `"0"` and `thinking` deltas to a reasoning block whose id
//! is generated on the fly. When text arrives during an active reasoning
//! block we close the reasoning block first (matches upstream behaviour).
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{
    FinishReason, FinishReasonKind, ResponseMetadata, StreamPart, ToolCallPart,
};
use llmsdk_provider::shared::Warning;
use llmsdk_provider_utils::time::rfc3339_from_unix_seconds;

use super::finish_reason::map as map_finish_reason;
use super::parse_response::collect_thinking_text;
use super::usage;
use super::wire::{ChatChunk, MistralContent, MistralContentPart, WireUsage};

const TEXT_ID: &str = "0";

/// State machine driving a Mistral Chat Completions stream.
#[derive(Debug)]
pub(crate) struct StreamState {
    initial_warnings: Option<Vec<Warning>>,
    finish_reason: FinishReason,
    last_usage: Option<WireUsage>,
    metadata_emitted: bool,
    is_first_chunk: bool,
    active_text: bool,
    active_reasoning_id: Option<String>,
    reasoning_id_seq: u64,
}

impl StreamState {
    /// Build with the warnings collected during request building.
    pub(crate) fn new(warnings: Vec<Warning>) -> Self {
        Self {
            initial_warnings: Some(warnings),
            finish_reason: FinishReason::new(FinishReasonKind::Other),
            last_usage: None,
            metadata_emitted: false,
            is_first_chunk: true,
            active_text: false,
            active_reasoning_id: None,
            reasoning_id_seq: 0,
        }
    }

    fn next_reasoning_id(&mut self) -> String {
        self.reasoning_id_seq = self.reasoning_id_seq.wrapping_add(1);
        format!("reasoning-{}", self.reasoning_id_seq)
    }

    /// Return the very first frame (`StreamStart` with warnings).
    pub(crate) fn start_frames(&mut self) -> Vec<StreamPart> {
        let warnings = self.initial_warnings.take().unwrap_or_default();
        vec![StreamPart::StreamStart { warnings }]
    }

    /// Handle one decoded SSE chunk; returns the frames to forward.
    #[allow(
        clippy::too_many_lines,
        reason = "single dispatcher mirroring upstream's mistral-chat-language-model.ts TransformStream body"
    )]
    pub(crate) fn on_chunk(&mut self, chunk: ChatChunk) -> Vec<StreamPart> {
        let mut out = Vec::new();

        if self.is_first_chunk {
            self.is_first_chunk = false;
            if !self.metadata_emitted
                && (chunk.id.is_some() || chunk.created.is_some() || chunk.model.is_some())
            {
                self.metadata_emitted = true;
                out.push(StreamPart::ResponseMetadata(ResponseMetadata {
                    id: chunk.id.clone(),
                    timestamp: chunk.created.map(rfc3339_from_unix_seconds),
                    model_id: chunk.model.clone(),
                    headers: None,
                }));
            }
        }

        if let Some(u) = chunk.usage {
            self.last_usage = Some(u);
        }

        let Some(choice) = chunk.choices.into_iter().next() else {
            return out;
        };

        if let Some(reason) = choice.finish_reason.as_deref()
            && !matches!(self.finish_reason.unified, FinishReasonKind::Error)
        {
            self.finish_reason = map_finish_reason(Some(reason));
        }

        let Some(delta) = choice.delta else {
            return out;
        };

        // Reasoning ("thinking") parts come first in the upstream walk because
        // any thinking text might precede the text content within the same
        // chunk; we emit them before opening any text block.
        if let Some(MistralContent::Parts(parts)) = &delta.content {
            for part in parts {
                if let MistralContentPart::Thinking { thinking } = part {
                    let reasoning_delta = collect_thinking_text(thinking);
                    if reasoning_delta.is_empty() {
                        continue;
                    }
                    if self.active_reasoning_id.is_none() {
                        // close any active text block first
                        if self.active_text {
                            out.push(StreamPart::TextEnd {
                                id: TEXT_ID.to_owned(),
                                provider_metadata: None,
                            });
                            self.active_text = false;
                        }
                        let id = self.next_reasoning_id();
                        self.active_reasoning_id = Some(id.clone());
                        out.push(StreamPart::ReasoningStart {
                            id,
                            provider_metadata: None,
                        });
                    }
                    out.push(StreamPart::ReasoningDelta {
                        id: self
                            .active_reasoning_id
                            .clone()
                            .unwrap_or_else(|| "reasoning".to_owned()),
                        delta: reasoning_delta,
                        provider_metadata: None,
                    });
                }
            }
        }

        // Now any text content (either bare string or the text parts inside
        // the array).
        let text_delta = extract_text(delta.content.as_ref());
        if let Some(text) = text_delta
            && !text.is_empty()
        {
            if !self.active_text {
                // close any active reasoning block before starting text
                if let Some(id) = self.active_reasoning_id.take() {
                    out.push(StreamPart::ReasoningEnd {
                        id,
                        provider_metadata: None,
                    });
                }
                out.push(StreamPart::TextStart {
                    id: TEXT_ID.to_owned(),
                    provider_metadata: None,
                });
                self.active_text = true;
            }
            out.push(StreamPart::TextDelta {
                id: TEXT_ID.to_owned(),
                delta: text,
                provider_metadata: None,
            });
        }

        // Tool calls (Mistral emits each tool call complete in one chunk).
        if let Some(tool_calls) = delta.tool_calls {
            for tc in tool_calls {
                let id = tc.id.clone();
                out.push(StreamPart::ToolInputStart {
                    id: id.clone(),
                    tool_name: tc.function.name.clone(),
                    provider_executed: None,
                    dynamic: None,
                    title: None,
                    provider_metadata: None,
                });
                out.push(StreamPart::ToolInputDelta {
                    id: id.clone(),
                    delta: tc.function.arguments.clone(),
                    provider_metadata: None,
                });
                out.push(StreamPart::ToolInputEnd {
                    id: id.clone(),
                    provider_metadata: None,
                });

                let input = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                    .unwrap_or(serde_json::Value::String(tc.function.arguments));
                out.push(StreamPart::ToolCall(ToolCallPart {
                    tool_call_id: id,
                    tool_name: tc.function.name,
                    input,
                    provider_executed: None,
                    dynamic: None,
                    provider_options: None,
                }));
            }
        }

        out
    }

    /// Surface a JSON parse failure as an in-stream error.
    pub(crate) fn on_parse_error(&mut self, raw: &str, message: &str) -> Vec<StreamPart> {
        self.finish_reason = FinishReason::new(FinishReasonKind::Error);
        vec![StreamPart::Error {
            error: serde_json::json!({ "message": message, "raw": raw }),
        }]
    }

    /// Final flush: emit any pending `*End` frames, then `Finish`.
    pub(crate) fn flush(self) -> Vec<StreamPart> {
        let mut out = Vec::new();

        if let Some(id) = self.active_reasoning_id {
            out.push(StreamPart::ReasoningEnd {
                id,
                provider_metadata: None,
            });
        }
        if self.active_text {
            out.push(StreamPart::TextEnd {
                id: TEXT_ID.to_owned(),
                provider_metadata: None,
            });
        }

        let usage_value = self
            .last_usage
            .as_ref()
            .map_or_else(usage::zero, usage::convert);

        out.push(StreamPart::Finish {
            usage: usage_value,
            finish_reason: self.finish_reason,
            provider_metadata: None,
        });
        out
    }
}

/// Mirrors `extractTextContent` in upstream — collapses the string / parts
/// union into a single string, skipping non-text parts.
fn extract_text(content: Option<&MistralContent>) -> Option<String> {
    match content? {
        MistralContent::Text(s) => Some(s.clone()),
        MistralContent::Parts(parts) => {
            let mut s = String::new();
            for p in parts {
                if let MistralContentPart::Text { text } = p {
                    s.push_str(text);
                }
            }
            if s.is_empty() { None } else { Some(s) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::wire::{
        ChatChunkChoice, ChatChunkDelta, MistralThinkingChunk, WireFunctionCall, WireToolCall,
        WireToolCallKind,
    };

    fn text_chunk(text: &str, finish: Option<&str>) -> ChatChunk {
        ChatChunk {
            choices: vec![ChatChunkChoice {
                delta: Some(ChatChunkDelta {
                    content: Some(MistralContent::Text(text.into())),
                    ..Default::default()
                }),
                finish_reason: finish.map(str::to_owned),
                index: 0,
            }],
            ..Default::default()
        }
    }

    fn thinking_chunk(text: &str) -> ChatChunk {
        ChatChunk {
            choices: vec![ChatChunkChoice {
                delta: Some(ChatChunkDelta {
                    content: Some(MistralContent::Parts(vec![MistralContentPart::Thinking {
                        thinking: vec![MistralThinkingChunk::Text { text: text.into() }],
                    }])),
                    ..Default::default()
                }),
                finish_reason: None,
                index: 0,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn start_then_text_then_finish() {
        let mut state = StreamState::new(vec![]);
        let s = state.start_frames();
        assert!(matches!(s[0], StreamPart::StreamStart { .. }));

        let f1 = state.on_chunk(text_chunk("hel", None));
        let f2 = state.on_chunk(text_chunk("lo", None));
        state.on_chunk(text_chunk("", Some("stop")));

        assert!(matches!(&f1[0], StreamPart::TextStart { .. }));
        assert!(matches!(&f1[1], StreamPart::TextDelta { delta, .. } if delta == "hel"));
        assert!(matches!(&f2[0], StreamPart::TextDelta { delta, .. } if delta == "lo"));

        let tail = state.flush();
        assert!(matches!(tail[0], StreamPart::TextEnd { .. }));
        let StreamPart::Finish { finish_reason, .. } = &tail[1] else {
            panic!("expected Finish");
        };
        assert_eq!(finish_reason.unified, FinishReasonKind::Stop);
    }

    #[test]
    fn thinking_then_text_closes_reasoning_first() {
        let mut state = StreamState::new(vec![]);
        let _ = state.start_frames();
        let f1 = state.on_chunk(thinking_chunk("think"));
        assert!(matches!(&f1[0], StreamPart::ReasoningStart { .. }));
        assert!(matches!(&f1[1], StreamPart::ReasoningDelta { .. }));
        let f2 = state.on_chunk(text_chunk("answer", None));
        assert!(matches!(&f2[0], StreamPart::ReasoningEnd { .. }));
        assert!(matches!(&f2[1], StreamPart::TextStart { .. }));
        assert!(matches!(&f2[2], StreamPart::TextDelta { .. }));
    }

    #[test]
    fn tool_call_one_chunk() {
        let mut state = StreamState::new(vec![]);
        let _ = state.start_frames();
        let frames = state.on_chunk(ChatChunk {
            choices: vec![ChatChunkChoice {
                delta: Some(ChatChunkDelta {
                    tool_calls: Some(vec![WireToolCall {
                        id: "call_w".into(),
                        kind: Some(WireToolCallKind::Function),
                        function: WireFunctionCall {
                            name: "weather".into(),
                            arguments: r#"{"city":"NYC"}"#.into(),
                        },
                    }]),
                    ..Default::default()
                }),
                finish_reason: Some("tool_calls".into()),
                index: 0,
            }],
            ..Default::default()
        });
        assert!(matches!(&frames[0], StreamPart::ToolInputStart { .. }));
        assert!(matches!(&frames[1], StreamPart::ToolInputDelta { .. }));
        assert!(matches!(&frames[2], StreamPart::ToolInputEnd { .. }));
        let StreamPart::ToolCall(tc) = &frames[3] else {
            panic!("expected ToolCall");
        };
        assert_eq!(tc.tool_call_id, "call_w");
        assert_eq!(tc.input["city"], "NYC");
    }

    #[test]
    fn parse_error_marks_finish_as_error() {
        let mut state = StreamState::new(vec![]);
        let _ = state.start_frames();
        let frames = state.on_parse_error("not-json", "expected value");
        assert!(matches!(frames[0], StreamPart::Error { .. }));
        let tail = state.flush();
        let StreamPart::Finish { finish_reason, .. } = tail.last().unwrap() else {
            panic!("expected Finish");
        };
        assert_eq!(finish_reason.unified, FinishReasonKind::Error);
    }
}
