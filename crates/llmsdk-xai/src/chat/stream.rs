//! Streaming state machine: SSE chunks → [`StreamPart`].
//!
//! Mirrors the `TransformStream` body in `xai-chat-language-model.ts`'s
//! `doStream`. Stateful accumulator that consumes one chunk at a time and
//! returns the parts to forward, plus a finalizer that flushes the trailing
//! `Finish` frame.
//!
//! Per-id block tracking:
//! - Text blocks: id = `text-{response_id_or_choice_index}`
//! - Reasoning blocks: id = `reasoning-{response_id_or_choice_index}`
//! - Tool blocks: tool-call-id (xAI delivers each tool call in a single chunk)
//!
//! When a text delta arrives while a reasoning block is active, we close
//! the reasoning block first (matches upstream behaviour).
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;

use llmsdk_provider::language_model::{
    FinishReason, FinishReasonKind, ResponseMetadata, Source, StreamPart, ToolCallPart,
};
use llmsdk_provider::shared::Warning;
use llmsdk_provider_utils::time::rfc3339_from_unix_seconds;

use super::finish_reason::map as map_finish_reason;
use super::parse_response::next_id as next_citation_id;
use super::usage;
use super::wire::{ChatChunk, WireUsage};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    #[default]
    Text,
    Reasoning,
}

#[derive(Debug, Default)]
struct Block {
    kind: BlockKind,
    ended: bool,
}

/// State machine driving an xAI Chat Completions stream.
#[derive(Debug)]
pub(crate) struct StreamState {
    initial_warnings: Option<Vec<Warning>>,
    finish_reason: FinishReason,
    last_usage: Option<WireUsage>,
    metadata_emitted: bool,
    blocks: HashMap<String, Block>,
    last_reasoning_deltas: HashMap<String, String>,
    active_reasoning_block_id: Option<String>,
    response_id: Option<String>,
    last_assistant_content: Option<String>,
    citation_id_seed: u64,
}

impl StreamState {
    /// Build with the warnings collected during request building and the
    /// last assistant message content (used for prefix-duplicate detection).
    pub(crate) fn new(warnings: Vec<Warning>, last_assistant_content: Option<String>) -> Self {
        Self {
            initial_warnings: Some(warnings),
            finish_reason: FinishReason::new(FinishReasonKind::Other),
            last_usage: None,
            metadata_emitted: false,
            blocks: HashMap::new(),
            last_reasoning_deltas: HashMap::new(),
            active_reasoning_block_id: None,
            response_id: None,
            last_assistant_content,
            citation_id_seed: 0,
        }
    }

    /// Return the very first frame (`StreamStart` with warnings).
    pub(crate) fn start_frames(&mut self) -> Vec<StreamPart> {
        let warnings = self.initial_warnings.take().unwrap_or_default();
        vec![StreamPart::StreamStart { warnings }]
    }

    /// Handle one decoded SSE chunk; returns the frames to forward.
    #[allow(
        clippy::too_many_lines,
        reason = "single dispatcher mirroring upstream's xai-chat-language-model.ts TransformStream body"
    )]
    pub(crate) fn on_chunk(&mut self, chunk: ChatChunk) -> Vec<StreamPart> {
        let mut out = Vec::new();

        if !self.metadata_emitted
            && (chunk.id.is_some() || chunk.created.is_some() || chunk.model.is_some())
        {
            self.metadata_emitted = true;
            self.response_id.clone_from(&chunk.id);
            out.push(StreamPart::ResponseMetadata(ResponseMetadata {
                id: chunk.id.clone(),
                timestamp: chunk.created.map(rfc3339_from_unix_seconds),
                model_id: chunk.model.clone(),
                headers: None,
            }));
        }
        if self.response_id.is_none() && chunk.id.is_some() {
            self.response_id.clone_from(&chunk.id);
        }

        if let Some(citations) = chunk.citations {
            for url in citations {
                let id = next_citation_id(&mut self.citation_id_seed);
                out.push(StreamPart::Source(Source::Url {
                    id,
                    url,
                    title: None,
                    provider_metadata: None,
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

        let choice_index = choice.index;
        let key_seed = self
            .response_id
            .clone()
            .unwrap_or_else(|| choice_index.to_string());

        // Text content
        if let Some(text) = delta.content
            && !text.is_empty()
        {
            // close any active reasoning block first
            if let Some(active_id) = self.active_reasoning_block_id.take()
                && let Some(block) = self.blocks.get_mut(&active_id)
                && !block.ended
            {
                out.push(StreamPart::ReasoningEnd {
                    id: active_id,
                    provider_metadata: None,
                });
                block.ended = true;
            }

            // skip prefix duplicate
            if self.last_assistant_content.as_deref() != Some(text.as_str()) {
                let block_id = format!("text-{key_seed}");
                if !self.blocks.contains_key(&block_id) {
                    self.blocks.insert(
                        block_id.clone(),
                        Block {
                            kind: BlockKind::Text,
                            ended: false,
                        },
                    );
                    out.push(StreamPart::TextStart {
                        id: block_id.clone(),
                        provider_metadata: None,
                    });
                }
                out.push(StreamPart::TextDelta {
                    id: block_id,
                    delta: text,
                    provider_metadata: None,
                });
            }
        }

        // Reasoning content
        if let Some(reasoning) = delta.reasoning_content
            && !reasoning.is_empty()
        {
            let block_id = format!("reasoning-{key_seed}");
            // ai-sdk dedupes when the exact same reasoning delta arrives twice
            // (xAI sometimes echoes the cumulative reasoning instead of a delta).
            if self
                .last_reasoning_deltas
                .get(&block_id)
                .map(String::as_str)
                != Some(reasoning.as_str())
            {
                self.last_reasoning_deltas
                    .insert(block_id.clone(), reasoning.clone());
                if !self.blocks.contains_key(&block_id) {
                    self.blocks.insert(
                        block_id.clone(),
                        Block {
                            kind: BlockKind::Reasoning,
                            ended: false,
                        },
                    );
                    self.active_reasoning_block_id = Some(block_id.clone());
                    out.push(StreamPart::ReasoningStart {
                        id: block_id.clone(),
                        provider_metadata: None,
                    });
                }
                out.push(StreamPart::ReasoningDelta {
                    id: block_id,
                    delta: reasoning,
                    provider_metadata: None,
                });
            }
        }

        // Tool calls — xAI delivers each in one chunk (id + name + full args).
        if let Some(tool_calls) = delta.tool_calls {
            // end active reasoning block first
            if let Some(active_id) = self.active_reasoning_block_id.take()
                && let Some(block) = self.blocks.get_mut(&active_id)
                && !block.ended
            {
                out.push(StreamPart::ReasoningEnd {
                    id: active_id,
                    provider_metadata: None,
                });
                block.ended = true;
            }

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

    /// Surface an error payload (e.g. JSON body returned as application/json
    /// instead of SSE) as an in-stream error.
    pub(crate) fn on_error(&mut self, message: &str, code: Option<&str>) -> Vec<StreamPart> {
        self.finish_reason = FinishReason::new(FinishReasonKind::Error);
        let mut payload = serde_json::Map::new();
        payload.insert(
            "message".into(),
            serde_json::Value::String(message.to_owned()),
        );
        if let Some(c) = code {
            payload.insert("code".into(), serde_json::Value::String(c.to_owned()));
        }
        vec![StreamPart::Error {
            error: serde_json::Value::Object(payload),
        }]
    }

    /// Final flush: emit any pending `*End` frames, then `Finish`.
    pub(crate) fn flush(self) -> Vec<StreamPart> {
        let mut out = Vec::new();

        for (id, block) in self.blocks {
            if block.ended {
                continue;
            }
            match block.kind {
                BlockKind::Text => out.push(StreamPart::TextEnd {
                    id,
                    provider_metadata: None,
                }),
                BlockKind::Reasoning => out.push(StreamPart::ReasoningEnd {
                    id,
                    provider_metadata: None,
                }),
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::wire::{
        ChatChunkChoice, ChatChunkDelta, WireFunctionCall, WireToolCall, WireToolCallKind,
    };

    fn delta(content: Option<&str>, reasoning: Option<&str>, finish: Option<&str>) -> ChatChunk {
        ChatChunk {
            choices: vec![ChatChunkChoice {
                delta: Some(ChatChunkDelta {
                    content: content.map(str::to_owned),
                    reasoning_content: reasoning.map(str::to_owned),
                    ..Default::default()
                }),
                finish_reason: finish.map(str::to_owned),
                index: 0,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn start_then_text_then_finish() {
        let mut state = StreamState::new(vec![], None);
        let s = state.start_frames();
        assert!(matches!(s[0], StreamPart::StreamStart { .. }));

        let f1 = state.on_chunk(delta(Some("hel"), None, None));
        let f2 = state.on_chunk(delta(Some("lo"), None, None));
        state.on_chunk(delta(None, None, Some("stop")));

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
    fn reasoning_then_text_closes_reasoning_first() {
        let mut state = StreamState::new(vec![], None);
        let _ = state.start_frames();
        let _ = state.on_chunk(delta(None, Some("think"), None));
        let frames = state.on_chunk(delta(Some("ans"), None, None));
        // first frame is reasoning-end, then text-start, then text-delta
        assert!(matches!(&frames[0], StreamPart::ReasoningEnd { .. }));
        assert!(matches!(&frames[1], StreamPart::TextStart { .. }));
        assert!(matches!(&frames[2], StreamPart::TextDelta { .. }));
    }

    #[test]
    fn reasoning_delta_dedupes_repeat() {
        let mut state = StreamState::new(vec![], None);
        let _ = state.start_frames();
        let f1 = state.on_chunk(delta(None, Some("think"), None));
        let f2 = state.on_chunk(delta(None, Some("think"), None));
        // f1 emits start + delta; f2 dedupes everything
        assert_eq!(f1.len(), 2);
        assert_eq!(f2.len(), 0);
    }

    #[test]
    fn tool_call_one_chunk() {
        let mut state = StreamState::new(vec![], None);
        let _ = state.start_frames();
        let frames = state.on_chunk(ChatChunk {
            choices: vec![ChatChunkChoice {
                delta: Some(ChatChunkDelta {
                    tool_calls: Some(vec![WireToolCall {
                        id: "call_w".into(),
                        kind: WireToolCallKind::Function,
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
    fn citations_become_sources() {
        let mut state = StreamState::new(vec![], None);
        let _ = state.start_frames();
        let frames = state.on_chunk(ChatChunk {
            citations: Some(vec!["https://example.com/x".into()]),
            ..Default::default()
        });
        let StreamPart::Source(Source::Url { url, .. }) = &frames[0] else {
            panic!("expected Source");
        };
        assert_eq!(url, "https://example.com/x");
    }

    #[test]
    fn parse_error_marks_finish_as_error() {
        let mut state = StreamState::new(vec![], None);
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
