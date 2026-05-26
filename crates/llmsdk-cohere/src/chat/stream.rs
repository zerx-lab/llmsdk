//! Streaming state machine: SSE chunks → [`StreamPart`].
//!
//! Mirrors the `TransformStream` body in `cohere-chat-language-model.ts`'s
//! `doStream`. Cohere streams a typed event union — see the upstream zod
//! schema and `https://docs.cohere.com/v2/docs/streaming` for the wire shape.
//!
//! Per-block tracking:
//! - `content-start` opens either a text or reasoning block keyed on `index`.
//! - `content-delta` routes to the currently open block.
//! - `content-end` closes the block.
//! - `tool-call-{start,delta,end}` accumulate one tool call's arguments and
//!   emit `ToolInputStart` / `ToolInputDelta` / `ToolInputEnd` / `ToolCall`.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;

use llmsdk_provider::language_model::{
    FinishReason, FinishReasonKind, ResponseMetadata, StreamPart, ToolCallPart,
};
use llmsdk_provider::shared::Warning;

use super::finish_reason::map as map_finish_reason;
use super::parse_response::{citation_to_source, next_id};
use super::usage;
use super::wire::{ChatChunk, ContentStartDeltaContent, WireUsage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Text,
    Reasoning,
}

#[derive(Debug)]
struct PendingTool {
    id: String,
    name: String,
    arguments: String,
    finished: bool,
}

/// Cohere chat streaming state machine.
#[derive(Debug)]
pub(crate) struct StreamState {
    initial_warnings: Option<Vec<Warning>>,
    finish_reason: FinishReason,
    last_usage: Option<WireUsage>,
    blocks: HashMap<u32, BlockKind>,
    block_ended: HashMap<u32, bool>,
    pending_tool: Option<PendingTool>,
    citation_id_seed: u64,
    /// Reserved virtual block id used to surface `tool_plan` streaming deltas as
    /// a single contiguous reasoning block (Cohere has no `index` for it).
    tool_plan_started: bool,
    tool_plan_ended: bool,
}

impl StreamState {
    /// Build with the warnings collected during request building.
    pub(crate) fn new(warnings: Vec<Warning>) -> Self {
        Self {
            initial_warnings: Some(warnings),
            finish_reason: FinishReason::new(FinishReasonKind::Other),
            last_usage: None,
            blocks: HashMap::new(),
            block_ended: HashMap::new(),
            pending_tool: None,
            citation_id_seed: 0,
            tool_plan_started: false,
            tool_plan_ended: false,
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
        reason = "single dispatcher mirroring upstream's cohere-chat-language-model.ts TransformStream body"
    )]
    pub(crate) fn on_chunk(&mut self, chunk: ChatChunk) -> Vec<StreamPart> {
        let mut out = Vec::new();

        match chunk {
            ChatChunk::MessageStart { id } => {
                out.push(StreamPart::ResponseMetadata(ResponseMetadata {
                    id,
                    timestamp: None,
                    model_id: None,
                    headers: None,
                }));
            }
            ChatChunk::ContentStart { index, delta } => match delta.message.content {
                ContentStartDeltaContent::Thinking { .. } => {
                    self.blocks.insert(index, BlockKind::Reasoning);
                    self.block_ended.insert(index, false);
                    out.push(StreamPart::ReasoningStart {
                        id: index.to_string(),
                        provider_metadata: None,
                    });
                }
                ContentStartDeltaContent::Text { .. } => {
                    self.blocks.insert(index, BlockKind::Text);
                    self.block_ended.insert(index, false);
                    out.push(StreamPart::TextStart {
                        id: index.to_string(),
                        provider_metadata: None,
                    });
                }
            },
            ChatChunk::ContentDelta { index, delta } => {
                let content = delta.message.content;
                let kind = self.blocks.get(&index).copied();
                if let Some(reasoning) = content.thinking
                    && matches!(kind, Some(BlockKind::Reasoning))
                {
                    out.push(StreamPart::ReasoningDelta {
                        id: index.to_string(),
                        delta: reasoning,
                        provider_metadata: None,
                    });
                } else if let Some(text) = content.text
                    && matches!(kind, Some(BlockKind::Text))
                {
                    out.push(StreamPart::TextDelta {
                        id: index.to_string(),
                        delta: text,
                        provider_metadata: None,
                    });
                }
            }
            ChatChunk::ContentEnd { index } => {
                let kind = self.blocks.get(&index).copied();
                self.block_ended.insert(index, true);
                match kind {
                    Some(BlockKind::Reasoning) => out.push(StreamPart::ReasoningEnd {
                        id: index.to_string(),
                        provider_metadata: None,
                    }),
                    Some(BlockKind::Text) | None => out.push(StreamPart::TextEnd {
                        id: index.to_string(),
                        provider_metadata: None,
                    }),
                }
            }
            ChatChunk::ToolPlanDelta { delta } => {
                let text = delta.plan_text().to_owned();
                if text.is_empty() {
                    return out;
                }
                if !self.tool_plan_started {
                    self.tool_plan_started = true;
                    let mut meta = serde_json::Map::new();
                    meta.insert("toolPlan".into(), serde_json::Value::Bool(true));
                    let mut wrapper = std::collections::HashMap::new();
                    wrapper.insert("cohere".to_owned(), meta);
                    out.push(StreamPart::ReasoningStart {
                        id: "tool-plan".to_owned(),
                        provider_metadata: Some(wrapper),
                    });
                }
                out.push(StreamPart::ReasoningDelta {
                    id: "tool-plan".to_owned(),
                    delta: text,
                    provider_metadata: None,
                });
            }
            ChatChunk::ToolCallStart { delta } => {
                let tc = delta.message.tool_calls;
                let id = tc.id.clone();
                let name = tc.function.name.clone();
                let initial = tc.function.arguments.clone();

                out.push(StreamPart::ToolInputStart {
                    id: id.clone(),
                    tool_name: name.clone(),
                    provider_executed: None,
                    dynamic: None,
                    title: None,
                    provider_metadata: None,
                });

                if !initial.is_empty() {
                    out.push(StreamPart::ToolInputDelta {
                        id: id.clone(),
                        delta: initial.clone(),
                        provider_metadata: None,
                    });
                }

                self.pending_tool = Some(PendingTool {
                    id,
                    name,
                    arguments: initial,
                    finished: false,
                });
            }
            ChatChunk::ToolCallDelta { delta } => {
                let args = delta.message.tool_calls.function.arguments;
                if let Some(pending) = self.pending_tool.as_mut()
                    && !pending.finished
                {
                    pending.arguments.push_str(&args);
                    out.push(StreamPart::ToolInputDelta {
                        id: pending.id.clone(),
                        delta: args,
                        provider_metadata: None,
                    });
                }
            }
            ChatChunk::ToolCallEnd => {
                if let Some(mut pending) = self.pending_tool.take()
                    && !pending.finished
                {
                    out.push(StreamPart::ToolInputEnd {
                        id: pending.id.clone(),
                        provider_metadata: None,
                    });

                    let trimmed = pending.arguments.trim();
                    let input = if trimmed.is_empty() {
                        serde_json::Value::Object(serde_json::Map::new())
                    } else {
                        match serde_json::from_str::<serde_json::Value>(trimmed) {
                            Ok(v) => v,
                            Err(e) => {
                                out.extend(self.on_parse_error(trimmed, &e.to_string()));
                                return out;
                            }
                        }
                    };

                    out.push(StreamPart::ToolCall(ToolCallPart {
                        tool_call_id: pending.id.clone(),
                        tool_name: pending.name.clone(),
                        input,
                        provider_executed: None,
                        dynamic: None,
                        provider_options: None,
                    }));
                    pending.finished = true;
                }
            }
            ChatChunk::CitationStart { delta } => {
                if let Some(delta) = delta
                    && let Some(message) = delta.message
                    && let Some(citation) = message.citations
                {
                    out.push(StreamPart::Source(citation_source(
                        citation,
                        &mut self.citation_id_seed,
                    )));
                }
            }
            ChatChunk::CitationEnd | ChatChunk::Other => {}
            ChatChunk::MessageEnd { delta } => {
                self.finish_reason = map_finish_reason(delta.finish_reason.as_deref());
                if let Some(u) = delta.usage {
                    self.last_usage = Some(u);
                }
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
    pub(crate) fn flush(mut self) -> Vec<StreamPart> {
        let mut out = Vec::new();

        // Close any blocks that never received content-end.
        for (index, kind) in &self.blocks {
            if self.block_ended.get(index).copied().unwrap_or(false) {
                continue;
            }
            match kind {
                BlockKind::Text => out.push(StreamPart::TextEnd {
                    id: index.to_string(),
                    provider_metadata: None,
                }),
                BlockKind::Reasoning => out.push(StreamPart::ReasoningEnd {
                    id: index.to_string(),
                    provider_metadata: None,
                }),
            }
        }

        // Close a pending tool that never received tool-call-end.
        if let Some(pending) = self.pending_tool.take()
            && !pending.finished
        {
            out.push(StreamPart::ToolInputEnd {
                id: pending.id.clone(),
                provider_metadata: None,
            });
        }

        // Close the tool-plan reasoning block if it was opened.
        if self.tool_plan_started && !self.tool_plan_ended {
            self.tool_plan_ended = true;
            out.push(StreamPart::ReasoningEnd {
                id: "tool-plan".to_owned(),
                provider_metadata: None,
            });
        }

        let usage_value = usage::convert(self.last_usage.as_ref());

        out.push(StreamPart::Finish {
            usage: usage_value,
            finish_reason: self.finish_reason,
            provider_metadata: None,
        });
        out
    }
}

fn citation_source(
    citation: super::wire::ChatResponseCitation,
    seed: &mut u64,
) -> llmsdk_provider::language_model::Source {
    // We piggyback on the non-stream helper, then unwrap to the Source.
    let content = citation_to_source(citation, seed);
    let llmsdk_provider::language_model::Content::Source(src) = content else {
        unreachable!("citation_to_source always returns Content::Source")
    };
    src
}

// Force the citation id seed helper used in tests to compile.
#[allow(dead_code, reason = "referenced only in unit tests")]
fn _force_next_id_use(seed: &mut u64) -> String {
    next_id(seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::wire::{
        ContentDeltaContent, ContentDeltaMessage, ContentDeltaWrapper, ContentStartDelta,
        ContentStartDeltaMessage, MessageEndDelta, ToolCallDeltaCall, ToolCallDeltaFunction,
        ToolCallDeltaMessage, ToolCallDeltaWrapper, ToolCallStartDelta, ToolCallStartDeltaMessage,
        WireFunctionCall, WireToolCall, WireToolCallKind, WireUsageTokens,
    };

    fn cs_text(index: u32) -> ChatChunk {
        ChatChunk::ContentStart {
            index,
            delta: ContentStartDelta {
                message: ContentStartDeltaMessage {
                    content: ContentStartDeltaContent::Text {
                        text: String::new(),
                    },
                },
            },
        }
    }

    fn cd_text(index: u32, t: &str) -> ChatChunk {
        ChatChunk::ContentDelta {
            index,
            delta: ContentDeltaWrapper {
                message: ContentDeltaMessage {
                    content: ContentDeltaContent {
                        text: Some(t.to_owned()),
                        thinking: None,
                    },
                },
            },
        }
    }

    fn cd_thinking(index: u32, t: &str) -> ChatChunk {
        ChatChunk::ContentDelta {
            index,
            delta: ContentDeltaWrapper {
                message: ContentDeltaMessage {
                    content: ContentDeltaContent {
                        text: None,
                        thinking: Some(t.to_owned()),
                    },
                },
            },
        }
    }

    #[test]
    fn text_block_full_lifecycle() {
        let mut state = StreamState::new(vec![]);
        let s = state.start_frames();
        assert!(matches!(s[0], StreamPart::StreamStart { .. }));
        let f1 = state.on_chunk(cs_text(0));
        let f2 = state.on_chunk(cd_text(0, "hel"));
        let f3 = state.on_chunk(cd_text(0, "lo"));
        let f4 = state.on_chunk(ChatChunk::ContentEnd { index: 0 });
        let _ = state.on_chunk(ChatChunk::MessageEnd {
            delta: MessageEndDelta {
                finish_reason: Some("COMPLETE".into()),
                usage: Some(WireUsage {
                    billed_units: Some(WireUsageTokens {
                        input_tokens: Some(1),
                        output_tokens: Some(2),
                    }),
                    tokens: Some(WireUsageTokens {
                        input_tokens: Some(1),
                        output_tokens: Some(2),
                    }),
                }),
            },
        });

        assert!(matches!(f1[0], StreamPart::TextStart { .. }));
        assert!(matches!(&f2[0], StreamPart::TextDelta { delta, .. } if delta == "hel"));
        assert!(matches!(&f3[0], StreamPart::TextDelta { delta, .. } if delta == "lo"));
        assert!(matches!(&f4[0], StreamPart::TextEnd { .. }));

        let tail = state.flush();
        let StreamPart::Finish {
            finish_reason,
            usage,
            ..
        } = tail.last().unwrap()
        else {
            panic!("expected Finish");
        };
        assert_eq!(finish_reason.unified, FinishReasonKind::Stop);
        assert_eq!(usage.input_tokens.total, Some(1));
    }

    #[test]
    fn reasoning_block_emits_reasoning_frames() {
        let mut state = StreamState::new(vec![]);
        let _ = state.start_frames();
        let f1 = state.on_chunk(ChatChunk::ContentStart {
            index: 0,
            delta: ContentStartDelta {
                message: ContentStartDeltaMessage {
                    content: ContentStartDeltaContent::Thinking {
                        thinking: String::new(),
                    },
                },
            },
        });
        let f2 = state.on_chunk(cd_thinking(0, "let me think"));
        let f3 = state.on_chunk(ChatChunk::ContentEnd { index: 0 });
        assert!(matches!(f1[0], StreamPart::ReasoningStart { .. }));
        assert!(
            matches!(&f2[0], StreamPart::ReasoningDelta { delta, .. } if delta == "let me think")
        );
        assert!(matches!(&f3[0], StreamPart::ReasoningEnd { .. }));
    }

    #[test]
    fn tool_call_three_chunks() {
        let mut state = StreamState::new(vec![]);
        let _ = state.start_frames();
        let f1 = state.on_chunk(ChatChunk::ToolCallStart {
            delta: ToolCallStartDelta {
                message: ToolCallStartDeltaMessage {
                    tool_calls: WireToolCall {
                        id: "c1".into(),
                        kind: WireToolCallKind::Function,
                        function: WireFunctionCall {
                            name: "weather".into(),
                            arguments: r#"{"city""#.into(),
                        },
                    },
                },
            },
        });
        let f2 = state.on_chunk(ChatChunk::ToolCallDelta {
            delta: ToolCallDeltaWrapper {
                message: ToolCallDeltaMessage {
                    tool_calls: ToolCallDeltaCall {
                        function: ToolCallDeltaFunction {
                            arguments: r#":"NYC"}"#.into(),
                        },
                    },
                },
            },
        });
        let f3 = state.on_chunk(ChatChunk::ToolCallEnd);

        assert!(matches!(&f1[0], StreamPart::ToolInputStart { id, .. } if id == "c1"));
        assert!(matches!(&f1[1], StreamPart::ToolInputDelta { .. }));
        assert!(matches!(&f2[0], StreamPart::ToolInputDelta { .. }));
        assert!(matches!(&f3[0], StreamPart::ToolInputEnd { .. }));
        let StreamPart::ToolCall(tc) = &f3[1] else {
            panic!("expected ToolCall");
        };
        assert_eq!(tc.tool_call_id, "c1");
        assert_eq!(tc.input["city"], "NYC");
    }

    #[test]
    fn malformed_tool_call_args_emit_stream_error() {
        // Mirrors ai-sdk: parseJSON throws JSONParseError on invalid JSON
        // (commit 3cfb7621e). Rust must not silently fall back to a string.
        let mut state = StreamState::new(vec![]);
        let _ = state.start_frames();
        let _ = state.on_chunk(ChatChunk::ToolCallStart {
            delta: ToolCallStartDelta {
                message: ToolCallStartDeltaMessage {
                    tool_calls: WireToolCall {
                        id: "c1".into(),
                        kind: WireToolCallKind::Function,
                        function: WireFunctionCall {
                            name: "weather".into(),
                            arguments: "not-json".into(),
                        },
                    },
                },
            },
        });
        let end = state.on_chunk(ChatChunk::ToolCallEnd);
        assert!(matches!(end.last().unwrap(), StreamPart::Error { .. }));
        assert!(!end.iter().any(|p| matches!(p, StreamPart::ToolCall(_))));
        let tail = state.flush();
        let StreamPart::Finish { finish_reason, .. } = tail.last().unwrap() else {
            panic!("expected Finish");
        };
        assert_eq!(finish_reason.unified, FinishReasonKind::Error);
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
