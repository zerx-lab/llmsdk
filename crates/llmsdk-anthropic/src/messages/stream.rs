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

use std::collections::BTreeMap;

use llmsdk_provider::language_model::{FinishReason, FinishReasonKind, StreamPart, ToolCallPart};
use llmsdk_provider::shared::{ProviderMetadata, Warning};
use serde_json::Map;

use super::finish_reason::map as map_finish_reason;
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
    },
    /// Extended-thinking block; tracks the latest signature observed via
    /// `signature_delta`.
    Reasoning { id: String },
}

#[derive(Debug)]
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
}

impl StreamState {
    pub(crate) fn new(warnings: Vec<Warning>) -> Self {
        Self {
            initial_warnings: Some(warnings),
            finish_reason: FinishReason::new(FinishReasonKind::Other),
            usage: ResponseUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
            metadata: None,
            metadata_emitted: false,
            blocks: BTreeMap::new(),
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
                if let Some(u) = usage
                    && let Some(out_tokens) = u.output_tokens
                {
                    self.usage.output_tokens = out_tokens;
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
                BlockKind::Text { id } => out.push(StreamPart::TextEnd {
                    id,
                    provider_metadata: None,
                }),
                BlockKind::ToolUse {
                    id,
                    name,
                    arguments,
                } => {
                    out.push(StreamPart::ToolInputEnd {
                        id: id.clone(),
                        provider_metadata: None,
                    });
                    out.push(StreamPart::ToolCall(build_tool_call(id, name, arguments)));
                }
                BlockKind::Reasoning { id } => out.push(StreamPart::ReasoningEnd {
                    id,
                    provider_metadata: None,
                }),
            }
        }
        out.push(StreamPart::Finish {
            usage: convert_usage(&self.usage),
            finish_reason: self.finish_reason,
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
            BlockStart::ToolUse { id, name, input } => {
                let arguments = match input {
                    Some(v) if !v.is_null() => serde_json::to_string(&v).unwrap_or_default(),
                    _ => String::new(),
                };
                self.blocks.insert(
                    index,
                    BlockKind::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                );
                let mut out = vec![StreamPart::ToolInputStart {
                    id: id.clone(),
                    tool_name: name,
                    provider_executed: None,
                    dynamic: None,
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
                BlockKind::ToolUse { id, arguments, .. },
                BlockDelta::InputJsonDelta { partial_json },
            ) => {
                if partial_json.is_empty() {
                    return Vec::new();
                }
                arguments.push_str(&partial_json);
                vec![StreamPart::ToolInputDelta {
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
            _ => Vec::new(),
        }
    }

    fn on_block_stop(&mut self, index: u32) -> Vec<StreamPart> {
        let Some(kind) = self.blocks.remove(&index) else {
            return Vec::new();
        };
        match kind {
            BlockKind::Text { id } => vec![StreamPart::TextEnd {
                id,
                provider_metadata: None,
            }],
            BlockKind::ToolUse {
                id,
                name,
                arguments,
            } => vec![
                StreamPart::ToolInputEnd {
                    id: id.clone(),
                    provider_metadata: None,
                },
                StreamPart::ToolCall(build_tool_call(id, name, arguments)),
            ],
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

fn build_tool_call(id: String, name: String, arguments: String) -> ToolCallPart {
    let input = if arguments.is_empty() {
        serde_json::Value::Object(serde_json::Map::new())
    } else {
        serde_json::from_str::<serde_json::Value>(&arguments)
            .unwrap_or(serde_json::Value::String(arguments))
    };
    ToolCallPart {
        tool_call_id: id,
        tool_name: name,
        input,
        provider_executed: None,
        dynamic: None,
        provider_options: None,
    }
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
                output_tokens: Some(2),
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
}
