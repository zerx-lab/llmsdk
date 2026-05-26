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
use std::sync::Arc;

use llmsdk_provider::language_model::{
    FinishReason, FinishReasonKind, Source, StreamPart, ToolCallPart,
};
use llmsdk_provider::shared::{ProviderMetadata, ProviderOptions, Warning};
use serde_json::{Map, Value as JsonValue};

use crate::config::GenerateIdFn;

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
        /// Normalized `caller` (`snake_case` `tool_id` → `camelCase` `toolId`)
        /// pre-staged at `content_block_start` so the closing `tool-call`
        /// frame can attach it via `provider_metadata.anthropic.caller`,
        /// mirroring upstream `anthropic-language-model.ts:1659`.
        caller: Option<JsonValue>,
    },
    /// Extended-thinking block; tracks the latest signature observed via
    /// `signature_delta`.
    Reasoning { id: String },
}

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
            .finish()
    }
}

impl StreamState {
    #[cfg(test)]
    pub(crate) fn new(warnings: Vec<Warning>) -> Self {
        Self::with_generate_id(warnings, None, false)
    }

    pub(crate) fn with_generate_id(
        warnings: Vec<Warning>,
        generate_id: Option<Arc<GenerateIdFn>>,
        mark_code_execution_dynamic: bool,
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
                    caller,
                } => {
                    out.push(StreamPart::ToolInputEnd {
                        id: id.clone(),
                        provider_metadata: None,
                    });
                    out.push(StreamPart::ToolCall(build_tool_call(
                        id, name, arguments, caller,
                    )));
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
            BlockKind::Text { id } => vec![StreamPart::TextEnd {
                id,
                provider_metadata: None,
            }],
            BlockKind::ToolUse {
                id,
                name,
                arguments,
                caller,
            } => vec![
                StreamPart::ToolInputEnd {
                    id: id.clone(),
                    provider_metadata: None,
                },
                StreamPart::ToolCall(build_tool_call(id, name, arguments, caller)),
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
    ToolCallPart {
        tool_call_id: id,
        tool_name: name,
        input,
        provider_executed: None,
        dynamic: None,
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
}
