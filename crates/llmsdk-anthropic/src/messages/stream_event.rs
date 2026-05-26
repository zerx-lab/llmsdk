//! Wire types for Messages-API SSE events.
//!
//! Mirrors `anthropicChunkSchema` (subset) — only the variants M6 handles:
//! `message_start`, `content_block_start`, `content_block_delta`,
//! `content_block_stop`, `message_delta`, `message_stop`, `ping`, `error`.
//! Server-tool / thinking event types deserialize to [`StreamEvent::Other`]
//! and are dropped.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum StreamEvent {
    MessageStart {
        message: StreamMessageMeta,
    },
    ContentBlockStart {
        index: u32,
        content_block: BlockStart,
    },
    ContentBlockDelta {
        index: u32,
        delta: BlockDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: MessageDeltaInner,
        #[serde(default)]
        usage: Option<MessageDeltaUsage>,
    },
    MessageStop,
    Ping,
    Error {
        error: StreamError,
    },
    /// Catch-all for server-tool / thinking events that M6 does not
    /// surface.
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct StreamMessageMeta {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    pub usage: MessageStartUsage,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[expect(
    clippy::struct_field_names,
    reason = "field names match Anthropic JSON wire format and must not be renamed"
)]
pub(crate) struct MessageStartUsage {
    pub input_tokens: u64,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum BlockStart {
    Text {
        #[serde(default)]
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Option<JsonValue>,
    },
    /// Extended-thinking block opened by the server.
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
    /// Server-redacted thinking block.
    RedactedThinking {
        #[serde(default)]
        data: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum BlockDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    /// Streamed visible-thinking fragment.
    ThinkingDelta {
        thinking: String,
    },
    /// Streamed signature attached to a thinking block.
    SignatureDelta {
        signature: String,
    },
    /// Citation attached to a text block (`web_search_result_location` /
    /// `page_location` / `char_location`). Raw JSON kept for fidelity.
    CitationsDelta {
        citation: JsonValue,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct MessageDeltaInner {
    #[serde(default)]
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct MessageDeltaUsage {
    #[serde(default)]
    pub output_tokens: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct StreamError {
    #[serde(default, rename = "type")]
    pub _kind: Option<String>,
    pub message: String,
}
