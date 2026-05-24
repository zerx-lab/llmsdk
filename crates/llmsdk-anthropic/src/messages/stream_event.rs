//! Wire types for Messages-API SSE events.
//!
//! Mirrors `anthropicChunkSchema` (subset) — only the variants M6 handles:
//! `message_start`, `content_block_start`, `content_block_delta`,
//! `content_block_stop`, `message_delta`, `message_stop`, `ping`, `error`.
//! Server-tool / thinking event types deserialize to [`StreamEvent::Other`]
//! and are dropped.
// Rust guideline compliant 2026-02-21

use serde::Deserialize;
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StreamMessageMeta {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    pub usage: MessageStartUsage,
}

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
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
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum BlockDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MessageDeltaInner {
    #[serde(default)]
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MessageDeltaUsage {
    #[serde(default)]
    pub output_tokens: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StreamError {
    #[serde(default, rename = "type")]
    pub _kind: Option<String>,
    pub message: String,
}
