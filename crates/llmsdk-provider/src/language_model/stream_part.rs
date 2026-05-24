//! Streamed delta parts produced by `do_stream`.
//!
//! Mirrors `language-model-v4-stream-part.ts`. Wire format and tag names
//! match ai-sdk exactly.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

use crate::json::JsonValue;
use crate::shared::{ProviderMetadata, Warning};

use crate::shared::FileData;

use super::content::{Source, ToolApprovalRequest, ToolResult};
use super::finish_reason::FinishReason;
use super::prompt::{FilePart, ToolCallPart};
use super::result::ResponseMetadata;
use super::usage::Usage;

/// One unit emitted on the stream returned by
/// [`super::LanguageModel::do_stream`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum StreamPart {
    /// Start of a text block.
    TextStart {
        /// Block id (used to correlate `text-delta` / `text-end`).
        id: String,
        /// Provider-specific metadata.
        #[serde(
            default,
            rename = "providerMetadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadata>,
    },
    /// Incremental text fragment.
    TextDelta {
        /// Block id.
        id: String,
        /// Text fragment.
        delta: String,
        /// Provider-specific metadata.
        #[serde(
            default,
            rename = "providerMetadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadata>,
    },
    /// End of a text block.
    TextEnd {
        /// Block id.
        id: String,
        /// Provider-specific metadata.
        #[serde(
            default,
            rename = "providerMetadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadata>,
    },
    /// Start of a reasoning block.
    ReasoningStart {
        /// Block id.
        id: String,
        /// Provider-specific metadata.
        #[serde(
            default,
            rename = "providerMetadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadata>,
    },
    /// Incremental reasoning fragment.
    ReasoningDelta {
        /// Block id.
        id: String,
        /// Reasoning fragment.
        delta: String,
        /// Provider-specific metadata.
        #[serde(
            default,
            rename = "providerMetadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadata>,
    },
    /// End of a reasoning block.
    ReasoningEnd {
        /// Block id.
        id: String,
        /// Provider-specific metadata.
        #[serde(
            default,
            rename = "providerMetadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadata>,
    },
    /// Start of a tool input being streamed.
    ToolInputStart {
        /// Tool call id.
        id: String,
        /// Tool name.
        #[serde(rename = "toolName")]
        tool_name: String,
        /// `true` if executed by the provider.
        #[serde(
            default,
            rename = "providerExecuted",
            skip_serializing_if = "Option::is_none"
        )]
        provider_executed: Option<bool>,
        /// `true` if defined at runtime.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dynamic: Option<bool>,
        /// Optional display title.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        /// Provider-specific metadata.
        #[serde(
            default,
            rename = "providerMetadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadata>,
    },
    /// Streamed chunk of a tool's input JSON.
    ToolInputDelta {
        /// Tool call id.
        id: String,
        /// JSON fragment.
        delta: String,
        /// Provider-specific metadata.
        #[serde(
            default,
            rename = "providerMetadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadata>,
    },
    /// End of a tool's input stream.
    ToolInputEnd {
        /// Tool call id.
        id: String,
        /// Provider-specific metadata.
        #[serde(
            default,
            rename = "providerMetadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadata>,
    },
    /// Approval requested for a provider-executed tool call.
    ToolApprovalRequest(ToolApprovalRequest),
    /// Final tool call with assembled input.
    ToolCall(ToolCallPart),
    /// Tool result emitted by a provider-executed tool.
    ToolResult(ToolResult),
    /// Provider-specific custom content.
    Custom {
        /// Custom kind tag, e.g. `"openai.web_search_result"`.
        kind: String,
        /// Provider-specific metadata.
        #[serde(
            default,
            rename = "providerMetadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadata>,
    },
    /// Citation / grounding source.
    Source(Source),
    /// File generated by the model (mid-stream emission).
    ///
    /// Mirrors ai-sdk's `LanguageModelV4File` stream part. The wire tag is
    /// `"file"` and shape matches a [`FilePart`] (filename / data / media
    /// type / provider options).
    File(FilePart),
    /// File generated as part of a reasoning trace (mid-stream emission).
    ///
    /// Mirrors ai-sdk's `LanguageModelV4ReasoningFile` stream part. Wire
    /// tag is `"reasoning-file"`.
    ReasoningFile {
        /// File payload.
        data: FileData,
        /// IANA media type.
        #[serde(rename = "mediaType")]
        media_type: String,
        /// Provider-specific metadata.
        #[serde(
            default,
            rename = "providerMetadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadata>,
    },
    /// Stream-start metadata.
    StreamStart {
        /// Warnings for the call.
        warnings: Vec<Warning>,
    },
    /// Response-level metadata available mid-stream.
    ResponseMetadata(ResponseMetadata),
    /// Terminal frame with totals.
    Finish {
        /// Final token usage.
        usage: Usage,
        /// Why the model stopped.
        #[serde(rename = "finishReason")]
        finish_reason: FinishReason,
        /// Provider-specific metadata.
        #[serde(
            default,
            rename = "providerMetadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadata>,
    },
    /// Raw provider chunk (only when `include_raw_chunks` is set).
    Raw {
        /// Provider-native value.
        #[serde(rename = "rawValue")]
        raw_value: JsonValue,
    },
    /// In-stream error from the provider.
    ///
    /// The stream is still alive; the outer `Result` is `Ok`.
    Error {
        /// Error payload as provided by the upstream.
        error: JsonValue,
    },
}
