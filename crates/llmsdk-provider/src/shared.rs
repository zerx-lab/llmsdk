//! Shared types reused by language / embedding / image models.
//!
//! Maps to `@ai-sdk/provider`'s `shared/v4/*`.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::json::{JsonObject, JsonValue};

/// Provider-specific options, keyed by provider id.
///
/// Mirrors `SharedV4ProviderOptions`. Outer key is the provider name
/// (e.g. `"openai"`), inner object is provider-defined.
///
/// # Examples
///
/// ```
/// use llmsdk_provider::shared::ProviderOptions;
/// use serde_json::json;
///
/// let mut opts = ProviderOptions::default();
/// opts.insert(
///     "openai".into(),
///     json!({ "reasoningEffort": "high" }).as_object().cloned().unwrap(),
/// );
/// ```
pub type ProviderOptions = HashMap<String, JsonObject>;

/// Provider-specific metadata returned by a provider, keyed by provider id.
///
/// Mirrors `SharedV4ProviderMetadata`.
pub type ProviderMetadata = HashMap<String, JsonObject>;

/// Mapping of provider names to provider-specific file / skill identifiers.
///
/// Mirrors `SharedV4ProviderReference`. Lets the same logical file or skill
/// be referenced across multiple providers without re-uploading.
///
/// # Examples
///
/// ```
/// use llmsdk_provider::shared::ProviderReference;
///
/// let mut r = ProviderReference::new();
/// r.insert("anthropic".into(), "file-abc123".into());
/// assert_eq!(r.get("anthropic"), Some(&"file-abc123".to_owned()));
/// ```
pub type ProviderReference = HashMap<String, String>;

/// HTTP headers attached to a request or response.
///
/// Mirrors `SharedV4Headers`. Value may be `None` when the caller wants the
/// provider to drop a default header.
pub type Headers = HashMap<String, Option<String>>;

/// Provider-emitted warning about a model call.
///
/// Mirrors `SharedV4Warning` (ai-sdk
/// `packages/provider/src/shared/v4/shared-v4-warning.ts`). The four
/// variants are wire-compatible with the upstream `type` tag:
/// `unsupported` / `compatibility` / `deprecated` / `other`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Warning {
    /// A feature is not supported by the model — the request was sent
    /// without it and the result may differ from the caller's intent.
    /// Mirrors upstream `{ type: 'unsupported', feature, details? }`.
    Unsupported {
        /// Name of the feature / setting / tool that was unsupported.
        /// Matches upstream `feature` field (`snake_case` wire name).
        feature: String,
        /// Optional human-readable details.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },

    /// A compatibility-mode feature is in use that may produce suboptimal
    /// results (the request still went through but with a coerced or
    /// downgraded shape). Mirrors upstream
    /// `{ type: 'compatibility', feature, details? }`.
    Compatibility {
        /// Name of the feature operating in compatibility mode.
        feature: String,
        /// Optional human-readable details.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },

    /// A deprecated setting / feature is being used; the message explains
    /// the recommended replacement. Mirrors upstream
    /// `{ type: 'deprecated', setting, message }`.
    Deprecated {
        /// Name of the deprecated setting / feature.
        setting: String,
        /// Human-readable message explaining what to use instead.
        message: String,
    },

    /// Generic warning for cases that don't fit the structured variants.
    /// Mirrors upstream `{ type: 'other', message }`.
    Other {
        /// Human-readable message.
        message: String,
    },
}

/// File data carried in prompts or tool results.
///
/// Mirrors `SharedV4FileData`. A tagged union over inline bytes, URL,
/// provider reference, or inline text.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum FileData {
    /// Inline bytes (base64-encoded when serialized to JSON).
    Data {
        /// Raw bytes or base64 string; provider crates decide encoding on the wire.
        data: FileBytes,
    },
    /// URL pointing to the file.
    Url {
        /// Absolute URL.
        url: String,
    },
    /// Provider-specific reference, e.g. an uploaded file id.
    Reference {
        /// `{ providerId: id }` map.
        reference: JsonObject,
    },
    /// Inline text payload.
    Text {
        /// Text content.
        text: String,
    },
}

/// Either raw bytes or a base64-encoded string.
///
/// Providers serialize bytes as base64 on the wire; this enum lets callers
/// hand off whichever they have without paying a re-encode cost.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum FileBytes {
    /// Base64-encoded string.
    Base64(String),
    /// Raw byte buffer.
    Bytes(Vec<u8>),
}

/// Request metadata for telemetry / debugging.
///
/// Mirrors the `request` field on `*GenerateResult` / `*StreamResult`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RequestInfo {
    /// HTTP body sent to the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<JsonValue>,
}

/// Response metadata for telemetry / debugging.
///
/// Mirrors the `response` field on `*GenerateResult` / `*StreamResult`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ResponseInfo {
    /// Response id reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Timestamp reported by the provider (ISO-8601 string for portability).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    /// Model id reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Response headers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<Headers>,
    /// Raw response body for debugging.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<JsonValue>,
}
