//! `OpenAI` error response parsing.
//!
//! Mirrors `@ai-sdk/openai/src/openai-error.ts`. The `OpenAI` error envelope
//! is `{ "error": { "message": "...", "type": "...", "code": "..." } }`.
//! We extract `message` to use as the human-readable summary; `type` /
//! `code` are surfaced via [`ProviderError::status_code`] and the response
//! body.
// Rust guideline compliant 2026-02-21

use serde::Deserialize;

/// Best-effort `OpenAI` error body.
///
/// Tolerant: any field may be missing — provider-compatible APIs deviate.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiErrorBody {
    pub(crate) error: OpenAiErrorInner,
}

/// Inner `error` object on the `OpenAI` envelope.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiErrorInner {
    pub(crate) message: String,
    #[serde(default, rename = "type")]
    pub(crate) _kind: Option<String>,
    #[serde(default)]
    pub(crate) _code: Option<serde_json::Value>,
}

/// Extract a one-line error message from a raw response body.
///
/// Falls back to the body itself (trimmed) when parsing fails. This keeps
/// us diagnosable against `OpenAI`-compatible providers that may return
/// different shapes.
pub(crate) fn extract_error_message(body: &str) -> String {
    match serde_json::from_str::<OpenAiErrorBody>(body) {
        Ok(parsed) => parsed.error.message,
        Err(_) => body.trim().to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_envelope() {
        let body = r#"{"error":{"message":"rate limited","type":"requests","code":"rate_limit_exceeded"}}"#;
        assert_eq!(extract_error_message(body), "rate limited");
    }

    #[test]
    fn falls_back_to_raw_body() {
        assert_eq!(extract_error_message("oops"), "oops");
    }

    #[test]
    fn tolerates_missing_optional_fields() {
        let body = r#"{"error":{"message":"missing key"}}"#;
        assert_eq!(extract_error_message(body), "missing key");
    }
}
