//! `OpenAI` error response parsing.
//!
//! Mirrors `@ai-sdk/openai/src/openai-error.ts`. The `OpenAI` error envelope
//! is `{ "error": { "message": "...", "type": "...", "code": "..." } }`.
//! We extract `message` to use as the human-readable summary; `type` /
//! `code` are surfaced via [`llmsdk_provider::ProviderError::status_code`]
//! and the response body.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::ProviderError;
use serde::Deserialize;

/// Best-effort `OpenAI` error body.
///
/// Tolerant: any field may be missing — provider-compatible APIs deviate.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiErrorBody {
    pub(crate) error: OpenAiErrorInner,
}

/// Inner `error` object on the `OpenAI` envelope.
///
/// Schema mirrors `openai-error.ts` upstream: `message` is required; `type`,
/// `code`, and `param` are loosely typed so OpenAI-compatible providers
/// that omit or rename them still parse.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiErrorInner {
    pub(crate) message: String,
    #[serde(default, rename = "type")]
    pub(crate) _kind: Option<String>,
    #[serde(default)]
    pub(crate) _code: Option<serde_json::Value>,
    /// `error.param` — the field that triggered the error (e.g. `"model"`
    /// for an invalid model id). Optional / forward-compatible.
    #[serde(default)]
    pub(crate) param: Option<serde_json::Value>,
}

/// Extract a one-line error message from a raw response body.
///
/// Falls back to the body itself (trimmed) when parsing fails. This keeps
/// us diagnosable against `OpenAI`-compatible providers that may return
/// different shapes.
pub(crate) fn extract_error_message(body: &str) -> String {
    match serde_json::from_str::<OpenAiErrorBody>(body) {
        Ok(parsed) => {
            // Annotate with the offending parameter when present so callers
            // can see which input field triggered the rejection — matches the
            // `param` field on the upstream error schema.
            match parsed.error.param.as_ref().and_then(|v| match v {
                serde_json::Value::String(s) if !s.is_empty() => Some(s.as_str()),
                _ => None,
            }) {
                Some(param) => format!("{} (param: {param})", parsed.error.message),
                None => parsed.error.message,
            }
        }
        Err(_) => body.trim().to_owned(),
    }
}

/// Rewrite the [`ProviderError`] message to include the `OpenAI`-reported
/// error text, when present.
///
/// The transport layer in `provider-utils` produces messages like
/// `"HTTP 429 Too Many Requests"`. For `OpenAI` we want
/// `"OpenAI API error: rate limited (HTTP 429)"`. Non-`ApiCall` errors and
/// errors without a parseable body pass through unchanged.
pub(crate) fn rewrite_openai_error(err: ProviderError) -> ProviderError {
    if !err.is_api_call() {
        return err;
    }
    let Some(body) = err.response_body() else {
        return err;
    };
    let detail = extract_error_message(body);
    if detail.is_empty() {
        return err;
    }
    let status = err.status_code();
    let url = err.url().unwrap_or("").to_owned();
    let mut builder = ProviderError::api_call_builder(
        url,
        match status {
            Some(s) => format!("OpenAI API error: {detail} (HTTP {s})"),
            None => format!("OpenAI API error: {detail}"),
        },
    )
    .response_body(body.to_owned())
    .retryable(err.is_retryable());
    if let Some(s) = status {
        builder = builder.status_code(s);
    }
    builder.build()
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
