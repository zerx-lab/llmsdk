//! Google Generative AI error envelope decoding.
//!
//! Mirrors `@ai-sdk/google/src/google-error.ts`. Google returns errors as
//! `{ "error": { "code", "message", "status" } }`. We re-extract the inner
//! `error.message` to give a clean one-line error message while preserving
//! the original envelope in `response_body`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::ProviderError;
use serde::{Deserialize, Serialize};

/// Decoded Google error envelope (kept public for downstream typing).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoogleErrorBody {
    /// Inner error object.
    pub error: GoogleErrorInner,
}

/// Inner error fields.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoogleErrorInner {
    /// Optional numeric code.
    #[serde(default)]
    pub code: Option<i64>,
    /// Human-readable message.
    pub message: String,
    /// Provider status string, e.g. `"INVALID_ARGUMENT"`.
    #[serde(default)]
    pub status: Option<String>,
}

/// Extract a one-line error message from a raw response body.
#[must_use]
pub(crate) fn extract_error_message(body: &str) -> String {
    match serde_json::from_str::<GoogleErrorBody>(body) {
        Ok(parsed) => parsed.error.message,
        Err(_) => body.trim().to_owned(),
    }
}

/// Rewrite a generic API-call error's user-facing message using the
/// `error.message` field from a Google error envelope when present.
#[must_use]
pub(crate) fn rewrite_google_error(err: ProviderError) -> ProviderError {
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
            Some(s) => format!("Google API error: {detail} (HTTP {s})"),
            None => format!("Google API error: {detail}"),
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
    fn parses_envelope() {
        let body = r#"{"error":{"code":400,"message":"bad input","status":"INVALID_ARGUMENT"}}"#;
        assert_eq!(extract_error_message(body), "bad input");
    }

    #[test]
    fn falls_back_to_raw_body() {
        assert_eq!(extract_error_message("oops"), "oops");
    }

    #[test]
    fn rewrite_uses_inner_message() {
        let err = ProviderError::api_call_builder("https://x/y", "HTTP 400")
            .status_code(400)
            .response_body(
                r#"{"error":{"code":400,"message":"Invalid model","status":"INVALID_ARGUMENT"}}"#,
            )
            .build();
        let rewritten = rewrite_google_error(err);
        assert!(format!("{rewritten}").contains("Invalid model"));
    }
}
