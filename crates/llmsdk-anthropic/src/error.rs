//! `Anthropic` error response parsing.
//!
//! Mirrors `@ai-sdk/anthropic/src/anthropic-error.ts`. Envelope shape:
//! `{ "type": "error", "error": { "type": "invalid_request_error", "message": "..." } }`.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::ProviderError;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ErrorBody {
    error: ErrorInner,
}

#[derive(Debug, Deserialize)]
struct ErrorInner {
    message: String,
    #[serde(default, rename = "type")]
    _kind: Option<String>,
}

/// Extract a one-line error message from a raw response body.
///
/// Falls back to the trimmed body when parsing fails.
pub(crate) fn extract_error_message(body: &str) -> String {
    match serde_json::from_str::<ErrorBody>(body) {
        Ok(parsed) => parsed.error.message,
        Err(_) => body.trim().to_owned(),
    }
}

/// Rewrite the [`ProviderError`] message to include the `Anthropic`-reported
/// error text.
pub(crate) fn rewrite_anthropic_error(err: ProviderError) -> ProviderError {
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
            Some(s) => format!("Anthropic API error: {detail} (HTTP {s})"),
            None => format!("Anthropic API error: {detail}"),
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
        let body =
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad input"}}"#;
        assert_eq!(extract_error_message(body), "bad input");
    }

    #[test]
    fn falls_back_to_raw_body() {
        assert_eq!(extract_error_message("oops"), "oops");
    }
}
