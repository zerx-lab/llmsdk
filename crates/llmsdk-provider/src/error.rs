//! Unified error type for all provider operations.
//!
//! Mirrors `@ai-sdk/provider`'s `errors/*` directory. ai-sdk uses a class per
//! error (`APICallError`, `InvalidPromptError`, ...); we collapse these into
//! a single canonical [`ProviderError`] struct backed by a private
//! [`ErrorKind`], following Microsoft's [M-ERRORS-CANONICAL-STRUCTS].
//!
//! Inspection is done via the `is_*()` helpers, never by matching `ErrorKind`
//! directly (it is private to keep the API additive).
// Rust guideline compliant 2026-02-21

use std::backtrace::Backtrace;
use std::collections::HashMap;
use std::fmt;

use serde_json::Value as JsonValue;

/// Convenience alias used across the crate.
pub type Result<T> = std::result::Result<T, ProviderError>;

/// Boxed source error preserved as the [`std::error::Error::source`] chain.
type Source = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Unified error returned by every provider operation.
///
/// Use the `is_*()` helpers to branch on cause. The underlying variant is
/// intentionally hidden so we can add new failure modes without a breaking
/// change.
///
/// Internally boxed so `Result<T, ProviderError>` stays one machine word
/// in the `Err` slot.
///
/// # Examples
///
/// ```
/// use llmsdk_provider::ProviderError;
///
/// let err = ProviderError::no_such_model("gpt-foo", "languageModel");
/// assert!(err.is_no_such_model());
/// assert_eq!(err.model_id(), Some("gpt-foo"));
/// ```
pub struct ProviderError {
    inner: Box<ErrorInner>,
}

struct ErrorInner {
    kind: ErrorKind,
    backtrace: Backtrace,
    source: Option<Source>,
}

/// Private enum carrying per-variant data.
///
/// Kept `pub(crate)` so provider crates can construct via the inherent
/// helpers below, never by naming the variant. Adding a new variant is not
/// a breaking change.
#[derive(Debug)]
#[expect(
    dead_code,
    reason = "value/text retained for Debug output and future accessors"
)]
pub(crate) enum ErrorKind {
    ApiCall(ApiCallData),
    InvalidArgument {
        argument: String,
        message: String,
    },
    InvalidPrompt {
        message: String,
    },
    TypeValidation {
        path: String,
        value: JsonValue,
        message: String,
    },
    JsonParse {
        text: String,
        message: String,
    },
    EmptyResponseBody,
    NoContentGenerated,
    NoSuchModel {
        model_id: String,
        model_type: String,
    },
    Unsupported {
        functionality: String,
    },
    LoadApiKey {
        message: String,
    },
    TooManyEmbeddingValues {
        max: usize,
        actual: usize,
    },
}

/// Detail payload for an HTTP-level API call failure.
#[derive(Debug, Default)]
pub(crate) struct ApiCallData {
    pub url: String,
    pub message: String,
    pub status_code: Option<u16>,
    pub response_headers: Option<HashMap<String, String>>,
    pub response_body: Option<String>,
    pub request_body: Option<JsonValue>,
    pub is_retryable: bool,
}

impl ProviderError {
    // ---- constructors -------------------------------------------------

    /// Build an HTTP API call error.
    ///
    /// `is_retryable` is `true` by default for 408 / 409 / 429 / 5xx,
    /// matching ai-sdk's `APICallError` defaults. Use [`Self::api_call_builder`]
    /// for full control.
    pub fn api_call(url: impl Into<String>, message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::ApiCall(ApiCallData {
            url: url.into(),
            message: message.into(),
            ..ApiCallData::default()
        }))
    }

    /// Open a builder for an API call error with optional fields.
    pub fn api_call_builder(
        url: impl Into<String>,
        message: impl Into<String>,
    ) -> ApiCallErrorBuilder {
        ApiCallErrorBuilder {
            data: ApiCallData {
                url: url.into(),
                message: message.into(),
                ..ApiCallData::default()
            },
            source: None,
            retryable_override: None,
        }
    }

    /// Build an invalid-argument error for a public parameter.
    pub fn invalid_argument(argument: impl Into<String>, message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::InvalidArgument {
            argument: argument.into(),
            message: message.into(),
        })
    }

    /// Build an invalid-prompt error.
    pub fn invalid_prompt(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::InvalidPrompt {
            message: message.into(),
        })
    }

    /// Build a type-validation error for a JSON path.
    pub fn type_validation(
        path: impl Into<String>,
        value: JsonValue,
        message: impl Into<String>,
    ) -> Self {
        Self::from_kind(ErrorKind::TypeValidation {
            path: path.into(),
            value,
            message: message.into(),
        })
    }

    /// Build a JSON parse error.
    pub fn json_parse(text: impl Into<String>, message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::JsonParse {
            text: text.into(),
            message: message.into(),
        })
    }

    /// The provider returned an empty body where content was expected.
    #[must_use]
    pub fn empty_response_body() -> Self {
        Self::from_kind(ErrorKind::EmptyResponseBody)
    }

    /// The provider returned successfully but produced no usable content.
    #[must_use]
    pub fn no_content_generated() -> Self {
        Self::from_kind(ErrorKind::NoContentGenerated)
    }

    /// No such model id for the given model type (`languageModel`, ...).
    pub fn no_such_model(model_id: impl Into<String>, model_type: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::NoSuchModel {
            model_id: model_id.into(),
            model_type: model_type.into(),
        })
    }

    /// The provider does not support the requested functionality.
    pub fn unsupported(functionality: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::Unsupported {
            functionality: functionality.into(),
        })
    }

    /// Could not load an API key from the environment / config.
    pub fn load_api_key(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::LoadApiKey {
            message: message.into(),
        })
    }

    /// Too many embedding inputs passed in a single call.
    #[must_use]
    pub fn too_many_embedding_values(max: usize, actual: usize) -> Self {
        Self::from_kind(ErrorKind::TooManyEmbeddingValues { max, actual })
    }

    // ---- inspection ---------------------------------------------------

    /// True for API-level HTTP failures.
    #[must_use]
    pub fn is_api_call(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::ApiCall(_))
    }

    /// True when the failure should be retried.
    ///
    /// Currently only [`Self::is_api_call`] errors carry retry information;
    /// all others return `false`.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        matches!(&self.inner.kind, ErrorKind::ApiCall(d) if d.is_retryable)
    }

    /// True when the error reports an unknown model id.
    #[must_use]
    pub fn is_no_such_model(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::NoSuchModel { .. })
    }

    /// True when the error reports unsupported functionality.
    #[must_use]
    pub fn is_unsupported(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::Unsupported { .. })
    }

    /// HTTP status code when [`Self::is_api_call`].
    #[must_use]
    pub fn status_code(&self) -> Option<u16> {
        match &self.inner.kind {
            ErrorKind::ApiCall(d) => d.status_code,
            _ => None,
        }
    }

    /// Captured response body when [`Self::is_api_call`].
    ///
    /// Returned by HTTP transports that read the full body before raising
    /// the error; otherwise `None`.
    #[must_use]
    pub fn response_body(&self) -> Option<&str> {
        match &self.inner.kind {
            ErrorKind::ApiCall(d) => d.response_body.as_deref(),
            _ => None,
        }
    }

    /// Request URL when [`Self::is_api_call`].
    #[must_use]
    pub fn url(&self) -> Option<&str> {
        match &self.inner.kind {
            ErrorKind::ApiCall(d) => Some(&d.url),
            _ => None,
        }
    }

    /// Model id when [`Self::is_no_such_model`].
    #[must_use]
    pub fn model_id(&self) -> Option<&str> {
        match &self.inner.kind {
            ErrorKind::NoSuchModel { model_id, .. } => Some(model_id),
            _ => None,
        }
    }

    /// Captured backtrace (empty unless `RUST_BACKTRACE` is set).
    pub fn backtrace(&self) -> &Backtrace {
        &self.inner.backtrace
    }

    // ---- internal -----------------------------------------------------

    fn from_kind(kind: ErrorKind) -> Self {
        Self {
            inner: Box::new(ErrorInner {
                kind,
                backtrace: Backtrace::capture(),
                source: None,
            }),
        }
    }

    pub(crate) fn with_source(mut self, source: Source) -> Self {
        self.inner.source = Some(source);
        self
    }
}

/// Builder for API call errors with optional fields.
///
/// Returned by [`ProviderError::api_call_builder`]. Methods are chainable;
/// call [`Self::build`] to finalize.
#[derive(Debug)]
pub struct ApiCallErrorBuilder {
    data: ApiCallData,
    source: Option<Source>,
    retryable_override: Option<bool>,
}

impl ApiCallErrorBuilder {
    /// Set the HTTP status code.
    #[must_use]
    pub fn status_code(mut self, code: u16) -> Self {
        self.data.status_code = Some(code);
        self
    }

    /// Set the response body.
    #[must_use]
    pub fn response_body(mut self, body: impl Into<String>) -> Self {
        self.data.response_body = Some(body.into());
        self
    }

    /// Set the response headers.
    #[must_use]
    pub fn response_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.data.response_headers = Some(headers);
        self
    }

    /// Set the request body that was sent (for telemetry).
    #[must_use]
    pub fn request_body(mut self, body: JsonValue) -> Self {
        self.data.request_body = Some(body);
        self
    }

    /// Override the auto-derived retry flag.
    ///
    /// Without this call, retryable defaults to `true` for status 408 / 409 /
    /// 429 / 5xx (matching `@ai-sdk/provider`).
    #[must_use]
    pub fn retryable(mut self, retryable: bool) -> Self {
        self.retryable_override = Some(retryable);
        self
    }

    /// Attach an upstream cause.
    #[must_use]
    pub fn source<E>(mut self, err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        self.source = Some(Box::new(err));
        self
    }

    /// Finalize the error.
    #[must_use]
    pub fn build(mut self) -> ProviderError {
        // Matches @ai-sdk/provider's defaults: retry 408/409/429 + 5xx.
        let derived = matches!(self.data.status_code, Some(408 | 409 | 429 | 500..));
        self.data.is_retryable = self.retryable_override.unwrap_or(derived);
        let err = ProviderError::from_kind(ErrorKind::ApiCall(self.data));
        if let Some(src) = self.source {
            err.with_source(src)
        } else {
            err
        }
    }
}

// ---- trait impls ------------------------------------------------------

impl fmt::Debug for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderError")
            .field("kind", &self.inner.kind)
            .field("source", &self.inner.source)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner.kind {
            ErrorKind::ApiCall(d) => {
                write!(f, "api call to {} failed: {}", d.url, d.message)?;
                if let Some(code) = d.status_code {
                    write!(f, " (status {code})")?;
                }
                Ok(())
            }
            ErrorKind::InvalidArgument { argument, message } => {
                write!(f, "invalid argument `{argument}`: {message}")
            }
            ErrorKind::InvalidPrompt { message } => write!(f, "invalid prompt: {message}"),
            ErrorKind::TypeValidation { path, message, .. } => {
                write!(f, "type validation failed at `{path}`: {message}")
            }
            ErrorKind::JsonParse { message, .. } => write!(f, "json parse error: {message}"),
            ErrorKind::EmptyResponseBody => f.write_str("empty response body"),
            ErrorKind::NoContentGenerated => f.write_str("no content generated"),
            ErrorKind::NoSuchModel {
                model_id,
                model_type,
            } => {
                write!(f, "no such {model_type}: `{model_id}`")
            }
            ErrorKind::Unsupported { functionality } => {
                write!(f, "unsupported functionality: {functionality}")
            }
            ErrorKind::LoadApiKey { message } => write!(f, "could not load api key: {message}"),
            ErrorKind::TooManyEmbeddingValues { max, actual } => {
                write!(f, "too many embedding values: max {max}, got {actual}")
            }
        }
    }
}

impl std::error::Error for ProviderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.inner
            .source
            .as_deref()
            .map(|e| e as &(dyn std::error::Error + 'static))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers_branch_correctly() {
        let e = ProviderError::no_such_model("gpt-foo", "languageModel");
        assert!(e.is_no_such_model());
        assert_eq!(e.model_id(), Some("gpt-foo"));
        assert!(!e.is_retryable());
    }

    #[test]
    fn api_call_builder_auto_retryable() {
        let e = ProviderError::api_call_builder("https://api.test", "boom")
            .status_code(503)
            .build();
        assert!(e.is_api_call());
        assert!(e.is_retryable());
        assert_eq!(e.status_code(), Some(503));
    }

    #[test]
    fn api_call_builder_explicit_non_retryable() {
        let e = ProviderError::api_call_builder("https://api.test", "boom")
            .status_code(500)
            .retryable(false)
            .build();
        assert!(!e.is_retryable());
    }

    #[test]
    fn display_format_stable() {
        let e = ProviderError::invalid_argument("temperature", "must be >= 0");
        assert_eq!(
            format!("{e}"),
            "invalid argument `temperature`: must be >= 0"
        );
    }
}
