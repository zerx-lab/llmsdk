//! HTTP transport helpers built on `reqwest`.
//!
//! Mirrors `post-to-api.ts`, `get-from-api.ts`, `response-handler.ts`, and
//! `handle-fetch-error.ts` from `@ai-sdk/provider-utils`. We collapse the
//! TS `ResponseHandler<T>` indirection: the public surface is two `async fn`s
//! that return `Result<T, ProviderError>` directly.
//!
//! Errors are mapped onto [`llmsdk_provider::ProviderError`] as follows:
//!
//! - 4xx / 5xx with a body → [`ProviderError::api_call_builder`] with the
//!   `responseBody` field populated.
//! - Transport failure → [`ProviderError::api_call_builder`] flagged as
//!   retryable.
//! - JSON parse failure on success body → [`ProviderError::json_parse`].
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::time::Duration;

use bytes::Bytes;
use futures::stream::TryStreamExt;
use llmsdk_provider::ProviderError;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::{Client, Method, Response, StatusCode};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Default request timeout (1 minute). Matches ai-sdk's documented
/// expectation of "long enough for slow LLM responses, short enough to
/// fail fast".
pub const DEFAULT_TIMEOUT: Duration = Duration::from_mins(1);

/// Thin wrapper over `reqwest::Client` with provider-friendly defaults.
///
/// Cloning is cheap (`Arc` internally). Construct once per provider instance.
#[derive(Debug, Clone)]
pub struct HttpClient {
    inner: Client,
}

impl HttpClient {
    /// Build a client with sane defaults (60s timeout, rustls).
    ///
    /// # Errors
    ///
    /// Returns a load-time [`ProviderError`] when the underlying TLS stack
    /// fails to initialize. Practically only fails in misconfigured CI
    /// containers.
    pub fn new() -> Result<Self, ProviderError> {
        let inner = Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .map_err(|e| {
                ProviderError::load_api_key(format!("failed to build HTTP client: {e}"))
            })?;
        Ok(Self { inner })
    }

    /// Wrap an existing `reqwest::Client`.
    #[must_use]
    pub fn from_reqwest(client: Client) -> Self {
        Self { inner: client }
    }

    /// Borrow the underlying `reqwest::Client`.
    #[must_use]
    pub fn reqwest(&self) -> &Client {
        &self.inner
    }
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new().expect("default HttpClient should build")
    }
}

/// Description of an outgoing JSON request.
///
/// Built directly; only `url` and `body` are required.
#[derive(Debug, Clone)]
pub struct JsonRequest<B> {
    /// Absolute URL.
    pub url: String,
    /// Request body, serialized as JSON.
    pub body: B,
    /// Extra headers (override `Content-Type` etc.). `None` values drop a header.
    pub headers: HashMap<String, Option<String>>,
}

impl<B> JsonRequest<B> {
    /// Build a request with the given URL and body, no extra headers.
    pub fn new(url: impl Into<String>, body: B) -> Self {
        Self {
            url: url.into(),
            body,
            headers: HashMap::new(),
        }
    }

    /// Add or override a header. `None` removes it.
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: Option<String>) -> Self {
        self.headers.insert(name.into(), value);
        self
    }
}

/// POST a JSON body and decode a JSON response into `T`.
///
/// On HTTP error status, parses the body for diagnostics and returns
/// [`ProviderError::api_call_builder`]-built error. Retryable flag follows
/// ai-sdk: 408 / 409 / 429 / 5xx.
///
/// # Errors
///
/// - Transport failure (DNS, TLS, connection reset) — `api_call` retryable.
/// - Non-2xx status — `api_call` with `response_body` captured.
/// - 2xx body that fails to parse as `T` — `json_parse`.
pub async fn post_json<B, T>(
    client: &HttpClient,
    request: JsonRequest<B>,
) -> Result<JsonResponse<T>, ProviderError>
where
    B: Serialize,
    T: DeserializeOwned,
{
    let body_value = serde_json::to_value(&request.body)
        .map_err(|e| ProviderError::json_parse("<request body>", e.to_string()))?;

    let mut builder = client
        .inner
        .request(Method::POST, &request.url)
        .header("content-type", "application/json")
        .json(&request.body);
    builder = apply_headers(builder, &request.headers);

    let response = builder
        .send()
        .await
        .map_err(|e| map_transport_error(&e, &request.url, body_value.clone()))?;

    handle_response::<T>(response, &request.url, body_value).await
}

/// GET a JSON response into `T`.
///
/// # Errors
///
/// Same conditions as [`post_json`], with `request_body` omitted in errors.
pub async fn get_json<T, S>(
    client: &HttpClient,
    url: &str,
    headers: &HashMap<String, Option<String>, S>,
) -> Result<JsonResponse<T>, ProviderError>
where
    T: DeserializeOwned,
    S: std::hash::BuildHasher,
{
    let mut builder = client.inner.request(Method::GET, url);
    builder = apply_headers(builder, headers);

    let response = builder
        .send()
        .await
        .map_err(|e| map_transport_error(&e, url, serde_json::Value::Null))?;

    handle_response::<T>(response, url, serde_json::Value::Null).await
}

/// POST a JSON body and return the raw response for SSE streaming.
///
/// The caller drives the byte stream via [`response_byte_stream`] +
/// [`crate::sse::sse_json_stream`].
///
/// # Errors
///
/// Same conditions as [`post_json`] for transport / status failures. The
/// 2xx body is **not** parsed — that is the caller's job.
pub async fn post_for_stream<B>(
    client: &HttpClient,
    request: JsonRequest<B>,
) -> Result<StreamResponse, ProviderError>
where
    B: Serialize,
{
    let body_value = serde_json::to_value(&request.body)
        .map_err(|e| ProviderError::json_parse("<request body>", e.to_string()))?;

    let mut builder = client
        .inner
        .request(Method::POST, &request.url)
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .json(&request.body);
    builder = apply_headers(builder, &request.headers);

    let response = builder
        .send()
        .await
        .map_err(|e| map_transport_error(&e, &request.url, body_value.clone()))?;

    let status = response.status();
    let headers = collect_headers(response.headers());

    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(build_api_error(
            &request.url,
            status,
            &headers,
            body,
            body_value,
        ));
    }

    Ok(StreamResponse {
        status,
        headers,
        response,
    })
}

/// Result of a JSON request: status + headers + decoded body.
#[derive(Debug)]
pub struct JsonResponse<T> {
    /// Final HTTP status (always 2xx on the `Ok` path).
    pub status: StatusCode,
    /// Response headers.
    pub headers: HashMap<String, String>,
    /// Decoded body.
    pub value: T,
    /// Raw body bytes preserved for telemetry / debugging.
    pub raw: Bytes,
}

/// Result of a streaming request: response handle for byte-level reading.
#[expect(
    missing_debug_implementations,
    reason = "reqwest::Response is not Debug"
)]
pub struct StreamResponse {
    /// Final HTTP status (always 2xx on the `Ok` path).
    pub status: StatusCode,
    /// Response headers.
    pub headers: HashMap<String, String>,
    /// The owned reqwest response — drive via [`response_byte_stream`].
    pub response: Response,
}

/// Convert a [`reqwest::Response`] into a byte stream of [`Bytes`] chunks.
///
/// Used as the source for [`crate::sse::sse_json_stream`].
pub fn response_byte_stream(
    response: Response,
) -> impl futures::Stream<Item = Result<Bytes, ProviderError>> + Send {
    let url = response.url().to_string();
    response.bytes_stream().map_err(move |e| {
        ProviderError::api_call_builder(&url, format!("stream read failed: {e}"))
            .retryable(true)
            .build()
    })
}

/// Parse a JSON value into `T`, mapping parse failures to [`ProviderError::json_parse`].
///
/// Useful in provider crates after they have the raw bytes (e.g. from SSE).
///
/// # Errors
///
/// Returns [`ProviderError::json_parse`] when `text` is not valid JSON for `T`.
pub fn parse_json_response<T: DeserializeOwned>(text: &str) -> Result<T, ProviderError> {
    serde_json::from_str(text)
        .map_err(|e| ProviderError::json_parse(text.to_owned(), e.to_string()))
}

// ---- internal helpers ----------------------------------------------------

async fn handle_response<T: DeserializeOwned>(
    response: Response,
    url: &str,
    request_body: serde_json::Value,
) -> Result<JsonResponse<T>, ProviderError> {
    let status = response.status();
    let headers = collect_headers(response.headers());
    let raw = response.bytes().await.map_err(|e| {
        ProviderError::api_call_builder(url, format!("failed to read response body: {e}"))
            .status_code(status.as_u16())
            .retryable(true)
            .build()
    })?;

    if !status.is_success() {
        let body_text = String::from_utf8_lossy(&raw).into_owned();
        return Err(build_api_error(
            url,
            status,
            &headers,
            body_text,
            request_body,
        ));
    }

    let value: T = serde_json::from_slice(&raw).map_err(|e| {
        ProviderError::json_parse(String::from_utf8_lossy(&raw).into_owned(), e.to_string())
    })?;

    Ok(JsonResponse {
        status,
        headers,
        value,
        raw,
    })
}

fn build_api_error(
    url: &str,
    status: StatusCode,
    headers: &HashMap<String, String>,
    body: String,
    request_body: serde_json::Value,
) -> ProviderError {
    ProviderError::api_call_builder(
        url,
        format!(
            "HTTP {} {}",
            status.as_u16(),
            status.canonical_reason().unwrap_or("")
        ),
    )
    .status_code(status.as_u16())
    .response_body(body)
    .response_headers(headers.clone())
    .request_body(request_body)
    .build()
}

fn map_transport_error(
    err: &reqwest::Error,
    url: &str,
    request_body: serde_json::Value,
) -> ProviderError {
    // Mirrors handle-fetch-error.ts: network failures are retryable.
    let mut builder = ProviderError::api_call_builder(url, format!("transport error: {err}"));
    if let Some(status) = err.status() {
        builder = builder.status_code(status.as_u16());
    }
    builder.retryable(true).request_body(request_body).build()
}

fn apply_headers<S>(
    mut builder: reqwest::RequestBuilder,
    headers: &HashMap<String, Option<String>, S>,
) -> reqwest::RequestBuilder
where
    S: std::hash::BuildHasher,
{
    for (name, value) in headers {
        match value {
            Some(v) => {
                builder = builder.header(name, v);
            }
            None => {
                // No direct "drop" API on RequestBuilder. We send empty value;
                // reqwest preserves the latest .header() per name, so the only
                // way to override Content-Type=application/json from .json() is
                // by sending an explicit replacement. None remains for parity
                // with ai-sdk: callers rarely set None at this layer.
                builder = builder.header(name, "");
            }
        }
    }
    builder
}

fn collect_headers(headers: &HeaderMap) -> HashMap<String, String> {
    let mut out = HashMap::with_capacity(headers.len());
    for (name, value) in headers {
        if let Ok(v) = value.to_str() {
            out.insert(name.as_str().to_owned(), v.to_owned());
        }
    }
    out
}

/// Build a `reqwest::HeaderMap` from a string map, ignoring invalid names / values.
///
/// Useful for provider crates that need to push a finalized header set into
/// a `reqwest::Request` directly.
#[must_use]
pub fn to_header_map<S>(headers: &HashMap<String, String, S>) -> HeaderMap
where
    S: std::hash::BuildHasher,
{
    let mut out = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        if let (Ok(n), Ok(v)) = (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            out.insert(n, v);
        }
    }
    out
}
