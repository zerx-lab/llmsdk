//! Server-sent-events parsing for streaming endpoints.
//!
//! Mirrors `parse-json-event-stream.ts`. The TS version chains
//! `EventSourceParserStream` + a JSON transform; we expose the same shape
//! with `eventsource-stream` over a byte stream.
//!
//! Behavior:
//! - The literal payload `[DONE]` (`OpenAI` / compatible) is filtered out.
//! - Empty / comment / retry-only frames are dropped.
//! - Each remaining frame is JSON-parsed into `T`.
//! - Parse errors become [`SseEvent::ParseError`] — the stream stays alive
//!   so the caller can surface them as `StreamPart::Error` if desired.
// Rust guideline compliant 2026-02-21

use std::pin::Pin;

use bytes::Bytes;
use eventsource_stream::Eventsource;
use futures::Stream;
use futures::stream::StreamExt;
use llmsdk_provider::ProviderError;
use serde::de::DeserializeOwned;

/// One decoded SSE event.
///
/// The stream surfaces in-band parse errors as [`Self::ParseError`] rather
/// than collapsing them into a transport failure; this matches ai-sdk's
/// `ParseResult` shape and lets providers translate them into
/// `StreamPart::Error` while keeping the stream alive.
#[derive(Debug)]
pub enum SseEvent<T> {
    /// A successfully decoded JSON payload.
    Data(T),
    /// A frame that arrived but failed JSON decoding.
    ParseError {
        /// Raw event data that failed to parse.
        raw: String,
        /// `serde_json` error message.
        message: String,
    },
}

/// Marker payload that ends an OpenAI-style SSE stream.
const DONE_MARKER: &str = "[DONE]";

/// Convert a byte stream into a stream of JSON events.
///
/// The input stream typically comes from
/// [`crate::http::response_byte_stream`] applied to a 2xx
/// [`crate::http::StreamResponse`].
///
/// # Examples
///
/// ```no_run
/// # async fn doc() -> Result<(), llmsdk_provider::ProviderError> {
/// use llmsdk_provider_utils::http::{HttpClient, JsonRequest, post_for_stream, response_byte_stream};
/// use llmsdk_provider_utils::sse::{SseEvent, sse_json_stream};
/// use futures::StreamExt;
/// use serde::Deserialize;
///
/// #[derive(Debug, Deserialize)]
/// struct Chunk { delta: String }
///
/// let client = HttpClient::new()?;
/// let req = JsonRequest::new("https://api.example.com/v1/stream", serde_json::json!({}));
/// let stream = post_for_stream(&client, req).await?;
/// let bytes = response_byte_stream(stream.response);
/// let mut events = sse_json_stream::<Chunk>(bytes);
/// while let Some(ev) = events.next().await {
///     match ev? {
///         SseEvent::Data(chunk) => println!("{chunk:?}"),
///         SseEvent::ParseError { raw, message } => eprintln!("parse error {message}: {raw}"),
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub fn sse_json_stream<T>(
    bytes: impl Stream<Item = Result<Bytes, ProviderError>> + Send + 'static,
) -> Pin<Box<dyn Stream<Item = Result<SseEvent<T>, ProviderError>> + Send>>
where
    T: DeserializeOwned + Send + 'static,
{
    let events = bytes
        .map(|chunk| chunk.map_err(SseStreamError::Provider))
        .eventsource()
        .filter_map(|event| async move {
            match event {
                Ok(msg) => {
                    let data = msg.data;
                    if data.is_empty() || data == DONE_MARKER {
                        None
                    } else {
                        Some(Ok::<_, ProviderError>(
                            match serde_json::from_str::<T>(&data) {
                                Ok(value) => SseEvent::Data(value),
                                Err(e) => SseEvent::ParseError {
                                    raw: data,
                                    message: e.to_string(),
                                },
                            },
                        ))
                    }
                }
                Err(eventsource_stream::EventStreamError::Transport(SseStreamError::Provider(
                    e,
                ))) => Some(Err(e)),
                Err(eventsource_stream::EventStreamError::Parser(e)) => Some(Err(
                    ProviderError::json_parse(String::new(), format!("SSE parse error: {e}")),
                )),
                Err(eventsource_stream::EventStreamError::Utf8(e)) => Some(Err(
                    ProviderError::json_parse(String::new(), format!("SSE UTF-8 error: {e}")),
                )),
            }
        });
    Box::pin(events)
}

/// Internal error newtype to plug `ProviderError` into `eventsource-stream`.
#[derive(Debug)]
enum SseStreamError {
    Provider(ProviderError),
}

impl std::fmt::Display for SseStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provider(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SseStreamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(e) => Some(e),
        }
    }
}
