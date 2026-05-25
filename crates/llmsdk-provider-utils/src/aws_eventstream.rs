//! AWS Event Stream binary-frame decoding (feature `aws-event-stream`).
//!
//! Mirrors `amazon-bedrock-event-stream-decoder.ts` from ai-sdk: take a raw
//! byte stream coming back from a `text/event-stream`-style AWS endpoint
//! (Bedrock `InvokeModelWithResponseStream`, `SageMaker` SSM, etc.) and yield
//! one decoded `EventStreamMessage` per binary frame.
//!
//! Framing + CRC32 validation is done by [`aws-smithy-eventstream`]. This
//! module:
//!
//! - hides the smithy `Message` / `HeaderValue` types behind a minimal,
//!   stable [`EventStreamMessage`] / [`EventStreamValue`] surface so future
//!   provider crates don't pin themselves to a specific smithy version,
//! - feeds a streaming [`futures::Stream`] through
//!   [`MessageFrameDecoder`](aws_smithy_eventstream::frame::MessageFrameDecoder),
//!   accumulating bytes until a full frame is available,
//! - maps every smithy error onto [`ProviderError`].
//!
//! [`aws-smithy-eventstream`]: https://crates.io/crates/aws-smithy-eventstream
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::pin::Pin;

use aws_smithy_eventstream::frame::{DecodedFrame, MessageFrameDecoder};
use aws_smithy_types::event_stream::HeaderValue as SmithyHeaderValue;
use bytes::{Bytes, BytesMut};
use futures::Stream;
use futures::stream::StreamExt;
use llmsdk_provider::ProviderError;

/// One header value attached to an [`EventStreamMessage`].
///
/// Maps every scalar variant from
/// [`aws_smithy_types::event_stream::HeaderValue`] onto a stable, provider-
/// friendly enum. We collapse `Timestamp` to an `f64` epoch-seconds value so
/// callers don't need to take a transitive dep on `aws_smithy_types`.
///
/// # Examples
///
/// ```
/// use llmsdk_provider_utils::aws_eventstream::EventStreamValue;
///
/// let v = EventStreamValue::String("contentBlockDelta".into());
/// assert_eq!(v.as_str(), Some("contentBlockDelta"));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum EventStreamValue {
    /// UTF-8 string header (`:event-type`, `:message-type`, ...).
    String(String),
    /// Boolean header.
    Bool(bool),
    /// 8-bit signed integer.
    Byte(i8),
    /// 16-bit signed integer.
    Int16(i16),
    /// 32-bit signed integer.
    Int32(i32),
    /// 64-bit signed integer ("Long" in the AWS wire spec).
    Long(i64),
    /// Raw binary header.
    Binary(Bytes),
    /// Epoch seconds (smithy stores sub-second precision; we keep both via `f64`).
    Timestamp(f64),
    /// 128-bit UUID, opaque to llmsdk.
    Uuid(u128),
}

impl EventStreamValue {
    /// Borrow as a string slice when this header is a `String`.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// One decoded event-stream message.
///
/// The smithy wire format is `:message-type` + `:event-type` + payload; the
/// payload itself is service-specific JSON (Bedrock chat deltas etc.). The
/// caller is expected to UTF-8 decode `payload` and JSON-parse it.
#[derive(Debug, Clone)]
pub struct EventStreamMessage {
    /// Headers keyed by name (case preserved).
    pub headers: HashMap<String, EventStreamValue>,
    /// Raw payload bytes — service-specific encoding (usually UTF-8 JSON).
    pub payload: Bytes,
}

impl EventStreamMessage {
    /// Convenience: borrow the `:event-type` header as `&str` if present.
    #[must_use]
    pub fn event_type(&self) -> Option<&str> {
        self.headers
            .get(":event-type")
            .and_then(EventStreamValue::as_str)
    }

    /// Convenience: borrow the `:message-type` header as `&str` if present.
    #[must_use]
    pub fn message_type(&self) -> Option<&str> {
        self.headers
            .get(":message-type")
            .and_then(EventStreamValue::as_str)
    }
}

/// Decode an AWS binary-frame event stream into a stream of
/// [`EventStreamMessage`] values.
///
/// The input is the raw byte stream coming back from
/// [`crate::http::response_byte_stream`] (or any other source). Bytes are
/// accumulated into an internal buffer until
/// [`MessageFrameDecoder::decode_frame`] reports a complete frame, then the
/// frame is parsed and yielded. CRC32 validation is performed by the smithy
/// decoder; mismatches surface as
/// [`ProviderError::api_call_builder`]-built errors with the frame buffer
/// preserved as `response_body`.
///
/// # Examples
///
/// ```no_run
/// # async fn doc() -> Result<(), llmsdk_provider::ProviderError> {
/// use futures::StreamExt;
/// use llmsdk_provider_utils::aws_eventstream::decode_event_stream;
/// use llmsdk_provider_utils::http::{HttpClient, JsonRequest, post_for_stream, response_byte_stream};
///
/// let client = HttpClient::new()?;
/// let req = JsonRequest::new("https://bedrock-runtime.us-east-1.amazonaws.com/foo", serde_json::json!({}));
/// let stream = post_for_stream(&client, req).await?;
/// let bytes = response_byte_stream(stream.response);
/// let mut events = decode_event_stream(bytes);
/// while let Some(msg) = events.next().await {
///     let msg = msg?;
///     println!("event {:?} payload bytes {}", msg.event_type(), msg.payload.len());
/// }
/// # Ok(()) }
/// ```
pub fn decode_event_stream<S>(
    bytes: S,
) -> Pin<Box<dyn Stream<Item = Result<EventStreamMessage, ProviderError>> + Send>>
where
    S: Stream<Item = Result<Bytes, ProviderError>> + Send + 'static,
{
    struct State<S> {
        upstream: Pin<Box<S>>,
        decoder: MessageFrameDecoder,
        buffer: BytesMut,
        eof: bool,
    }

    let state = State {
        upstream: Box::pin(bytes),
        decoder: MessageFrameDecoder::new(),
        buffer: BytesMut::new(),
        eof: false,
    };

    let stream = futures::stream::unfold(state, |mut state| async move {
        loop {
            // 1) Try to decode from the current buffer. `decode_frame` is
            //    stateful: it advances `state.buffer` (which impls `Buf`)
            //    only when it actually reads bytes. Passing a cloned cursor
            //    would cause the prelude to be re-read on every call, so we
            //    operate on the live buffer.
            let decode_result = state.decoder.decode_frame(&mut state.buffer);
            match decode_result {
                Ok(DecodedFrame::Complete(message)) => {
                    let mapped = convert_message(&message);
                    return Some((Ok(mapped), state));
                }
                Ok(DecodedFrame::Incomplete) => {
                    if state.eof {
                        // No more bytes coming and no complete frame — end.
                        if !state.buffer.is_empty() {
                            let err = ProviderError::api_call_builder(
                                "<event-stream>",
                                format!(
                                    "AWS event-stream ended mid-frame with {} bytes buffered",
                                    state.buffer.len()
                                ),
                            )
                            .build();
                            return Some((Err(err), state));
                        }
                        return None;
                    }
                    // fall through to pull more bytes
                }
                Err(e) => {
                    let err = ProviderError::api_call_builder(
                        "<event-stream>",
                        format!("AWS event-stream frame decode failed: {e}"),
                    )
                    .response_body(format!("{:02x?}", state.buffer.as_ref()))
                    .build();
                    // Drop the buffer so we don't loop on the same garbage.
                    state.buffer.clear();
                    return Some((Err(err), state));
                }
            }

            // 2) Pull the next chunk.
            match state.upstream.next().await {
                Some(Ok(chunk)) => state.buffer.extend_from_slice(&chunk),
                Some(Err(e)) => return Some((Err(e), state)),
                None => state.eof = true,
            }
        }
    });

    Box::pin(stream)
}

fn convert_message(message: &aws_smithy_types::event_stream::Message) -> EventStreamMessage {
    let mut headers = HashMap::with_capacity(message.headers().len());
    for header in message.headers() {
        let name = header.name().as_str().to_owned();
        let value = convert_value(header.value());
        headers.insert(name, value);
    }
    EventStreamMessage {
        headers,
        payload: message.payload().clone(),
    }
}

fn convert_value(value: &SmithyHeaderValue) -> EventStreamValue {
    match value {
        SmithyHeaderValue::Bool(b) => EventStreamValue::Bool(*b),
        SmithyHeaderValue::Byte(v) => EventStreamValue::Byte(*v),
        SmithyHeaderValue::Int16(v) => EventStreamValue::Int16(*v),
        SmithyHeaderValue::Int32(v) => EventStreamValue::Int32(*v),
        SmithyHeaderValue::Int64(v) => EventStreamValue::Long(*v),
        SmithyHeaderValue::ByteArray(b) => EventStreamValue::Binary(b.clone()),
        SmithyHeaderValue::String(s) => EventStreamValue::String(s.as_str().to_owned()),
        SmithyHeaderValue::Timestamp(ts) => EventStreamValue::Timestamp(ts.as_secs_f64()),
        SmithyHeaderValue::Uuid(u) => EventStreamValue::Uuid(*u),
        // SmithyHeaderValue is #[non_exhaustive]; future scalars degrade to
        // string Debug to preserve forward compatibility.
        other => EventStreamValue::String(format!("{other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use aws_smithy_eventstream::frame::write_message_to;
    use aws_smithy_types::event_stream::{
        Header as SmithyHeader, HeaderValue as SmithyValue, Message as SmithyMessage,
    };
    use futures::stream;

    fn make_frame(event_type: &str, payload: &[u8]) -> Vec<u8> {
        // StrBytes requires owned data; clone the &str into owned Strings up-front.
        let evt: String = event_type.to_owned();
        let message = SmithyMessage::new_from_parts(
            vec![
                SmithyHeader::new(
                    ":message-type",
                    SmithyValue::String(String::from("event").into()),
                ),
                SmithyHeader::new(":event-type", SmithyValue::String(evt.into())),
                SmithyHeader::new(
                    ":content-type",
                    SmithyValue::String(String::from("application/json").into()),
                ),
            ],
            Bytes::copy_from_slice(payload),
        );
        let mut buf = Vec::new();
        write_message_to(&message, &mut buf).expect("encode test frame");
        buf
    }

    #[tokio::test]
    async fn decodes_single_known_fixture() {
        let frame = make_frame("contentBlockDelta", br#"{"delta":"hi"}"#);
        let upstream = stream::iter(vec![Ok::<_, ProviderError>(Bytes::from(frame))]);
        let mut events = decode_event_stream(upstream);

        let msg = events.next().await.expect("at least one event").unwrap();
        assert_eq!(msg.event_type(), Some("contentBlockDelta"));
        assert_eq!(msg.message_type(), Some("event"));
        assert_eq!(&msg.payload[..], br#"{"delta":"hi"}"#);
        // String/Bool round-trip
        assert!(matches!(
            msg.headers.get(":content-type"),
            Some(EventStreamValue::String(s)) if s == "application/json"
        ));

        // Stream should end cleanly.
        assert!(events.next().await.is_none());
    }

    #[tokio::test]
    async fn decodes_multiple_frames_split_across_chunks() {
        let f1 = make_frame("first", b"{}");
        let f2 = make_frame("second", b"{}");
        let f3 = make_frame("third", b"{}");
        let mut combined = Vec::new();
        combined.extend_from_slice(&f1);
        combined.extend_from_slice(&f2);
        combined.extend_from_slice(&f3);

        // Split into 7-byte chunks to exercise the buffer-accumulation path.
        let chunks: Vec<Result<Bytes, ProviderError>> = combined
            .chunks(7)
            .map(|c| Ok(Bytes::copy_from_slice(c)))
            .collect();
        let mut events = decode_event_stream(stream::iter(chunks));

        let mut seen = Vec::new();
        while let Some(ev) = events.next().await {
            seen.push(ev.unwrap().event_type().unwrap_or_default().to_owned());
        }
        assert_eq!(seen, vec!["first", "second", "third"]);
    }

    #[tokio::test]
    async fn corrupted_frame_yields_error_then_terminates() {
        // Build a valid frame, then flip one byte in the payload region
        // to break the message CRC.
        let mut frame = make_frame("delta", b"payload");
        // Last 4 bytes are message CRC32; flip one of them to force a
        // checksum mismatch.
        let last = frame.len() - 1;
        frame[last] ^= 0xff;
        let upstream = stream::iter(vec![Ok::<_, ProviderError>(Bytes::from(frame))]);
        let mut events = decode_event_stream(upstream);

        let first = events.next().await.expect("error event");
        assert!(first.is_err(), "expected decode error, got {first:?}");
        // After surfacing the error the buffer is cleared, so the stream
        // ends cleanly.
        assert!(events.next().await.is_none());
    }
}
