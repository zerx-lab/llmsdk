//! Integration tests for SSE parsing.
//!
//! Uses synthetic byte streams; no HTTP involved.
// Rust guideline compliant 2026-02-21

use bytes::Bytes;
use futures::stream::{self, StreamExt};
use llmsdk_provider::ProviderError;
use llmsdk_provider_utils::sse::{SseEvent, sse_json_stream};
use serde::Deserialize;

#[derive(Debug, Deserialize, PartialEq)]
struct Chunk {
    delta: String,
}

fn frames(parts: &[&str]) -> impl futures::Stream<Item = Result<Bytes, ProviderError>> + Send {
    let chunks: Vec<Result<Bytes, ProviderError>> = parts
        .iter()
        .map(|s| Ok(Bytes::copy_from_slice(s.as_bytes())))
        .collect();
    stream::iter(chunks)
}

#[tokio::test]
async fn parses_happy_frames() {
    let stream = frames(&[
        "data: {\"delta\":\"hello\"}\n\n",
        "data: {\"delta\":\" world\"}\n\n",
        "data: [DONE]\n\n",
    ]);
    let mut events = sse_json_stream::<Chunk>(stream);

    let first = events.next().await.unwrap().unwrap();
    let second = events.next().await.unwrap().unwrap();
    assert!(events.next().await.is_none(), "[DONE] should terminate");

    match (first, second) {
        (SseEvent::Data(a), SseEvent::Data(b)) => {
            assert_eq!(a.delta, "hello");
            assert_eq!(b.delta, " world");
        }
        other => panic!("unexpected events: {other:?}"),
    }
}

#[tokio::test]
async fn ignores_empty_and_comment_frames() {
    let stream = frames(&[
        ": keepalive\n\n",
        "\n",
        "data: {\"delta\":\"x\"}\n\n",
        "data: [DONE]\n\n",
    ]);
    let mut events = sse_json_stream::<Chunk>(stream);
    let evt = events.next().await.unwrap().unwrap();
    assert!(matches!(evt, SseEvent::Data(Chunk { delta }) if delta == "x"));
    assert!(events.next().await.is_none());
}

#[tokio::test]
async fn parse_error_surfaces_inline() {
    let stream = frames(&[
        "data: not-json\n\n",
        "data: {\"delta\":\"ok\"}\n\n",
        "data: [DONE]\n\n",
    ]);
    let mut events = sse_json_stream::<Chunk>(stream);

    let bad = events.next().await.unwrap().unwrap();
    let good = events.next().await.unwrap().unwrap();
    assert!(events.next().await.is_none());

    match bad {
        SseEvent::ParseError { raw, message } => {
            assert_eq!(raw, "not-json");
            assert!(!message.is_empty());
        }
        SseEvent::Data(_) => panic!("expected ParseError"),
    }
    match good {
        SseEvent::Data(c) => assert_eq!(c.delta, "ok"),
        SseEvent::ParseError { .. } => panic!("expected Data"),
    }
}

#[tokio::test]
async fn split_frames_across_chunks() {
    let stream = frames(&["data: {\"de", "lta\":\"chunked\"}\n", "\ndata: [DONE]\n\n"]);
    let mut events = sse_json_stream::<Chunk>(stream);
    let evt = events.next().await.unwrap().unwrap();
    assert!(matches!(evt, SseEvent::Data(Chunk { delta }) if delta == "chunked"));
    assert!(events.next().await.is_none());
}

#[tokio::test]
async fn transport_error_propagates() {
    let err = ProviderError::api_call_builder("https://test", "boom").build();
    let s = stream::iter(vec![
        Ok(Bytes::from("data: {\"delta\":\"x\"}\n\n")),
        Err(err),
    ]);
    let mut events = sse_json_stream::<Chunk>(s);
    // First event ok
    let evt = events.next().await.unwrap().unwrap();
    assert!(matches!(evt, SseEvent::Data(_)));
    // Then the transport error surfaces
    let next = events.next().await.unwrap();
    assert!(next.is_err());
}
