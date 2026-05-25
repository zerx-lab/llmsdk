//! Contract tests for `ConverseStream` — end-to-end through the AWS binary
//! `EventStream` decoder.
//!
//! Frames are encoded with `aws-smithy-eventstream` (only re-used inside
//! `llmsdk-provider-utils`'s test code; here we craft them by hand using the
//! same crate via `cargo test` 's transitive dev-dep graph). To keep the
//! contract test free of an explicit dep, the encoder is bypassed: we craft
//! a minimal frame manually using the documented length-prefixed layout and
//! pre-computed CRCs. The decoder in `llmsdk_provider_utils::aws_eventstream`
//! validates the CRCs, so this exercises the full decode path end-to-end.
// Rust guideline compliant 2026-05-25

use bytes::Bytes;
use futures::StreamExt;
use llmsdk_amazon_bedrock::AmazonBedrock;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{CallOptions, Message, StreamPart, TextPart, UserPart};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Build a single `EventStream` frame.
///
/// Layout:
/// - prelude  = `total_len:u32 + headers_len:u32 + prelude_crc:u32`
/// - headers  = sequence of `name_len:u8 + name + value_type:u8 + ...`
/// - payload  = raw bytes
/// - `message_crc` = `u32`
fn build_frame(event_type: &str, payload: &[u8]) -> Vec<u8> {
    // Two string headers: `:message-type = "event"`, `:event-type = "<event_type>"`,
    // `:content-type = "application/json"`.
    let mut headers: Vec<u8> = Vec::new();
    append_string_header(&mut headers, ":message-type", "event");
    append_string_header(&mut headers, ":event-type", event_type);
    append_string_header(&mut headers, ":content-type", "application/json");

    let headers_len: u32 = u32::try_from(headers.len()).expect("headers fit");
    let total_len: u32 =
        u32::try_from(12 + headers.len() + payload.len() + 4).expect("frame fits in u32");

    let mut frame: Vec<u8> = Vec::with_capacity(total_len as usize);
    frame.extend_from_slice(&total_len.to_be_bytes());
    frame.extend_from_slice(&headers_len.to_be_bytes());
    let prelude_crc = crc32(&frame[..8]);
    frame.extend_from_slice(&prelude_crc.to_be_bytes());
    frame.extend_from_slice(&headers);
    frame.extend_from_slice(payload);
    let msg_crc = crc32(&frame);
    frame.extend_from_slice(&msg_crc.to_be_bytes());
    frame
}

fn append_string_header(out: &mut Vec<u8>, name: &str, value: &str) {
    out.push(u8::try_from(name.len()).expect("name fits"));
    out.extend_from_slice(name.as_bytes());
    out.push(7); // value type tag for STRING
    out.extend_from_slice(
        &u16::try_from(value.len())
            .expect("value fits")
            .to_be_bytes(),
    );
    out.extend_from_slice(value.as_bytes());
}

// Minimal CRC-32/IEEE (polynomial 0xEDB88320) implementation. Matches the
// `aws-smithy-eventstream` encoder used by the upstream tests.
fn crc32(input: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut c = u32::try_from(i).expect("byte index fits");
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        *slot = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for byte in input {
        crc = table[((crc ^ u32::from(*byte)) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

fn provider(server: &MockServer) -> AmazonBedrock {
    AmazonBedrock::builder()
        .region("us-east-1")
        .api_key("bearer-test")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
}

#[tokio::test]
async fn streaming_text_emits_text_start_delta_end_finish() {
    let server = MockServer::start().await;

    // Build the frame sequence: contentBlockDelta (text) -> contentBlockStop
    // -> messageStop -> metadata. Each event payload is JSON.
    let mut body = Vec::new();
    body.extend(build_frame(
        "contentBlockDelta",
        json!({ "contentBlockIndex": 0, "delta": { "text": "hi" } })
            .to_string()
            .as_bytes(),
    ));
    body.extend(build_frame(
        "contentBlockDelta",
        json!({ "contentBlockIndex": 0, "delta": { "text": " there" } })
            .to_string()
            .as_bytes(),
    ));
    body.extend(build_frame(
        "contentBlockStop",
        json!({ "contentBlockIndex": 0 }).to_string().as_bytes(),
    ));
    body.extend(build_frame(
        "messageStop",
        json!({ "stopReason": "end_turn" }).to_string().as_bytes(),
    ));
    body.extend(build_frame(
        "metadata",
        json!({ "usage": { "inputTokens": 4, "outputTokens": 2, "totalTokens": 6 } })
            .to_string()
            .as_bytes(),
    ));

    Mock::given(method("POST"))
        .and(path(
            "/model/anthropic.claude-3-5-haiku-20241022-v1%3A0/converse-stream",
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/vnd.amazon.eventstream")
                .set_body_bytes(Bytes::from(body)),
        )
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.language_model("anthropic.claude-3-5-haiku-20241022-v1:0");
    let result = model
        .do_stream(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserPart::Text(TextPart {
                    text: "hi".into(),
                    provider_options: None,
                })],
                provider_options: None,
            }],
            ..Default::default()
        })
        .await
        .expect("stream opens");

    let mut parts: Vec<StreamPart> = Vec::new();
    let mut stream = result.stream;
    while let Some(p) = stream.next().await {
        parts.push(p.expect("part"));
    }
    // sanity: contains text-start, text-delta, text-end, finish (in some order)
    let kinds: Vec<&'static str> = parts
        .iter()
        .map(|p| match p {
            StreamPart::StreamStart { .. } => "stream-start",
            StreamPart::ResponseMetadata(_) => "response-metadata",
            StreamPart::TextStart { .. } => "text-start",
            StreamPart::TextDelta { .. } => "text-delta",
            StreamPart::TextEnd { .. } => "text-end",
            StreamPart::Finish { .. } => "finish",
            _ => "other",
        })
        .collect();
    assert!(kinds.contains(&"text-start"), "kinds={kinds:?}");
    assert!(kinds.contains(&"text-delta"), "kinds={kinds:?}");
    assert!(kinds.contains(&"text-end"), "kinds={kinds:?}");
    assert!(kinds.contains(&"finish"), "kinds={kinds:?}");
}
