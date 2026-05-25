//! Contract tests for [`CohereChatModel::do_stream`].
// Rust guideline compliant 2026-05-25

use futures::StreamExt;
use llmsdk_cohere::Cohere;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, FinishReasonKind, Message, StreamPart, TextPart, UserPart,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Cohere {
    Cohere::builder()
        .api_key("cohere-test")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
}

fn user_text(text: &str) -> Message {
    Message::User {
        content: vec![UserPart::Text(TextPart {
            text: text.into(),
            provider_options: None,
        })],
        provider_options: None,
    }
}

fn sse(events: &[&str]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for e in events {
        let _ = write!(out, "data: {e}\n\n");
    }
    out
}

#[tokio::test]
async fn stream_text_lifecycle() {
    let server = MockServer::start().await;
    let body = sse(&[
        r#"{"type":"message-start","id":"r1"}"#,
        r#"{"type":"content-start","index":0,"delta":{"message":{"content":{"type":"text","text":""}}}}"#,
        r#"{"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"hel"}}}}"#,
        r#"{"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"lo"}}}}"#,
        r#"{"type":"content-end","index":0}"#,
        r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":5,"output_tokens":3}}}}"#,
    ]);

    Mock::given(method("POST"))
        .and(path("/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;

    let result = provider(&server)
        .chat("command-a-03-2025")
        .do_stream(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("stream opens");

    let mut stream = result.stream;
    let mut frames = Vec::new();
    while let Some(item) = stream.next().await {
        frames.push(item.expect("ok"));
    }

    assert!(matches!(&frames[0], StreamPart::StreamStart { .. }));
    assert!(matches!(&frames[1], StreamPart::ResponseMetadata(_)));
    assert!(matches!(&frames[2], StreamPart::TextStart { .. }));
    assert!(matches!(&frames[3], StreamPart::TextDelta { delta, .. } if delta == "hel"));
    assert!(matches!(&frames[4], StreamPart::TextDelta { delta, .. } if delta == "lo"));
    assert!(matches!(&frames[5], StreamPart::TextEnd { .. }));
    let StreamPart::Finish {
        finish_reason,
        usage,
        ..
    } = frames.last().unwrap()
    else {
        panic!("expected Finish");
    };
    assert_eq!(finish_reason.unified, FinishReasonKind::Stop);
    assert_eq!(usage.input_tokens.total, Some(5));
}

#[tokio::test]
async fn stream_tool_call_lifecycle() {
    let server = MockServer::start().await;
    let body = sse(&[
        r#"{"type":"message-start","id":"r2"}"#,
        r#"{"type":"tool-call-start","delta":{"message":{"tool_calls":{"id":"c1","type":"function","function":{"name":"weather","arguments":"{\"city\""}}}}}"#,
        r#"{"type":"tool-call-delta","delta":{"message":{"tool_calls":{"function":{"arguments":":\"NYC\"}"}}}}}"#,
        r#"{"type":"tool-call-end"}"#,
        r#"{"type":"message-end","delta":{"finish_reason":"TOOL_CALL","usage":{"tokens":{"input_tokens":1,"output_tokens":2}}}}"#,
    ]);

    Mock::given(method("POST"))
        .and(path("/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;

    let result = provider(&server)
        .chat("command-a-03-2025")
        .do_stream(CallOptions {
            prompt: vec![user_text("weather?")],
            ..Default::default()
        })
        .await
        .expect("stream opens");

    let mut stream = result.stream;
    let mut frames = Vec::new();
    while let Some(item) = stream.next().await {
        frames.push(item.expect("ok"));
    }

    // Locate the ToolCall frame.
    let tool_call = frames
        .iter()
        .find_map(|f| {
            if let StreamPart::ToolCall(tc) = f {
                Some(tc)
            } else {
                None
            }
        })
        .expect("ToolCall");
    assert_eq!(tool_call.tool_call_id, "c1");
    assert_eq!(tool_call.tool_name, "weather");
    assert_eq!(tool_call.input["city"], "NYC");

    let finish = frames
        .iter()
        .rev()
        .find_map(|f| {
            if let StreamPart::Finish { finish_reason, .. } = f {
                Some(finish_reason)
            } else {
                None
            }
        })
        .expect("Finish");
    assert_eq!(finish.unified, FinishReasonKind::ToolCalls);
}

#[tokio::test]
async fn stream_reasoning_block() {
    let server = MockServer::start().await;
    let body = sse(&[
        r#"{"type":"content-start","index":0,"delta":{"message":{"content":{"type":"thinking","thinking":""}}}}"#,
        r#"{"type":"content-delta","index":0,"delta":{"message":{"content":{"thinking":"let me think"}}}}"#,
        r#"{"type":"content-end","index":0}"#,
        r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":2}}}}"#,
    ]);

    Mock::given(method("POST"))
        .and(path("/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;

    let result = provider(&server)
        .chat("command-a-reasoning-08-2025")
        .do_stream(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("stream opens");

    let mut stream = result.stream;
    let mut frames = Vec::new();
    while let Some(item) = stream.next().await {
        frames.push(item.expect("ok"));
    }

    let has_reasoning_delta = frames
        .iter()
        .any(|f| matches!(f, StreamPart::ReasoningDelta { delta, .. } if delta == "let me think"));
    assert!(has_reasoning_delta);
}
