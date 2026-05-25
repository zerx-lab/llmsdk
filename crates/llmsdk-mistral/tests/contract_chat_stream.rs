//! Contract tests for [`MistralChatModel::do_stream`].
// Rust guideline compliant 2026-05-25

use futures::StreamExt;
use llmsdk_mistral::Mistral;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, FinishReasonKind, Message, StreamPart, TextPart, UserPart,
};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Mistral {
    Mistral::builder()
        .api_key("test-api-key")
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

/// Wraps a list of JSON object strings into SSE `data:` frames.
fn sse_body(frames: &[&str]) -> String {
    let mut out = String::new();
    for f in frames {
        out.push_str("data: ");
        out.push_str(f);
        out.push_str("\n\n");
    }
    out.push_str("data: [DONE]\n\n");
    out
}

#[tokio::test]
async fn streams_text_chunks_to_text_parts() {
    let server = MockServer::start().await;
    let frames = sse_body(&[
        r#"{"id":"r1","object":"chat.completion.chunk","created":1769088720,"model":"mistral-small-latest","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
        r#"{"id":"r1","object":"chat.completion.chunk","created":1769088720,"model":"mistral-small-latest","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#,
        r#"{"id":"r1","object":"chat.completion.chunk","created":1769088720,"model":"mistral-small-latest","choices":[{"index":0,"delta":{"content":", world"},"finish_reason":null}]}"#,
        r#"{"id":"r1","object":"chat.completion.chunk","created":1769088720,"model":"mistral-small-latest","choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":13,"total_tokens":21,"completion_tokens":8}}"#,
    ]);
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(frames),
        )
        .mount(&server)
        .await;

    let provider = provider(&server);
    let mut result = provider
        .chat("mistral-small-latest")
        .do_stream(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    let mut parts = Vec::new();
    while let Some(p) = result.stream.next().await {
        parts.push(p.expect("ok"));
    }

    let mut text = String::new();
    let mut finish_found = false;
    for p in &parts {
        match p {
            StreamPart::TextDelta { delta, .. } => text.push_str(delta),
            StreamPart::Finish { finish_reason, .. } => {
                assert_eq!(finish_reason.unified, FinishReasonKind::Stop);
                finish_found = true;
            }
            _ => {}
        }
    }
    assert_eq!(text, "Hello, world");
    assert!(finish_found);
}

#[tokio::test]
async fn streams_thinking_then_text_emits_reasoning_block_first() {
    let server = MockServer::start().await;
    let frames = sse_body(&[
        r#"{"id":"r1","object":"chat.completion.chunk","created":1,"model":"magistral-medium-2507","choices":[{"index":0,"delta":{"role":"assistant","content":[{"type":"thinking","thinking":[{"type":"text","text":"The user is asking"}]}]},"finish_reason":null}]}"#,
        r#"{"id":"r1","object":"chat.completion.chunk","created":1,"model":"magistral-medium-2507","choices":[{"index":0,"delta":{"content":[{"type":"text","text":"2 + 2 = 4"}]},"finish_reason":null}]}"#,
        r#"{"id":"r1","object":"chat.completion.chunk","created":1,"model":"magistral-medium-2507","choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"total_tokens":56,"completion_tokens":46}}"#,
    ]);
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(frames),
        )
        .mount(&server)
        .await;

    let provider = provider(&server);
    let mut result = provider
        .chat("magistral-medium-2507")
        .do_stream(CallOptions {
            prompt: vec![user_text("what is 2+2")],
            ..Default::default()
        })
        .await
        .expect("ok");

    let mut parts = Vec::new();
    while let Some(p) = result.stream.next().await {
        parts.push(p.expect("ok"));
    }

    // Expect at least: StreamStart, ResponseMetadata, ReasoningStart, ReasoningDelta,
    // ReasoningEnd, TextStart, TextDelta, TextEnd, Finish
    let reasoning_index = parts
        .iter()
        .position(|p| matches!(p, StreamPart::ReasoningStart { .. }))
        .expect("reasoning-start");
    let text_index = parts
        .iter()
        .position(|p| matches!(p, StreamPart::TextStart { .. }))
        .expect("text-start");
    assert!(reasoning_index < text_index);
    assert!(matches!(parts.last(), Some(StreamPart::Finish { .. })));
}

#[tokio::test]
async fn streams_tool_call_in_one_chunk() {
    let server = MockServer::start().await;
    let frames = sse_body(&[
        r#"{"id":"r1","object":"chat.completion.chunk","created":1,"model":"mistral-small-latest","choices":[{"index":0,"delta":{"content":null,"tool_calls":[{"id":"gSIMJiOkT","function":{"name":"weather","arguments":"{\"location\": \"San Francisco\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":124,"total_tokens":146,"completion_tokens":22}}"#,
    ]);
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(frames),
        )
        .mount(&server)
        .await;

    let provider = provider(&server);
    let mut result = provider
        .chat("mistral-small-latest")
        .do_stream(CallOptions {
            prompt: vec![user_text("weather?")],
            ..Default::default()
        })
        .await
        .expect("ok");

    let mut parts = Vec::new();
    while let Some(p) = result.stream.next().await {
        parts.push(p.expect("ok"));
    }

    assert!(
        parts
            .iter()
            .any(|p| matches!(p, StreamPart::ToolInputStart { .. }))
    );
    let tc = parts
        .iter()
        .find_map(|p| match p {
            StreamPart::ToolCall(tc) => Some(tc),
            _ => None,
        })
        .expect("ToolCall present");
    assert_eq!(tc.tool_call_id, "gSIMJiOkT");
    assert_eq!(tc.tool_name, "weather");
    let StreamPart::Finish { finish_reason, .. } = parts.last().unwrap() else {
        panic!("expected Finish");
    };
    assert_eq!(finish_reason.unified, FinishReasonKind::ToolCalls);
}
