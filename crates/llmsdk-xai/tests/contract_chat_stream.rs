//! Contract tests for [`XaiChatModel::do_stream`].
// Rust guideline compliant 2026-05-25

use futures::StreamExt;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, FinishReasonKind, Message, StreamPart, TextPart, UserPart,
};
use llmsdk_xai::Xai;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Xai {
    Xai::builder()
        .api_key("xai-test")
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

async fn collect_stream(provider: Xai) -> Vec<StreamPart> {
    let model = provider.chat("grok-4.3");
    let result = model
        .do_stream(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("do_stream ok");
    let mut parts = Vec::new();
    let mut stream = result.stream;
    while let Some(item) = stream.next().await {
        parts.push(item.expect("no transport error"));
    }
    parts
}

fn sse_response(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(body.to_owned())
}

#[tokio::test]
async fn streams_text_then_finish() {
    let body = concat!(
        r#"data: {"id":"chatcmpl-s1","created":1,"model":"grok-4.3","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"content":"hel"}}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"content":"lo"}}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "\n\n",
        r#"data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}"#,
        "\n\n",
        "data: [DONE]\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(serde_json::json!({ "stream": true })))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;

    let parts = collect_stream(provider(&server)).await;

    assert!(matches!(parts[0], StreamPart::StreamStart { .. }));
    assert!(matches!(parts[1], StreamPart::ResponseMetadata(_)));
    let deltas: Vec<&str> = parts
        .iter()
        .filter_map(|p| match p {
            StreamPart::TextDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, vec!["hel", "lo"]);
    assert!(
        parts
            .iter()
            .any(|p| matches!(p, StreamPart::TextEnd { .. }))
    );
    let Some(StreamPart::Finish { finish_reason, .. }) = parts.last() else {
        panic!("expected Finish");
    };
    assert_eq!(finish_reason.unified, FinishReasonKind::Stop);
}

#[tokio::test]
async fn streams_reasoning_then_text_closes_reasoning_first() {
    let body = concat!(
        r#"data: {"id":"r1","choices":[{"index":0,"delta":{"reasoning_content":"think..."}}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"content":"ans"}}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "\n\n",
        "data: [DONE]\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;

    let parts = collect_stream(provider(&server)).await;
    let kinds: Vec<&'static str> = parts
        .iter()
        .map(|p| match p {
            StreamPart::StreamStart { .. } => "stream-start",
            StreamPart::ResponseMetadata(_) => "response-metadata",
            StreamPart::ReasoningStart { .. } => "reasoning-start",
            StreamPart::ReasoningDelta { .. } => "reasoning-delta",
            StreamPart::ReasoningEnd { .. } => "reasoning-end",
            StreamPart::TextStart { .. } => "text-start",
            StreamPart::TextDelta { .. } => "text-delta",
            StreamPart::TextEnd { .. } => "text-end",
            StreamPart::Finish { .. } => "finish",
            _ => "other",
        })
        .collect();
    // The reasoning-end must appear before the text-start.
    let r_end = kinds.iter().position(|k| *k == "reasoning-end").unwrap();
    let t_start = kinds.iter().position(|k| *k == "text-start").unwrap();
    assert!(r_end < t_start);
}

#[tokio::test]
async fn streams_tool_call_in_single_chunk() {
    let body = concat!(
        r#"data: {"id":"r1","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_w","type":"function","function":{"name":"weather","arguments":"{\"city\":\"NYC\"}"}}]},"finish_reason":"tool_calls"}]}"#,
        "\n\n",
        "data: [DONE]\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;

    let parts = collect_stream(provider(&server)).await;
    assert!(
        parts
            .iter()
            .any(|p| matches!(p, StreamPart::ToolInputStart { .. }))
    );
    assert!(parts.iter().any(|p| matches!(p, StreamPart::ToolCall(_))));
    let Some(StreamPart::Finish { finish_reason, .. }) = parts.last() else {
        panic!("expected Finish");
    };
    assert_eq!(finish_reason.unified, FinishReasonKind::ToolCalls);
}

#[tokio::test]
async fn json_error_body_becomes_in_stream_error() {
    // wiremock's `set_body_string` hard-codes `text/plain`; we want to
    // simulate xAI's `application/json` error envelope, so pass the bytes
    // directly through `set_body_raw` which preserves our content-type.
    let body = br#"{"code":"rate_limit","error":"too many"}"#;
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body.to_vec(), "application/json"))
        .mount(&server)
        .await;

    let parts = collect_stream(provider(&server)).await;
    assert!(parts.iter().any(|p| matches!(p, StreamPart::Error { .. })));
    let Some(StreamPart::Finish { finish_reason, .. }) = parts.last() else {
        panic!("expected Finish");
    };
    assert_eq!(finish_reason.unified, FinishReasonKind::Error);
}
