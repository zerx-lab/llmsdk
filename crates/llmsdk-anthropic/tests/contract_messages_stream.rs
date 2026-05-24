//! Contract tests for [`AnthropicMessagesModel::do_stream`].
// Rust guideline compliant 2026-02-21

use futures::StreamExt;
use llmsdk_anthropic::Anthropic;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, FinishReasonKind, Message, StreamPart, TextPart, UserPart,
};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Anthropic {
    Anthropic::builder()
        .api_key("sk-ant-test")
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

async fn collect(provider: Anthropic) -> Vec<StreamPart> {
    let model = provider.messages("claude-3-5-sonnet-latest");
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

fn sse(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(body.to_owned())
}

#[tokio::test]
async fn streams_text_block_and_finish() {
    let body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s1\",\"model\":\"claude-3-5\",\"usage\":{\"input_tokens\":4}}}\n\n",
        "event: ping\n",
        "data: {\"type\":\"ping\"}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hel\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(serde_json::json!({ "stream": true })))
        .respond_with(sse(body))
        .mount(&server)
        .await;

    let parts = collect(provider(&server)).await;

    // Order: StreamStart, ResponseMetadata, TextStart, TextDelta("hel"),
    // TextDelta("lo"), TextEnd, Finish
    assert!(matches!(parts[0], StreamPart::StreamStart { .. }));
    assert!(matches!(parts[1], StreamPart::ResponseMetadata(_)));
    assert!(matches!(parts[2], StreamPart::TextStart { .. }));
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

    let last = parts.last().unwrap();
    if let StreamPart::Finish {
        finish_reason,
        usage,
        ..
    } = last
    {
        assert_eq!(finish_reason.unified, FinishReasonKind::Stop);
        assert_eq!(usage.input_tokens.no_cache, Some(4));
        assert_eq!(usage.output_tokens.total, Some(2));
    } else {
        panic!("expected Finish");
    }
}

#[tokio::test]
async fn streams_tool_use_block() {
    let body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s2\",\"usage\":{\"input_tokens\":1}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu_w\",\"name\":\"weather\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"ci\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"ty\\\":\\\"NYC\\\"}\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":12}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(sse(body))
        .mount(&server)
        .await;

    let parts = collect(provider(&server)).await;
    let tool_call_idx = parts
        .iter()
        .position(|p| matches!(p, StreamPart::ToolCall(_)))
        .expect("ToolCall present");
    let tool_start_idx = parts
        .iter()
        .position(|p| matches!(p, StreamPart::ToolInputStart { .. }))
        .expect("ToolInputStart present");
    assert!(tool_call_idx > tool_start_idx);

    if let StreamPart::ToolCall(tc) = &parts[tool_call_idx] {
        assert_eq!(tc.tool_call_id, "tu_w");
        assert_eq!(tc.tool_name, "weather");
        assert_eq!(tc.input["city"], "NYC");
    }
    if let StreamPart::Finish { finish_reason, .. } = parts.last().unwrap() {
        assert_eq!(finish_reason.unified, FinishReasonKind::ToolCalls);
    }
}

#[tokio::test]
async fn mid_stream_error_event_marks_finish_as_error() {
    let body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s3\",\"usage\":{\"input_tokens\":1}}}\n\n",
        "event: error\n",
        "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"server overloaded\"}}\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(sse(body))
        .mount(&server)
        .await;

    let parts = collect(provider(&server)).await;
    assert!(parts.iter().any(|p| matches!(p, StreamPart::Error { .. })));
    if let StreamPart::Finish { finish_reason, .. } = parts.last().unwrap() {
        assert_eq!(finish_reason.unified, FinishReasonKind::Error);
    } else {
        panic!("expected Finish at end");
    }
}

#[tokio::test]
async fn http_429_before_stream_returns_outer_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
            "type": "error",
            "error": { "type": "rate_limit_error", "message": "slow down" }
        })))
        .mount(&server)
        .await;
    let Err(err) = provider(&server)
        .messages("claude-3-5-sonnet-latest")
        .do_stream(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
    else {
        panic!("should error");
    };
    assert!(err.is_api_call());
    assert!(err.is_retryable());
    assert!(format!("{err}").contains("slow down"));
}

#[tokio::test]
async fn ignores_other_unknown_event_types() {
    let body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"x\",\"usage\":{\"input_tokens\":1}}}\n\n",
        "event: future_feature\n",
        "data: {\"type\":\"future_feature\",\"payload\":42}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":0}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(sse(body))
        .mount(&server)
        .await;
    let parts = collect(provider(&server)).await;
    // No Error frames; just StreamStart + ResponseMetadata + Finish.
    assert!(!parts.iter().any(|p| matches!(p, StreamPart::Error { .. })));
    assert!(matches!(parts.last(), Some(StreamPart::Finish { .. })));
}
