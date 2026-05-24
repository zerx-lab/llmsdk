//! Contract tests for [`OpenAiChatModel::do_stream`].
//!
//! Wiremock serves a complete SSE payload in one response body. The
//! underlying `eventsource-stream` parser frames events correctly regardless
//! of chunking, so collecting the produced [`StreamPart`]s is sufficient.
// Rust guideline compliant 2026-02-21

use futures::StreamExt;
use llmsdk_openai::OpenAi;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, FinishReasonKind, Message, StreamPart, TextPart, UserPart,
};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> OpenAi {
    OpenAi::builder()
        .api_key("sk-test")
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

async fn collect_stream(provider: OpenAi) -> Vec<StreamPart> {
    let model = provider.chat("gpt-4o-mini");
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

const SSE_HEADER: &str = "text/event-stream";

fn sse_response(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", SSE_HEADER)
        .set_body_string(body.to_owned())
}

#[tokio::test]
async fn streams_text_then_finish() {
    let body = concat!(
        "data: {\"id\":\"chatcmpl-s1\",\"created\":1,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n",
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

    // Expected order:
    // StreamStart, ResponseMetadata, TextStart, TextDelta("hel"), TextDelta("lo"),
    // TextEnd, Finish
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
        assert_eq!(usage.input_tokens.total, Some(3));
        assert_eq!(usage.output_tokens.total, Some(2));
    } else {
        panic!("expected Finish frame at end");
    }
}

#[tokio::test]
async fn streams_tool_call_across_chunks() {
    let body = concat!(
        "data: {\"id\":\"chatcmpl-s2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_w\",\"type\":\"function\",\"function\":{\"name\":\"weather\",\"arguments\":\"{\\\"ci\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ty\\\":\\\"NYC\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;

    let parts = collect_stream(provider(&server)).await;

    // Find ToolInputStart, ToolInputDelta(s), ToolInputEnd, ToolCall
    let tool_input_start = parts
        .iter()
        .position(|p| matches!(p, StreamPart::ToolInputStart { .. }))
        .expect("ToolInputStart present");
    let tool_call_idx = parts
        .iter()
        .position(|p| matches!(p, StreamPart::ToolCall(_)))
        .expect("ToolCall present");
    assert!(tool_call_idx > tool_input_start);

    if let StreamPart::ToolCall(tc) = &parts[tool_call_idx] {
        assert_eq!(tc.tool_call_id, "call_w");
        assert_eq!(tc.tool_name, "weather");
        assert_eq!(tc.input["city"], "NYC");
    } else {
        panic!("not a tool call");
    }

    if let StreamPart::Finish { finish_reason, .. } = parts.last().unwrap() {
        assert_eq!(finish_reason.unified, FinishReasonKind::ToolCalls);
    }
}

#[tokio::test]
async fn mid_stream_error_chunk_surfaces() {
    let body = concat!(
        "data: {\"id\":\"chatcmpl-s3\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"oops\"}}]}\n\n",
        "data: {\"error\":{\"message\":\"server boom\",\"type\":\"server_error\"}}\n\n",
        "data: [DONE]\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;

    let parts = collect_stream(provider(&server)).await;

    let has_error = parts.iter().any(|p| matches!(p, StreamPart::Error { .. }));
    assert!(has_error, "expected an Error frame");

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
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .set_body_json(serde_json::json!({ "error": { "message": "slow down" } })),
        )
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.chat("gpt-4o-mini");
    let Err(err) = model
        .do_stream(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
    else {
        panic!("should be 429");
    };
    assert!(err.is_api_call());
    assert!(err.is_retryable());
    assert!(format!("{err}").contains("slow down"));
}

#[tokio::test]
async fn parse_failure_yields_error_frame() {
    // A data line that is not valid JSON should surface as a StreamPart::Error
    // without killing the stream.
    let body = concat!(
        "data: not-json\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;

    let parts = collect_stream(provider(&server)).await;
    assert!(parts.iter().any(|p| matches!(p, StreamPart::Error { .. })));
    // Stream still terminates with Finish.
    assert!(matches!(parts.last(), Some(StreamPart::Finish { .. })));
}
