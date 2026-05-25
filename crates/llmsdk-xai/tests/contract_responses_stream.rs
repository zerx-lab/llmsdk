//! Contract tests for [`XaiResponsesLanguageModel::do_stream`].
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

async fn collect_stream(provider: &Xai, model_id: &str) -> Vec<StreamPart> {
    let model = provider.responses(model_id);
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
        r#"data: {"type":"response.created","response":{"id":"resp_s1","model":"grok-4.3","object":"response","output":[]}}"#,
        "\n\n",
        r#"data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"hel"}"#,
        "\n\n",
        r#"data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"lo"}"#,
        "\n\n",
        r#"data: {"type":"response.completed","response":{"id":"resp_s1","model":"grok-4.3","object":"response","output":[],"status":"completed","usage":{"input_tokens":3,"output_tokens":2}}}"#,
        "\n\n",
        "data: [DONE]\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(serde_json::json!({ "stream": true })))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;

    let parts = collect_stream(&provider(&server), "grok-4.3").await;

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

    let finish = parts
        .iter()
        .rev()
        .find_map(|p| match p {
            StreamPart::Finish { finish_reason, .. } => Some(finish_reason.unified),
            _ => None,
        })
        .expect("expected Finish");
    assert_eq!(finish, FinishReasonKind::Stop);
}

#[tokio::test]
async fn streams_function_call_arguments_then_tool_call() {
    let body = concat!(
        r#"data: {"type":"response.created","response":{"id":"resp_t1","model":"grok-4.3","object":"response","output":[]}}"#,
        "\n\n",
        r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"weather","arguments":""}}"#,
        "\n\n",
        r#"data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"{\"city\":"}"#,
        "\n\n",
        r#"data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"\"NYC\"}"}"#,
        "\n\n",
        r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"weather","arguments":"{\"city\":\"NYC\"}"}}"#,
        "\n\n",
        r#"data: {"type":"response.completed","response":{"id":"resp_t1","model":"grok-4.3","object":"response","output":[],"status":"completed","usage":{"input_tokens":0,"output_tokens":0}}}"#,
        "\n\n",
        "data: [DONE]\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;

    let parts = collect_stream(&provider(&server), "grok-4.3").await;

    // ToolInputStart present
    assert!(parts.iter().any(
        |p| matches!(p, StreamPart::ToolInputStart { tool_name, .. } if tool_name == "weather")
    ));
    // Two deltas present
    let delta_count = parts
        .iter()
        .filter(|p| matches!(p, StreamPart::ToolInputDelta { .. }))
        .count();
    assert!(delta_count >= 2);
    // ToolCall emitted with assembled input
    let Some(StreamPart::ToolCall(tc)) = parts
        .iter()
        .find(|p| matches!(p, StreamPart::ToolCall(_)))
        .cloned()
    else {
        panic!("expected ToolCall");
    };
    assert_eq!(tc.tool_call_id, "call_1");
    assert_eq!(tc.input["city"], "NYC");

    // Finish reason is tool-calls because we saw a function_call
    let finish = parts
        .iter()
        .rev()
        .find_map(|p| match p {
            StreamPart::Finish { finish_reason, .. } => Some(finish_reason.unified),
            _ => None,
        })
        .expect("expected Finish");
    assert_eq!(finish, FinishReasonKind::ToolCalls);
}

#[tokio::test]
async fn streams_reasoning_summary_then_text() {
    let body = concat!(
        r#"data: {"type":"response.created","response":{"id":"resp_r1","model":"grok-4.20-reasoning","object":"response","output":[]}}"#,
        "\n\n",
        r#"data: {"type":"response.reasoning_summary_part.added","item_id":"rs_1","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}}"#,
        "\n\n",
        r#"data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"thinking"}"#,
        "\n\n",
        r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"thinking"}],"status":"completed"}}"#,
        "\n\n",
        r#"data: {"type":"response.completed","response":{"id":"resp_r1","model":"grok-4.20-reasoning","object":"response","output":[],"status":"completed","usage":{"input_tokens":0,"output_tokens":0}}}"#,
        "\n\n",
        "data: [DONE]\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;

    let parts = collect_stream(&provider(&server), "grok-4.20-reasoning").await;

    assert!(
        parts
            .iter()
            .any(|p| matches!(p, StreamPart::ReasoningStart { .. }))
    );
    assert!(
        parts
            .iter()
            .any(|p| matches!(p, StreamPart::ReasoningDelta { delta, .. } if delta == "thinking"))
    );
    assert!(
        parts
            .iter()
            .any(|p| matches!(p, StreamPart::ReasoningEnd { .. }))
    );
}

#[tokio::test]
async fn error_chunk_surfaces_as_error_frame() {
    let body = concat!(
        r#"data: {"type":"response.created","response":{"id":"resp_e1","object":"response","output":[]}}"#,
        "\n\n",
        r#"data: {"type":"error","code":"e1","message":"boom"}"#,
        "\n\n",
        "data: [DONE]\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;

    let parts = collect_stream(&provider(&server), "grok-4.3").await;
    assert!(parts.iter().any(|p| matches!(p, StreamPart::Error { .. })));
}
