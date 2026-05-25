//! Contract tests for [`OpenAiResponsesLanguageModel::do_stream`].
// Rust guideline compliant 2026-02-21

use futures::StreamExt;
use llmsdk_openai::OpenAi;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, FinishReasonKind, Message, StreamPart, TextPart, UserPart,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> OpenAi {
    OpenAi::builder()
        .api_key("sk-test")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
}

fn user_text(s: &str) -> Message {
    Message::User {
        content: vec![UserPart::Text(TextPart {
            text: s.into(),
            provider_options: None,
        })],
        provider_options: None,
    }
}

fn sse(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(body.to_owned())
}

async fn collect(provider: OpenAi, model_id: &str) -> Vec<StreamPart> {
    let model = provider.responses(model_id);
    let res = model
        .do_stream(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("stream open");
    let mut out = Vec::new();
    let mut s = res.stream;
    while let Some(p) = s.next().await {
        out.push(p.expect("no transport error"));
    }
    out
}

#[tokio::test]
async fn streams_text_then_finish() {
    let body = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_s1\",\"created_at\":1,\"model\":\"gpt-4o-mini\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\"}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"delta\":\"hel\"}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"delta\":\"lo\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\",\"annotations\":[]}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":2}}}\n\n",
        "data: [DONE]\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(sse(body))
        .mount(&server)
        .await;

    let parts = collect(provider(&server), "gpt-4o-mini").await;
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
    let StreamPart::Finish {
        finish_reason,
        usage,
        ..
    } = parts.last().unwrap()
    else {
        panic!("expected Finish");
    };
    assert_eq!(finish_reason.unified, FinishReasonKind::Stop);
    assert_eq!(usage.input_tokens.total, Some(3));
    assert_eq!(usage.output_tokens.total, Some(2));
}

#[tokio::test]
async fn streams_function_call_with_argument_deltas() {
    let body = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_s2\",\"created_at\":1,\"model\":\"gpt-4o-mini\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_x\",\"name\":\"weather\",\"arguments\":\"\"}}\n\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"output_index\":0,\"delta\":\"{\\\"city\"}\n\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"output_index\":0,\"delta\":\"\\\":\\\"NYC\\\"}\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_x\",\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\\\"NYC\\\"}\",\"status\":\"completed\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":4,\"output_tokens\":2}}}\n\n",
        "data: [DONE]\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(sse(body))
        .mount(&server)
        .await;

    let parts = collect(provider(&server), "gpt-4o-mini").await;
    let deltas: Vec<&str> = parts
        .iter()
        .filter_map(|p| match p {
            StreamPart::ToolInputDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas.join(""), "{\"city\":\"NYC\"}");
    assert!(matches!(
        parts.last(),
        Some(StreamPart::Finish {
            finish_reason,
            ..
        }) if finish_reason.unified == FinishReasonKind::ToolCalls
    ));
}

#[tokio::test]
async fn streams_url_citation_as_source() {
    let body = concat!(
        "data: {\"type\":\"response.output_text.annotation.added\",\"annotation\":{\"type\":\"url_citation\",\"start_index\":0,\"end_index\":1,\"url\":\"https://x\",\"title\":\"X\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\n",
        "data: [DONE]\n\n",
    );
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(sse(body))
        .mount(&server)
        .await;

    let parts = collect(provider(&server), "gpt-4o-mini").await;
    assert!(parts.iter().any(|p| matches!(p, StreamPart::Source(_))));
}

#[tokio::test]
async fn error_chunk_emits_error_stream_part() {
    let body = concat!(
        "data: {\"type\":\"error\",\"sequence_number\":1,\"error\":{\"type\":\"rate_limit\",\"code\":\"rate_limit_exceeded\",\"message\":\"slow\"}}\n\n",
        "data: [DONE]\n\n",
    );
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(sse(body))
        .mount(&server)
        .await;
    let parts = collect(provider(&server), "gpt-4o-mini").await;
    assert!(parts.iter().any(|p| matches!(p, StreamPart::Error { .. })));
    let StreamPart::Finish { finish_reason, .. } = parts.last().unwrap() else {
        panic!("expected finish");
    };
    assert_eq!(finish_reason.unified, FinishReasonKind::Error);
}
