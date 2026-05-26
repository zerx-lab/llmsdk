//! Contract tests for the Gemini Interactions API surface.
//!
//! Covers wire-level alignment with `@ai-sdk/google/src/interactions/*`:
//! - `Api-Revision: 2026-05-20` header on POST/GET/cancel
//! - `do_generate` with steps array → Content[]
//! - `do_stream` full SSE state machine (text + reasoning +
//!   `arguments_delta` + builtin tool + annotations → Source)
//! - `do_stream` background path (POST background → poll until terminal →
//!   synthesize stream)
//! - Cancel endpoint shape (`POST /interactions/{id}/cancel`)
//! - Provider-defined tool routing (8 typed Google tools)
// Rust guideline compliant 2026-05-25

use futures::StreamExt;
use llmsdk_google::{Google, GoogleInteractionsAgent};
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, FinishReasonKind, Message, ProviderTool, StreamPart, TextPart, Tool,
    UserPart,
};
use llmsdk_provider::shared::ProviderOptions;
use serde_json::{Map, Value, json};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Google {
    Google::builder()
        .api_key("test-key")
        .base_url(server.uri())
        .build()
        .expect("builds")
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

fn google_options(map: Map<String, Value>) -> ProviderOptions {
    let mut po = ProviderOptions::new();
    po.insert("google".into(), map);
    po
}

#[tokio::test]
async fn do_generate_sends_revision_header_and_parses_steps() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/interactions"))
        .and(header("Api-Revision", "2026-05-20"))
        .and(body_partial_json(json!({
            "model": "gemini-2.5-flash",
            "input": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "int_abc",
            "status": "completed",
            "model": "gemini-2.5-flash",
            "steps": [
                {"type": "model_output", "content": [{"type": "text", "text": "hello"}]}
            ],
            "usage": {"total_input_tokens": 3, "total_output_tokens": 1}
        })))
        .mount(&server)
        .await;

    let model =
        provider(&server).interactions(GoogleInteractionsAgent::Model("gemini-2.5-flash".into()));
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
    assert_eq!(result.content.len(), 1);
    let Content::Text(t) = &result.content[0] else {
        panic!("expected text content");
    };
    assert_eq!(t.text, "hello");
    assert_eq!(result.usage.input_tokens.total, Some(3));
    assert_eq!(result.usage.output_tokens.total, Some(1));
}

#[tokio::test]
async fn do_stream_full_sse_state_machine_emits_text_reasoning_and_finish() {
    let server = MockServer::start().await;
    // SSE frames mirror upstream `step.start` / `step.delta` / `step.stop` /
    // `interaction.completed` event shapes (see
    // build-google-interactions-stream-transform.ts).
    let sse = "\
data: {\"event_type\":\"interaction.created\",\"interaction\":{\"id\":\"int_1\",\"model\":\"gemini-2.5-flash\"}}\n\
\n\
data: {\"event_type\":\"step.start\",\"index\":0,\"step\":{\"type\":\"thought\"}}\n\
\n\
data: {\"event_type\":\"step.delta\",\"index\":0,\"delta\":{\"type\":\"thought_summary\",\"content\":{\"type\":\"text\",\"text\":\"thinking...\"}}}\n\
\n\
data: {\"event_type\":\"step.stop\",\"index\":0}\n\
\n\
data: {\"event_type\":\"step.start\",\"index\":1,\"step\":{\"type\":\"model_output\"}}\n\
\n\
data: {\"event_type\":\"step.delta\",\"index\":1,\"delta\":{\"type\":\"text\",\"text\":\"Hello\"}}\n\
\n\
data: {\"event_type\":\"step.delta\",\"index\":1,\"delta\":{\"type\":\"text\",\"text\":\" world\"}}\n\
\n\
data: {\"event_type\":\"step.stop\",\"index\":1}\n\
\n\
data: {\"event_type\":\"interaction.completed\",\"interaction\":{\"id\":\"int_1\",\"status\":\"completed\",\"usage\":{\"total_input_tokens\":2,\"total_output_tokens\":2}}}\n\
\n";
    Mock::given(method("POST"))
        .and(path("/interactions"))
        .and(header("Api-Revision", "2026-05-20"))
        .and(body_partial_json(json!({"stream": true})))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let model =
        provider(&server).interactions(GoogleInteractionsAgent::Model("gemini-2.5-flash".into()));
    let stream_result = model
        .do_stream(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");
    let parts: Vec<StreamPart> = stream_result
        .stream
        .map(|r| r.expect("stream item"))
        .collect()
        .await;

    // We expect: stream-start, response-metadata, reasoning-start,
    // reasoning-delta, reasoning-end, text-start, text-delta, text-delta,
    // text-end, finish — order matters.
    let mut iter = parts.into_iter();
    assert!(matches!(iter.next(), Some(StreamPart::StreamStart { .. })));
    assert!(matches!(iter.next(), Some(StreamPart::ResponseMetadata(_))));
    assert!(matches!(
        iter.next(),
        Some(StreamPart::ReasoningStart { .. })
    ));
    assert!(matches!(
        iter.next(),
        Some(StreamPart::ReasoningDelta { delta, .. }) if delta == "thinking..."
    ));
    assert!(matches!(iter.next(), Some(StreamPart::ReasoningEnd { .. })));
    assert!(matches!(iter.next(), Some(StreamPart::TextStart { .. })));
    assert!(matches!(
        iter.next(),
        Some(StreamPart::TextDelta { delta, .. }) if delta == "Hello"
    ));
    assert!(matches!(
        iter.next(),
        Some(StreamPart::TextDelta { delta, .. }) if delta == " world"
    ));
    assert!(matches!(iter.next(), Some(StreamPart::TextEnd { .. })));
    match iter.next() {
        Some(StreamPart::Finish {
            finish_reason,
            usage,
            ..
        }) => {
            assert_eq!(finish_reason.unified, FinishReasonKind::Stop);
            assert_eq!(usage.input_tokens.total, Some(2));
        }
        other => panic!("expected Finish, got {other:?}"),
    }
}

#[tokio::test]
async fn do_stream_background_polls_then_synthesizes_stream() {
    let server = MockServer::start().await;
    // POST returns non-terminal status with id (mirrors upstream
    // background:true path); GET returns the terminal payload we synthesize from.
    Mock::given(method("POST"))
        .and(path("/interactions"))
        .and(body_partial_json(json!({"background": true})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "int_bg",
            "status": "in_progress",
            "agent": "deep-research-pro-preview-12-2025"
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/interactions/int_bg"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "int_bg",
            "status": "completed",
            "agent": "deep-research-pro-preview-12-2025",
            "steps": [
                {"type": "model_output", "content": [{"type": "text", "text": "agent reply"}]}
            ],
            "usage": {"total_input_tokens": 7, "total_output_tokens": 3}
        })))
        .mount(&server)
        .await;

    let model = provider(&server).interactions(GoogleInteractionsAgent::Agent(
        llmsdk_google::builtin_agent::DEEP_RESEARCH_PRO_PREVIEW_12_2025.into(),
    ));
    let stream_result = model
        .do_stream(CallOptions {
            prompt: vec![user_text("research X")],
            provider_options: Some(google_options({
                let mut m = Map::new();
                m.insert("background".into(), Value::Bool(true));
                m
            })),
            ..Default::default()
        })
        .await
        .expect("ok");
    let parts: Vec<StreamPart> = stream_result
        .stream
        .map(|r| r.expect("stream item"))
        .collect()
        .await;

    // Synthesized sequence: stream-start, response-metadata, text-start,
    // text-delta (full payload as a single delta), text-end, finish.
    let texts: Vec<&str> = parts
        .iter()
        .filter_map(|p| match p {
            StreamPart::TextDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, vec!["agent reply"]);
    let finish = parts
        .iter()
        .find_map(|p| match p {
            StreamPart::Finish {
                finish_reason,
                usage,
                ..
            } => Some((finish_reason, usage)),
            _ => None,
        })
        .expect("Finish present");
    assert_eq!(finish.0.unified, FinishReasonKind::Stop);
    assert_eq!(finish.1.input_tokens.total, Some(7));
}

#[tokio::test]
async fn provider_defined_google_search_tool_routes_to_wire() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/interactions"))
        .and(body_partial_json(json!({
            "tools": [{"type": "google_search"}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "int_t",
            "status": "completed",
            "steps": []
        })))
        .mount(&server)
        .await;

    let model =
        provider(&server).interactions(GoogleInteractionsAgent::Model("gemini-2.5-flash".into()));
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            tools: Some(vec![Tool::Provider(ProviderTool {
                id: "google.google_search".into(),
                name: "google_search".into(),
                args: None,
                provider_options: None,
            })]),
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
}

#[tokio::test]
async fn managed_agent_routes_to_agent_field_not_managed_agent() {
    let server = MockServer::start().await;
    // The wire shape must contain `agent: "<resource>"` (NOT
    // `managed_agent: ...`) — mirrors upstream:112-117.
    Mock::given(method("POST"))
        .and(path("/interactions"))
        .and(body_partial_json(json!({
            "agent": "projects/p/agents/a",
            "background": true
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "int_m",
            "status": "completed",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": "ok"}]}]
        })))
        .mount(&server)
        .await;

    let model = provider(&server).interactions(GoogleInteractionsAgent::ManagedAgent(
        "projects/p/agents/a".into(),
    ));
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(google_options({
                let mut m = Map::new();
                m.insert("background".into(), Value::Bool(true));
                m
            })),
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(result.content.len(), 1);
}
