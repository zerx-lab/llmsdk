//! Contract tests for [`OpenAiResponsesLanguageModel::do_generate`].
// Rust guideline compliant 2026-02-21

use llmsdk_openai::OpenAi;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, FinishReasonKind, Message, TextPart, UserPart,
};
use llmsdk_provider::shared::ProviderOptions;
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
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

fn happy_response() -> serde_json::Value {
    json!({
        "id": "resp_1",
        "created_at": 1_700_000_000_u64,
        "model": "gpt-4o-mini",
        "output": [{
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "hello",
                "annotations": []
            }]
        }],
        "usage": { "input_tokens": 5, "output_tokens": 1 }
    })
}

#[tokio::test]
async fn happy_path_returns_text_and_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer sk-test"))
        .and(body_partial_json(json!({
            "model": "gpt-4o-mini",
            "input": [{
                "role": "user",
                "content": [{"type": "input_text", "text": "ping"}]
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(happy_response()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.responses("gpt-4o-mini");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("ping")],
            ..Default::default()
        })
        .await
        .expect("call succeeds");

    assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
    assert_eq!(result.usage.input_tokens.total, Some(5));
    let Content::Text(t) = &result.content[0] else {
        panic!("expected text content");
    };
    assert_eq!(t.text, "hello");
}

#[tokio::test]
async fn reasoning_effort_routes_into_reasoning_block() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "model": "o3-mini",
            "reasoning": { "effort": "high" }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(happy_response()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.responses("o3-mini");
    let mut po = ProviderOptions::new();
    po.insert(
        "openai".into(),
        json!({"reasoningEffort": "high"})
            .as_object()
            .unwrap()
            .clone(),
    );
    let _result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("explain quantum")],
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("call");
}

#[tokio::test]
async fn temperature_stripped_on_reasoning_model_with_warning() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(happy_response()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.responses("o3-mini");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            temperature: Some(0.7),
            ..Default::default()
        })
        .await
        .expect("call");
    assert!(result.warnings.iter().any(|w| matches!(
        w,
        llmsdk_provider::shared::Warning::UnsupportedSetting { setting, .. } if setting == "temperature"
    )));
}

#[tokio::test]
async fn error_body_maps_to_api_call_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_e",
            "error": {
                "message": "blocked",
                "type": "policy_violation",
                "code": "content_blocked"
            }
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.responses("gpt-4o-mini");
    let err = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect_err("should error");
    assert!(err.to_string().to_lowercase().contains("blocked"));
}
