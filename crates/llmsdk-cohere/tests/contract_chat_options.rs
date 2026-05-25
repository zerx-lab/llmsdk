//! Contract tests for provider-options + sampling-knob mapping on the chat
//! endpoint.
// Rust guideline compliant 2026-05-25

use llmsdk_cohere::Cohere;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{CallOptions, Message, TextPart, UserPart};
use llmsdk_provider::shared::ProviderOptions;
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Cohere {
    Cohere::builder()
        .api_key("k")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
}

fn user(text: &str) -> Message {
    Message::User {
        content: vec![UserPart::Text(TextPart {
            text: text.into(),
            provider_options: None,
        })],
        provider_options: None,
    }
}

fn ok_response() -> serde_json::Value {
    json!({
        "generation_id": "g",
        "message": {
            "role": "assistant",
            "content": [{ "type": "text", "text": "ok" }]
        },
        "finish_reason": "COMPLETE"
    })
}

#[tokio::test]
async fn maps_sampling_knobs_to_cohere_field_names() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat"))
        .and(body_partial_json(json!({
            "max_tokens": 100,
            "temperature": 0.5,
            "p": 0.8,
            "k": 5,
            "seed": 42,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "stop_sequences": ["END"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let _ = provider(&server)
        .chat("command-a-03-2025")
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            max_output_tokens: Some(100),
            temperature: Some(0.5),
            top_p: Some(0.8),
            top_k: Some(5),
            seed: Some(42),
            frequency_penalty: Some(0.1),
            presence_penalty: Some(0.2),
            stop_sequences: Some(vec!["END".into()]),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn forwards_thinking_provider_option() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat"))
        .and(body_partial_json(json!({
            "thinking": { "type": "enabled", "token_budget": 2048 }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let mut po = ProviderOptions::new();
    po.insert(
        "cohere".into(),
        json!({"thinking": {"type": "enabled", "tokenBudget": 2048}})
            .as_object()
            .cloned()
            .unwrap(),
    );

    let _ = provider(&server)
        .chat("command-a-reasoning-08-2025")
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("ok");
}
