//! Contract tests for xAI Chat provider-options pass-through.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{CallOptions, Message, ReasoningEffort, TextPart, UserPart};
use llmsdk_provider::shared::ProviderOptions;
use llmsdk_xai::Xai;
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Xai {
    Xai::builder()
        .api_key("xai-test")
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
        "id": "r1",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop"
        }]
    })
}

fn xai_options(map: &serde_json::Value) -> ProviderOptions {
    let mut po = ProviderOptions::new();
    po.insert("xai".into(), map.as_object().cloned().unwrap());
    po
}

#[tokio::test]
async fn reasoning_effort_provider_option_overrides_top_level() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({ "reasoning_effort": "high" })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let _ = provider(&server)
        .chat("grok-4.20-reasoning")
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            reasoning: Some(ReasoningEffort::Low),
            provider_options: Some(xai_options(&json!({ "reasoningEffort": "high" }))),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn search_parameters_serialized_to_wire() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "search_parameters": {
                "mode": "auto",
                "return_citations": true,
                "max_search_results": 10,
                "sources": [
                    { "type": "web", "country": "US", "safe_search": true },
                    { "type": "x", "included_x_handles": ["@elon"] },
                    { "type": "news", "country": "GB" },
                    { "type": "rss", "links": ["https://example.com/feed.xml"] }
                ]
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let _ = provider(&server)
        .chat("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            provider_options: Some(xai_options(&json!({
                "searchParameters": {
                    "mode": "auto",
                    "returnCitations": true,
                    "maxSearchResults": 10,
                    "sources": [
                        { "type": "web", "country": "US", "safeSearch": true },
                        { "type": "x", "includedXHandles": ["@elon"] },
                        { "type": "news", "country": "GB" },
                        { "type": "rss", "links": ["https://example.com/feed.xml"] }
                    ]
                }
            }))),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn top_logprobs_forces_logprobs_true() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "logprobs": true,
            "top_logprobs": 5
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let _ = provider(&server)
        .chat("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            provider_options: Some(xai_options(&json!({ "topLogprobs": 5 }))),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn parallel_function_calling_passthrough() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(
            json!({ "parallel_function_calling": false }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let _ = provider(&server)
        .chat("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            provider_options: Some(xai_options(&json!({
                "parallel_function_calling": false
            }))),
            ..Default::default()
        })
        .await
        .expect("ok");
}
