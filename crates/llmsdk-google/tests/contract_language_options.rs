//! Provider options + thinking budget contract tests.
// Rust guideline compliant 2026-05-25

use llmsdk_google::Google;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{CallOptions, Message, ReasoningEffort, TextPart, UserPart};
use llmsdk_provider::shared::ProviderOptions;
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Google {
    Google::builder()
        .api_key("k")
        .base_url(server.uri())
        .build()
        .expect("ok")
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

#[tokio::test]
async fn safety_settings_passthrough() {
    let server = MockServer::start().await;
    let mut opts = ProviderOptions::new();
    opts.insert(
        "google".into(),
        json!({
            "safetySettings":[{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_LOW_AND_ABOVE"}]
        })
        .as_object().unwrap().clone(),
    );
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:generateContent"))
        .and(body_partial_json(json!({
            "safetySettings":[{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_LOW_AND_ABOVE"}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]
        })))
        .mount(&server).await;
    let model = provider(&server).chat("gemini-2.5-flash");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(opts),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn reasoning_thinking_budget_set_for_25() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:generateContent"))
        .and(body_partial_json(json!({
            "generationConfig": { "thinkingConfig": { "thinkingBudget": 8192 } }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]
        })))
        .mount(&server).await;
    let model = provider(&server).chat("gemini-2.5-flash");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            reasoning: Some(ReasoningEffort::Medium),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn cached_content_passthrough() {
    let server = MockServer::start().await;
    let mut opts = ProviderOptions::new();
    opts.insert(
        "google".into(),
        json!({"cachedContent":"cachedContents/abc"})
            .as_object()
            .unwrap()
            .clone(),
    );
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:generateContent"))
        .and(body_partial_json(json!({
            "cachedContent": "cachedContents/abc"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]
        })))
        .mount(&server)
        .await;
    let model = provider(&server).chat("gemini-2.5-flash");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(opts),
            ..Default::default()
        })
        .await
        .expect("ok");
}
