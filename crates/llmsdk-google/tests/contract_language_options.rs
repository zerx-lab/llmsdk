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
async fn shared_request_type_ignored_on_google_provider_with_warning() {
    let server = MockServer::start().await;
    let mut opts = ProviderOptions::new();
    opts.insert(
        "google".into(),
        json!({"sharedRequestType": "flex", "requestType": "shared"})
            .as_object()
            .unwrap()
            .clone(),
    );
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]
        })))
        .mount(&server)
        .await;
    let model = provider(&server).chat("gemini-2.5-flash");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(opts),
            ..Default::default()
        })
        .await
        .expect("ok");
    // The Google (non-Vertex) provider must ignore these Vertex-only options
    // with a warning rather than injecting the paygo headers.
    assert!(
        result
            .warnings
            .iter()
            .any(|w| format!("{w:?}").contains("sharedRequestType")),
        "expected a warning about sharedRequestType/requestType being ignored on the Google provider, got {:?}",
        result.warnings,
    );
    // The captured request on the mock server should *not* carry the paygo
    // headers — verified by inspecting recorded requests.
    let requests = server
        .received_requests()
        .await
        .expect("captured requests available");
    let last = requests.last().expect("at least one request received");
    assert!(
        last.headers
            .get("x-vertex-ai-llm-shared-request-type")
            .is_none(),
        "Google provider must not inject Vertex paygo headers"
    );
    assert!(
        last.headers.get("x-vertex-ai-llm-request-type").is_none(),
        "Google provider must not inject Vertex paygo headers"
    );
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
