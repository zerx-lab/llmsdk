//! Contract tests for non-streaming Gemini language model.
// Rust guideline compliant 2026-05-25

use llmsdk_google::Google;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, FinishReasonKind, Message, TextPart, UserPart,
};
use serde_json::json;
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

#[tokio::test]
async fn happy_path_text() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:generateContent"))
        .and(header("x-goog-api-key", "test-key"))
        .and(body_partial_json(json!({
            "contents": [{ "role": "user", "parts": [{"text":"hi"}] }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": { "role": "model", "parts": [{"text": "hello from gemini"}] },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 3,
                "totalTokenCount": 8
            }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).chat("gemini-2.5-flash");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.content.len(), 1);
    let Content::Text(t) = &result.content[0] else {
        panic!("expected text content");
    };
    assert_eq!(t.text, "hello from gemini");
    assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
    assert_eq!(result.usage.input_tokens.total, Some(5));
    assert_eq!(result.usage.output_tokens.total, Some(3));
}

#[tokio::test]
async fn system_instruction_routed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:generateContent"))
        .and(body_partial_json(json!({
            "systemInstruction": { "parts": [{"text": "You are concise"}] }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": { "role": "model", "parts": [{"text": "ok"}] },
                "finishReason": "STOP"
            }]
        })))
        .mount(&server)
        .await;

    let model = provider(&server).chat("gemini-2.5-flash");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![
                Message::System {
                    content: "You are concise".into(),
                    provider_options: None,
                },
                user_text("hi"),
            ],
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(result.content.len(), 1);
}

#[tokio::test]
async fn finish_reason_safety_maps_to_content_filter() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": { "role": "model", "parts": [] },
                "finishReason": "SAFETY"
            }]
        })))
        .mount(&server)
        .await;

    let model = provider(&server).chat("gemini-2.5-flash");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(
        result.finish_reason.unified,
        FinishReasonKind::ContentFilter
    );
}

#[tokio::test]
async fn http_error_envelope_rewritten() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:generateContent"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": {
                "code": 400,
                "message": "Invalid argument: bad input",
                "status": "INVALID_ARGUMENT"
            }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).chat("gemini-2.5-flash");
    let err = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect_err("should error");
    let msg = format!("{err}");
    assert!(msg.contains("Invalid argument"), "got: {msg}");
}

#[tokio::test]
async fn cache_read_input_tokens_set() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": { "role": "model", "parts": [{"text":"ok"}] },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 10,
                "cachedContentTokenCount": 80,
                "thoughtsTokenCount": 5,
                "totalTokenCount": 115
            }
        })))
        .mount(&server)
        .await;
    let model = provider(&server).chat("gemini-2.5-flash");
    let r = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(r.usage.input_tokens.total, Some(100));
    assert_eq!(r.usage.input_tokens.cache_read, Some(80));
    assert_eq!(r.usage.input_tokens.no_cache, Some(20));
    assert_eq!(r.usage.output_tokens.reasoning, Some(5));
    assert_eq!(r.usage.output_tokens.text, Some(10));
    assert_eq!(r.usage.output_tokens.total, Some(15));
}
