//! Contract tests for Mistral-specific advanced behaviour:
//! - assistant `prefix: true` on trailing assistant messages
//! - `image_url` vs `document_url` routing for PDF / image attachments
//! - cached-tokens usage breakdown
// Rust guideline compliant 2026-05-25

use llmsdk_mistral::Mistral;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    AssistantPart, CallOptions, FilePart, Message, TextPart, UserPart,
};
use llmsdk_provider::shared::FileData;
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Mistral {
    Mistral::builder()
        .api_key("test-api-key")
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

fn ok_response() -> serde_json::Value {
    json!({
        "id": "r1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    })
}

#[tokio::test]
async fn trailing_assistant_message_has_prefix_true() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "messages": [
                { "role": "user", "content": [{"type":"text","text":"hi"}] },
                {
                    "role": "assistant",
                    "content": "continue this",
                    "prefix": true
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let _ = provider
        .chat("mistral-small-latest")
        .do_generate(CallOptions {
            prompt: vec![
                user_text("hi"),
                Message::Assistant {
                    content: vec![AssistantPart::Text(TextPart {
                        text: "continue this".into(),
                        provider_options: None,
                    })],
                    provider_options: None,
                },
            ],
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn pdf_url_routes_to_document_url() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "document_url",
                    "document_url": "https://example.com/file.pdf"
                }]
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let _ = provider
        .chat("mistral-small-latest")
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserPart::File(FilePart {
                    filename: None,
                    data: FileData::Url {
                        url: "https://example.com/file.pdf".into(),
                    },
                    media_type: "application/pdf".into(),
                    provider_options: None,
                })],
                provider_options: None,
            }],
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn image_url_routes_to_image_url() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": "https://example.com/a.png"
                }]
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let _ = provider
        .chat("pixtral-large-latest")
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserPart::File(FilePart {
                    filename: None,
                    data: FileData::Url {
                        url: "https://example.com/a.png".into(),
                    },
                    media_type: "image/png".into(),
                    provider_options: None,
                })],
                provider_options: None,
            }],
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn cached_tokens_split_via_num_cached_tokens() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "r1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hi" },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "num_cached_tokens": 60
            }
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let result = provider
        .chat("mistral-small-latest")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.usage.input_tokens.total, Some(100));
    assert_eq!(result.usage.input_tokens.cache_read, Some(60));
    assert_eq!(result.usage.input_tokens.no_cache, Some(40));
    assert_eq!(result.usage.output_tokens.text, Some(10));
}

#[tokio::test]
async fn cached_tokens_split_via_prompt_tokens_details() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "r1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hi" },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 5,
                "total_tokens": 55,
                "prompt_tokens_details": { "cached_tokens": 20 }
            }
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let result = provider
        .chat("mistral-small-latest")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.usage.input_tokens.cache_read, Some(20));
    assert_eq!(result.usage.input_tokens.no_cache, Some(30));
}
