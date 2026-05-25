//! Contract tests for [`MistralChatModel::do_generate`] — happy paths.
// Rust guideline compliant 2026-05-25

use llmsdk_mistral::Mistral;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, FinishReasonKind, Message, TextPart, UserPart,
};
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
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

#[tokio::test]
async fn happy_path_string_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-api-key"))
        .and(body_partial_json(json!({
            "model": "mistral-small-latest",
            "messages": [{
                "role": "user",
                "content": [{ "type": "text", "text": "Hello" }]
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "5319bd0299614c679a0068a4f2c8ffd0",
            "created": 1_769_088_720_u64,
            "model": "mistral-small-latest",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hello from mistral" },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8
            }
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.chat("mistral-small-latest");

    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("Hello")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.content.len(), 1);
    let Content::Text(t) = &result.content[0] else {
        panic!("expected text");
    };
    assert_eq!(t.text, "hello from mistral");
    assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
    assert_eq!(result.usage.input_tokens.total, Some(5));
    assert_eq!(result.usage.output_tokens.total, Some(3));
    assert!(result.warnings.is_empty());
}

#[tokio::test]
async fn thinking_part_becomes_reasoning_block() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "r1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [
                                { "type": "text", "text": "Let me think..." }
                            ]
                        },
                        { "type": "text", "text": "42" }
                    ]
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 4,
                "total_tokens": 6
            }
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let result = provider
        .chat("magistral-medium-latest")
        .do_generate(CallOptions {
            prompt: vec![user_text("what is 6 * 7?")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.content.len(), 2);
    assert!(matches!(result.content[0], Content::Reasoning(_)));
    assert!(matches!(result.content[1], Content::Text(_)));
}

#[tokio::test]
async fn http_error_yields_provider_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "object": "error",
            "message": "rate limited",
            "type": "rate_limit_exceeded",
            "param": null,
            "code": "rate_limit_exceeded"
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let err = provider
        .chat("mistral-small-latest")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect_err("must error");
    assert!(format!("{err}").contains("429"));
}

#[tokio::test]
async fn custom_request_header_forwarded() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("x-trace-id", "abc123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "r1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "ok" },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2
            }
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let mut headers = llmsdk_provider::shared::Headers::new();
    headers.insert("x-trace-id".into(), Some("abc123".into()));
    let _ = provider
        .chat("mistral-small-latest")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            headers: Some(headers),
            ..Default::default()
        })
        .await
        .expect("ok");
}
