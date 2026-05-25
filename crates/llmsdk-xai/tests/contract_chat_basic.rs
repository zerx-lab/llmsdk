//! Contract tests for [`XaiChatModel::do_generate`].
// Rust guideline compliant 2026-05-25

use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, FinishReasonKind, Message, TextPart, UserPart,
};
use llmsdk_xai::Xai;
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Xai {
    Xai::builder()
        .api_key("xai-test")
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
async fn happy_path_text() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer xai-test"))
        .and(body_partial_json(json!({
            "model": "grok-4.3",
            "messages": [{ "role": "user", "content": "hi" }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1",
            "created": 1_700_000_000_u64,
            "model": "grok-4.3",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hello from grok" },
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
    let model = provider.chat("grok-4.3");

    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.content.len(), 1);
    let Content::Text(t) = &result.content[0] else {
        panic!("expected text");
    };
    assert_eq!(t.text, "hello from grok");
    assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
    assert_eq!(result.usage.input_tokens.total, Some(5));
    assert_eq!(result.usage.output_tokens.total, Some(3));
    assert!(result.warnings.is_empty());
}

#[tokio::test]
async fn reasoning_content_becomes_reasoning_block() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "r1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "42",
                    "reasoning_content": "Let me think..."
                },
                "finish_reason": "stop"
            }]
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let result = provider
        .chat("grok-4.20-reasoning")
        .do_generate(CallOptions {
            prompt: vec![user_text("what is 6 * 7?")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.content.len(), 2);
    assert!(matches!(result.content[0], Content::Text(_)));
    assert!(matches!(result.content[1], Content::Reasoning(_)));
}

#[tokio::test]
async fn citations_become_source_parts() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "r1",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "answer" },
                "finish_reason": "stop"
            }],
            "citations": [
                "https://example.com/a",
                "https://example.com/b"
            ]
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let result = provider
        .chat("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.content.len(), 3);
    let Content::Source(_) = &result.content[1] else {
        panic!("expected source");
    };
    let Content::Source(_) = &result.content[2] else {
        panic!("expected source");
    };
}

#[tokio::test]
async fn error_envelope_returns_api_call_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "code": "rate_limit_exceeded",
            "error": "rate limited",
            "choices": []
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let err = provider
        .chat("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect_err("must error");
    assert!(format!("{err}").contains("rate limited"));
}
