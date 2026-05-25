//! Contract tests for [`CohereChatModel::do_generate`].
// Rust guideline compliant 2026-05-25

use llmsdk_cohere::Cohere;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, FinishReasonKind, Message, TextPart, UserPart,
};
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Cohere {
    Cohere::builder()
        .api_key("cohere-test")
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
        .and(path("/chat"))
        .and(header("authorization", "Bearer cohere-test"))
        .and(body_partial_json(json!({
            "model": "command-a-03-2025",
            "messages": [{ "role": "user", "content": "hi" }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "generation_id": "gen-1",
            "message": {
                "role": "assistant",
                "content": [
                    { "type": "text", "text": "hello from cohere" }
                ]
            },
            "finish_reason": "COMPLETE",
            "usage": {
                "billed_units": { "input_tokens": 5, "output_tokens": 3 },
                "tokens": { "input_tokens": 5, "output_tokens": 3 }
            }
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.chat("command-a-03-2025");

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
    assert_eq!(t.text, "hello from cohere");
    assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
    assert_eq!(result.usage.input_tokens.total, Some(5));
    assert_eq!(result.usage.output_tokens.total, Some(3));
    assert!(result.warnings.is_empty());
}

#[tokio::test]
async fn thinking_becomes_reasoning_block() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "generation_id": "g2",
            "message": {
                "role": "assistant",
                "content": [
                    { "type": "thinking", "thinking": "let me think" },
                    { "type": "text", "text": "42" }
                ]
            },
            "finish_reason": "COMPLETE"
        })))
        .mount(&server)
        .await;

    let result = provider(&server)
        .chat("command-a-reasoning-08-2025")
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
async fn citations_become_document_sources() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "generation_id": "g3",
            "message": {
                "role": "assistant",
                "content": [{ "type": "text", "text": "the answer" }],
                "citations": [
                    {
                        "start": 0,
                        "end": 3,
                        "text": "the",
                        "type": "inline",
                        "sources": [{
                            "type": "document",
                            "id": "d0",
                            "document": {
                                "id": "d0",
                                "text": "the",
                                "title": "Article A"
                            }
                        }]
                    }
                ]
            },
            "finish_reason": "COMPLETE"
        })))
        .mount(&server)
        .await;

    let result = provider(&server)
        .chat("command-a-03-2025")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(result.content.len(), 2);
    let Content::Source(llmsdk_provider::language_model::Source::Document { title, .. }) =
        &result.content[1]
    else {
        panic!("expected document source");
    };
    assert_eq!(title, "Article A");
}

#[tokio::test]
async fn tool_plan_lands_in_provider_metadata() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "generation_id": "g4",
            "message": {
                "role": "assistant",
                "tool_plan": "I will call weather",
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": { "name": "weather", "arguments": "{}" }
                }]
            },
            "finish_reason": "TOOL_CALL"
        })))
        .mount(&server)
        .await;

    let result = provider(&server)
        .chat("command-a-03-2025")
        .do_generate(CallOptions {
            prompt: vec![user_text("weather?")],
            ..Default::default()
        })
        .await
        .expect("ok");
    let meta = result.provider_metadata.unwrap();
    assert_eq!(meta["cohere"]["toolPlan"], "I will call weather");
}

#[tokio::test]
async fn http_error_propagates() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({"message": "bad request"})))
        .mount(&server)
        .await;

    let err = provider(&server)
        .chat("command-a-03-2025")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect_err("must error");
    assert!(format!("{err}").contains("400"));
}
