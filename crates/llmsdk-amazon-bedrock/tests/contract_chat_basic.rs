//! Contract tests for the Converse API (non-streaming happy path).
// Rust guideline compliant 2026-05-25

use llmsdk_amazon_bedrock::AmazonBedrock;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, FinishReasonKind, Message, TextPart, UserPart,
};
use serde_json::json;
use wiremock::matchers::{body_partial_json, header_exists, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> AmazonBedrock {
    AmazonBedrock::builder()
        .region("us-east-1")
        .api_key("bedrock-test-token")
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
async fn happy_path_text_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(
            "/model/anthropic.claude-3-5-haiku-20241022-v1%3A0/converse",
        ))
        .and(header_exists("authorization"))
        .and(body_partial_json(json!({
            "messages": [{
                "role": "user",
                "content": [{ "text": "hi" }]
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {
                "message": {
                    "content": [{ "text": "hello from bedrock" }],
                    "role": "assistant"
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 5,
                "outputTokens": 3,
                "totalTokens": 8
            }
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.language_model("anthropic.claude-3-5-haiku-20241022-v1:0");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            max_output_tokens: Some(128),
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.content.len(), 1);
    let Content::Text(t) = &result.content[0] else {
        panic!("expected text content");
    };
    assert_eq!(t.text, "hello from bedrock");
    assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
    assert_eq!(result.usage.input_tokens.no_cache, Some(5));
    assert_eq!(result.usage.output_tokens.total, Some(3));
}

#[tokio::test]
async fn http_error_propagates_with_status() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "message": "bad request"
        })))
        .mount(&server)
        .await;

    let err = provider(&server)
        .language_model("anthropic.claude-3-5-haiku-20241022-v1:0")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect_err("400 must propagate");
    let msg = format!("{err}");
    assert!(msg.contains("400"), "expected 400 in error: {msg}");
}

#[tokio::test]
async fn reasoning_block_is_surfaced_as_content_reasoning() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {
                "message": {
                    "content": [
                        {
                            "reasoningContent": {
                                "reasoningText": {
                                    "text": "let me think",
                                    "signature": "sig-abc"
                                }
                            }
                        },
                        { "text": "answer" }
                    ],
                    "role": "assistant"
                }
            },
            "stopReason": "end_turn"
        })))
        .mount(&server)
        .await;

    let result = provider(&server)
        .language_model("anthropic.claude-opus-4-5-20251101-v1:0")
        .do_generate(CallOptions {
            prompt: vec![user_text("question?")],
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(result.content.len(), 2);
    assert!(matches!(result.content[0], Content::Reasoning(_)));
    assert!(matches!(result.content[1], Content::Text(_)));
}

#[tokio::test]
async fn tool_use_response_becomes_tool_call() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {
                "message": {
                    "content": [{
                        "toolUse": {
                            "toolUseId": "tu-1",
                            "name": "weather",
                            "input": { "city": "NYC" }
                        }
                    }],
                    "role": "assistant"
                }
            },
            "stopReason": "tool_use"
        })))
        .mount(&server)
        .await;

    let result = provider(&server)
        .language_model("anthropic.claude-3-5-haiku-20241022-v1:0")
        .do_generate(CallOptions {
            prompt: vec![user_text("weather in NYC?")],
            ..Default::default()
        })
        .await
        .expect("ok");
    let Content::ToolCall(tc) = &result.content[0] else {
        panic!("expected tool call");
    };
    assert_eq!(tc.tool_call_id, "tu-1");
    assert_eq!(tc.tool_name, "weather");
    assert_eq!(tc.input["city"], "NYC");
    assert_eq!(result.finish_reason.unified, FinishReasonKind::ToolCalls);
}
