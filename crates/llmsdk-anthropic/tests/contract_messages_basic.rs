//! Contract tests for [`AnthropicMessagesModel::do_generate`].
// Rust guideline compliant 2026-02-21

use llmsdk_anthropic::Anthropic;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, FinishReasonKind, FunctionTool, Message, TextPart, Tool, ToolChoice,
    UserPart,
};
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Anthropic {
    Anthropic::builder()
        .api_key("sk-ant-test")
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
async fn happy_path_text_and_required_headers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", "sk-ant-test"))
        .and(header("anthropic-version", "2023-06-01"))
        .and(body_partial_json(json!({
            "model": "claude-3-5-sonnet-latest",
            "max_tokens": 64,
            "messages": [{ "role": "user", "content": [{ "type": "text", "text": "hi" }] }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1",
            "type": "message",
            "model": "claude-3-5-sonnet-latest",
            "content": [{ "type": "text", "text": "hello" }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 3, "output_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).messages("claude-3-5-sonnet-latest");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            max_output_tokens: Some(64),
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(result.content.len(), 1);
    if let Content::Text(t) = &result.content[0] {
        assert_eq!(t.text, "hello");
    } else {
        panic!("expected text");
    }
    assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
    assert_eq!(result.usage.input_tokens.total, Some(3));
    assert_eq!(result.usage.output_tokens.total, Some(1));
}

#[tokio::test]
async fn system_is_pulled_to_top_level() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "system": "Be brief.\n\nNo emojis."
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_2",
            "type": "message",
            "content": [{ "type": "text", "text": "ok" }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).messages("claude-3-5-sonnet-latest");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![
                Message::System {
                    content: "Be brief.".into(),
                    provider_options: None,
                },
                Message::System {
                    content: "No emojis.".into(),
                    provider_options: None,
                },
                user_text("hi"),
            ],
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn tool_use_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "tools": [{ "name": "get_weather" }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_3",
            "type": "message",
            "content": [{
                "type": "tool_use",
                "id": "tu_w",
                "name": "get_weather",
                "input": { "city": "NYC" }
            }],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 5, "output_tokens": 7 }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).messages("claude-3-5-sonnet-latest");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("weather?")],
            tools: Some(vec![Tool::Function(FunctionTool {
                name: "get_weather".into(),
                description: Some("Get weather".into()),
                input_schema: serde_json::from_value(json!({"type": "object"})).unwrap(),
                input_examples: None,
                strict: None,
                provider_options: None,
            })]),
            tool_choice: Some(ToolChoice::Required),
            ..Default::default()
        })
        .await
        .expect("ok");

    let Content::ToolCall(tc) = &result.content[0] else {
        panic!("expected tool call");
    };
    assert_eq!(tc.tool_call_id, "tu_w");
    assert_eq!(tc.tool_name, "get_weather");
    assert_eq!(tc.input["city"], "NYC");
    assert_eq!(result.finish_reason.unified, FinishReasonKind::ToolCalls);
}

#[tokio::test]
async fn http_429_with_anthropic_error_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "type": "error",
            "error": { "type": "rate_limit_error", "message": "rate limited" }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).messages("claude-3-5-sonnet-latest");
    let err = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect_err("should error");
    assert!(err.is_api_call());
    assert!(err.is_retryable());
    assert_eq!(err.status_code(), Some(429));
    let msg = format!("{err}");
    assert!(
        msg.contains("rate limited"),
        "expected upstream message in: {msg}"
    );
}

#[tokio::test]
async fn warns_on_unsupported_settings() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_5",
            "type": "message",
            "content": [{ "type": "text", "text": "ok" }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        })))
        .mount(&server)
        .await;
    let model = provider(&server).messages("claude-3-5-sonnet-latest");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            seed: Some(7),
            frequency_penalty: Some(0.5),
            ..Default::default()
        })
        .await
        .expect("ok");
    assert!(result.warnings.len() >= 2);
}

#[tokio::test]
async fn default_max_tokens_when_not_provided() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({ "max_tokens": 4096 })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_6",
            "type": "message",
            "content": [{ "type": "text", "text": "ok" }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        })))
        .mount(&server)
        .await;
    let model = provider(&server).messages("claude-3-5-sonnet-latest");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");
}
