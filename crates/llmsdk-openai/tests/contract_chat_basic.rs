//! Contract tests for [`OpenAiChatModel::do_generate`].
//!
//! Each test boots a `wiremock` server, points the provider at it, and
//! asserts the request shape + response mapping.
// Rust guideline compliant 2026-02-21

use llmsdk_openai::OpenAi;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, FinishReasonKind, FunctionTool, Message, TextPart, Tool, ToolChoice,
    UserPart,
};
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> OpenAi {
    OpenAi::builder()
        .api_key("sk-test")
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

fn happy_response() -> serde_json::Value {
    json!({
        "id": "chatcmpl-1",
        "created": 1_700_000_000_u64,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "hello" },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "total_tokens": 6
        }
    })
}

#[tokio::test]
async fn happy_path_text() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer sk-test"))
        .and(body_partial_json(json!({
            "model": "gpt-4o-mini",
            "messages": [{ "role": "user", "content": "hi" }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(happy_response()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.chat("gpt-4o-mini");

    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
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
    assert_eq!(result.usage.input_tokens.total, Some(5));
    assert!(result.warnings.is_empty());
    assert_eq!(
        result
            .response
            .as_ref()
            .and_then(|r| r.metadata.id.as_deref()),
        Some("chatcmpl-1")
    );
}

#[tokio::test]
async fn tool_call_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-2",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_w",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"NYC\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.chat("gpt-4o-mini");

    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("weather?")],
            tools: Some(vec![Tool::Function(FunctionTool {
                name: "get_weather".into(),
                description: Some("Get weather".into()),
                input_schema: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
                input_examples: None,
                strict: Some(true),
                provider_options: None,
            })]),
            tool_choice: Some(ToolChoice::Auto),
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.content.len(), 1);
    let Content::ToolCall(tc) = &result.content[0] else {
        panic!("expected tool call");
    };
    assert_eq!(tc.tool_call_id, "call_w");
    assert_eq!(tc.tool_name, "get_weather");
    assert_eq!(tc.input["city"], "NYC");
    assert_eq!(result.finish_reason.unified, FinishReasonKind::ToolCalls);
}

#[tokio::test]
async fn maps_openai_error_envelope() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": {
                "message": "Rate limit exceeded on requests",
                "type": "requests",
                "code": "rate_limit_exceeded"
            }
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.chat("gpt-4o-mini");

    let err = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect_err("should error");

    assert!(err.is_api_call());
    assert!(err.is_retryable(), "429 retryable");
    assert_eq!(err.status_code(), Some(429));
    let msg = format!("{err}");
    assert!(
        msg.contains("Rate limit exceeded on requests"),
        "expected upstream message, got: {msg}"
    );
}

#[tokio::test]
async fn http_400_not_retryable() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(400)
                .set_body_json(json!({ "error": { "message": "bad input" } })),
        )
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.chat("gpt-4o-mini");

    let err = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect_err("should error");
    assert!(!err.is_retryable());
    assert_eq!(err.status_code(), Some(400));
    assert!(format!("{err}").contains("bad input"));
}

#[tokio::test]
async fn empty_choices_yields_no_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": "x", "choices": [] })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.chat("gpt-4o-mini");
    let err = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect_err("should error");
    assert!(!err.is_api_call());
    assert!(format!("{err}").contains("no content"));
}

#[tokio::test]
async fn top_k_emits_warning_but_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(happy_response()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.chat("gpt-4o-mini");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            top_k: Some(40),
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(result.warnings.len(), 1);
}

#[tokio::test]
async fn sends_per_call_header_override() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("x-llmsdk-trace", "abc"))
        .respond_with(ResponseTemplate::new(200).set_body_json(happy_response()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.chat("gpt-4o-mini");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            headers: Some(
                [("x-llmsdk-trace".to_owned(), Some("abc".to_owned()))]
                    .into_iter()
                    .collect(),
            ),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn do_stream_returns_unsupported() {
    let server = MockServer::start().await;
    let provider = provider(&server);
    let model = provider.chat("gpt-4o-mini");
    match model
        .do_stream(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
    {
        Err(e) => assert!(e.is_unsupported()),
        Ok(_) => panic!("M3 must not stream"),
    }
}
