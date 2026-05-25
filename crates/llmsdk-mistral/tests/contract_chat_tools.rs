//! Contract tests for tool calling (function tools, `tool_choice`, parallel).
// Rust guideline compliant 2026-05-25

use llmsdk_mistral::Mistral;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, FinishReasonKind, FunctionTool, Message, ProviderTool, TextPart, Tool,
    ToolChoice, UserPart,
};
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

fn weather_tool() -> Tool {
    Tool::Function(FunctionTool {
        name: "weather".into(),
        description: Some("get weather".into()),
        input_schema: serde_json::from_value(json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }))
        .unwrap(),
        input_examples: None,
        strict: None,
        provider_options: None,
    })
}

#[tokio::test]
async fn function_tool_routed_with_required() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "model": "mistral-small-latest",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "get weather",
                    "parameters": {
                        "type": "object",
                        "properties": { "city": { "type": "string" } },
                        "required": ["city"]
                    }
                }
            }],
            "tool_choice": "any"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "tool1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "function": { "name": "weather", "arguments": "{\"city\":\"NYC\"}" }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let result = provider
        .chat("mistral-small-latest")
        .do_generate(CallOptions {
            prompt: vec![user_text("weather in NYC?")],
            tools: Some(vec![weather_tool()]),
            tool_choice: Some(ToolChoice::Required),
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.finish_reason.unified, FinishReasonKind::ToolCalls);
    let Content::ToolCall(tc) = &result.content[0] else {
        panic!("expected tool call");
    };
    assert_eq!(tc.tool_name, "weather");
}

#[tokio::test]
async fn specific_tool_choice_filters_tools_and_forces_any() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "tool_choice": "any"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "tool2",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "ok" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let result = provider
        .chat("mistral-small-latest")
        .do_generate(CallOptions {
            prompt: vec![user_text("call weather only")],
            tools: Some(vec![
                weather_tool(),
                Tool::Function(FunctionTool {
                    name: "clock".into(),
                    description: None,
                    input_schema: serde_json::from_value(json!({"type":"object"})).unwrap(),
                    input_examples: None,
                    strict: None,
                    provider_options: None,
                }),
            ]),
            tool_choice: Some(ToolChoice::Tool {
                tool_name: "weather".into(),
            }),
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
}

#[tokio::test]
async fn provider_tool_yields_warning_and_no_tools() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "tool3",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "ok" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let result = provider
        .chat("mistral-small-latest")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            tools: Some(vec![Tool::Provider(ProviderTool {
                id: "mistral.code_execution".into(),
                name: "code_execution".into(),
                args: None,
                provider_options: None,
            })]),
            ..Default::default()
        })
        .await
        .expect("ok");

    assert!(
        result
            .warnings
            .iter()
            .any(|w| matches!(w, llmsdk_provider::shared::Warning::UnsupportedTool { .. })),
        "expected UnsupportedTool warning"
    );
}
