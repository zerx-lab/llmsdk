//! Contract tests covering tool-related wire shapes on the chat endpoint.
// Rust guideline compliant 2026-05-25

use llmsdk_cohere::Cohere;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, FunctionTool, Message, TextPart, Tool, ToolChoice, UserPart,
};
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Cohere {
    Cohere::builder()
        .api_key("k")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
}

fn user(text: &str) -> Message {
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
            "properties": {"city": {"type": "string"}}
        }))
        .unwrap(),
        input_examples: None,
        strict: None,
        provider_options: None,
    })
}

#[tokio::test]
async fn sends_tools_with_required_choice() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat"))
        .and(body_partial_json(json!({
            "tools": [{
                "type": "function",
                "function": { "name": "weather" }
            }],
            "tool_choice": "REQUIRED"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "generation_id": "g",
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": { "name": "weather", "arguments": "{\"city\":\"NYC\"}" }
                }]
            },
            "finish_reason": "TOOL_CALL"
        })))
        .mount(&server)
        .await;

    let result = provider(&server)
        .chat("command-a-03-2025")
        .do_generate(CallOptions {
            prompt: vec![user("weather?")],
            tools: Some(vec![weather_tool()]),
            tool_choice: Some(ToolChoice::Required),
            ..Default::default()
        })
        .await
        .expect("ok");

    let Content::ToolCall(tc) = &result.content[0] else {
        panic!("expected tool call");
    };
    assert_eq!(tc.tool_call_id, "c1");
    assert_eq!(tc.tool_name, "weather");
}

#[tokio::test]
async fn specific_tool_choice_filters_and_requires() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat"))
        .and(body_partial_json(json!({
            "tool_choice": "REQUIRED",
            "tools": [{ "function": { "name": "weather" } }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "generation_id": "g",
            "message": { "role": "assistant", "content": [] },
            "finish_reason": "COMPLETE"
        })))
        .mount(&server)
        .await;

    let _ = provider(&server)
        .chat("command-a-03-2025")
        .do_generate(CallOptions {
            prompt: vec![user("weather?")],
            tools: Some(vec![
                weather_tool(),
                Tool::Function(FunctionTool {
                    name: "ignored".into(),
                    description: None,
                    input_schema: serde_json::from_value(json!({"type": "object"})).unwrap(),
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
}

#[tokio::test]
async fn empty_tool_arguments_become_empty_object() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "generation_id": "g",
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "c2",
                    "type": "function",
                    "function": { "name": "ping", "arguments": "null" }
                }]
            },
            "finish_reason": "TOOL_CALL"
        })))
        .mount(&server)
        .await;

    let result = provider(&server)
        .chat("command-a-03-2025")
        .do_generate(CallOptions {
            prompt: vec![user("ping?")],
            tools: Some(vec![weather_tool()]),
            ..Default::default()
        })
        .await
        .expect("ok");

    let Content::ToolCall(tc) = &result.content[0] else {
        panic!("expected tool call");
    };
    assert!(tc.input.is_object());
}
