//! Contract tests for xAI chat tool calling.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, FinishReasonKind, FunctionTool, Message, TextPart, Tool, ToolChoice,
    UserPart,
};
use llmsdk_xai::Xai;
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Xai {
    Xai::builder()
        .api_key("xai-test")
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
        name: "get_weather".into(),
        description: Some("get weather for a city".into()),
        input_schema: serde_json::from_value(json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
            "additionalProperties": false
        }))
        .unwrap(),
        input_examples: None,
        strict: Some(true),
        provider_options: None,
    })
}

#[tokio::test]
async fn tool_call_returned_via_function() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        // additionalProperties=false must be stripped from the request body.
        .and(body_partial_json(json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "strict": true
                }
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "r1",
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

    let result = provider(&server)
        .chat("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user("weather?")],
            tools: Some(vec![weather_tool()]),
            tool_choice: Some(ToolChoice::Auto),
            ..Default::default()
        })
        .await
        .expect("ok");

    let Content::ToolCall(tc) = &result.content[0] else {
        panic!("expected ToolCall");
    };
    assert_eq!(tc.tool_call_id, "call_w");
    assert_eq!(tc.tool_name, "get_weather");
    assert_eq!(tc.input["city"], "NYC");
    assert_eq!(result.finish_reason.unified, FinishReasonKind::ToolCalls);
}

#[tokio::test]
async fn tool_choice_specific_tool_serializes_function_envelope() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "tool_choice": {
                "type": "function",
                "function": { "name": "get_weather" }
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "r2",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "" },
                "finish_reason": "tool_calls"
            }]
        })))
        .mount(&server)
        .await;

    let _ = provider(&server)
        .chat("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            tools: Some(vec![weather_tool()]),
            tool_choice: Some(ToolChoice::Tool {
                tool_name: "get_weather".into(),
            }),
            ..Default::default()
        })
        .await
        .expect("ok");
}
