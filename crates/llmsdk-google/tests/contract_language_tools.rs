//! Contract tests for Gemini tool calls.
// Rust guideline compliant 2026-05-25

use llmsdk_google::tools::GoogleSearchArgs;
use llmsdk_google::{Google, tools};
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, FinishReasonKind, FunctionTool, Message, TextPart, Tool, UserPart,
};
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Google {
    Google::builder()
        .api_key("test-key")
        .base_url(server.uri())
        .build()
        .expect("builds")
}

fn fn_tool() -> Tool {
    Tool::Function(FunctionTool {
        name: "getWeather".into(),
        description: Some("Look up weather".into()),
        input_schema: serde_json::from_value(json!({
            "type":"object",
            "properties":{"city":{"type":"string"}},
            "required":["city"]
        }))
        .unwrap(),
        input_examples: None,
        strict: None,
        provider_options: None,
    })
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
async fn function_call_routed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:generateContent"))
        .and(body_partial_json(json!({
            "tools": [{
                "functionDeclarations": [{
                    "name": "getWeather"
                }]
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "getWeather",
                            "args": {"city": "Tokyo"}
                        }
                    }]
                },
                "finishReason": "STOP"
            }]
        })))
        .mount(&server)
        .await;

    let model = provider(&server).chat("gemini-2.5-flash");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("weather")],
            tools: Some(vec![fn_tool()]),
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.content.len(), 1);
    let Content::ToolCall(tc) = &result.content[0] else {
        panic!("expected tool call");
    };
    assert_eq!(tc.tool_name, "getWeather");
    assert_eq!(tc.input["city"], "Tokyo");
    assert_eq!(result.finish_reason.unified, FinishReasonKind::ToolCalls);
}

#[tokio::test]
async fn google_search_tool_routed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:generateContent"))
        .and(body_partial_json(json!({
            "tools": [{"googleSearch": {}}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": { "role": "model", "parts": [{"text":"hello"}] },
                "finishReason": "STOP"
            }]
        })))
        .mount(&server)
        .await;

    let model = provider(&server).chat("gemini-2.5-flash");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            tools: Some(vec![tools::google_search(GoogleSearchArgs::default())]),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn code_execution_tool_call_and_result_paired() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        { "executableCode": { "language": "PYTHON", "code": "print(1)" } },
                        { "codeExecutionResult": { "outcome": "OUTCOME_OK", "output": "1" } }
                    ]
                },
                "finishReason": "STOP"
            }]
        })))
        .mount(&server)
        .await;

    let model = provider(&server).chat("gemini-2.5-flash");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("run print(1)")],
            tools: Some(vec![tools::code_execution()]),
            ..Default::default()
        })
        .await
        .expect("ok");

    assert!(result.content.len() >= 2);
    let mut saw_call = false;
    let mut saw_result = false;
    for c in &result.content {
        match c {
            Content::ToolCall(tc) if tc.tool_name == "code_execution" => saw_call = true,
            Content::ToolResult(tr) if tr.tool_name == "code_execution" => saw_result = true,
            _ => {}
        }
    }
    assert!(saw_call && saw_result);
}
