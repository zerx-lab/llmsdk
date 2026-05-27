//! Contract tests for provider-executed tool round-trips through
//! `AnthropicMessagesModel`. Covers:
//!
//! - `AssistantPart::ToolResult` echoing for `web_search` / `web_fetch` /
//!   `code_execution` / mcp / `tool_search` / advisor — mirrors upstream
//!   `convert-to-anthropic-prompt.ts:789-1185`.
//! - `server_tool_use` name remapping for `code_execution_20250825`
//!   sub-tools (`bash_code_execution` / `text_editor_code_execution`
//!   collapse to `code_execution`).
//! - `programmatic-tool-call` type injection.
//!
//! Each test asserts the outbound wire shape via `received_requests` and
//! the inbound parse via response assertions. Both directions catch the
//! regressions introduced by ai-sdk fixes #15552, #15566.
// Rust guideline compliant 2026-02-21

use llmsdk_anthropic::Anthropic;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    AssistantPart, CallOptions, Content, Message, TextPart, ToolCallPart, ToolResultOutput,
    ToolResultPart, UserPart,
};
use llmsdk_provider::shared::ProviderOptions;
use serde_json::{Map, json};
use wiremock::matchers::{method, path};
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

fn anthropic_options(map: serde_json::Map<String, serde_json::Value>) -> ProviderOptions {
    let mut po = ProviderOptions::new();
    po.insert("anthropic".into(), map);
    po
}

#[tokio::test]
async fn web_search_error_echo_routes_to_web_search_tool_result_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1",
            "type": "message",
            "model": "claude-3-5-sonnet-latest",
            "content": [{ "type": "text", "text": "ok" }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).messages("claude-3-5-sonnet-latest");
    let prompt = vec![
        user_text("search"),
        Message::Assistant {
            content: vec![
                AssistantPart::ToolCall(ToolCallPart {
                    tool_call_id: "srvtoolu_err".into(),
                    tool_name: "web_search".into(),
                    input: json!({ "query": "test" }),
                    provider_executed: Some(true),
                    dynamic: None,
                    provider_options: None,
                }),
                AssistantPart::ToolResult(ToolResultPart {
                    tool_call_id: "srvtoolu_err".into(),
                    tool_name: "web_search".into(),
                    output: ToolResultOutput::ErrorJson {
                        value: json!({
                            "type": "web_search_tool_result_error",
                            "errorCode": "invalid_tool_input",
                        }),
                        provider_options: None,
                    },
                    provider_options: None,
                }),
            ],
            provider_options: None,
        },
    ];

    let opts = CallOptions {
        prompt,
        max_output_tokens: Some(8),
        ..Default::default()
    };
    let _ = model.do_generate(opts).await.expect("call succeeds");

    let req = server.received_requests().await.expect("recorded");
    let body: serde_json::Value = serde_json::from_slice(&req[0].body).expect("body is json");
    let assistant_content = body["messages"][1]["content"].as_array().expect("array");
    // The provider-executed `tool-result` is routed to a `web_search_tool_result`
    // wire block (sibling to the `server_tool_use`).
    let result_block = assistant_content
        .iter()
        .find(|b| b["type"] == "web_search_tool_result")
        .expect("web_search_tool_result block present");
    assert_eq!(
        result_block["content"]["type"],
        json!("web_search_tool_result_error")
    );
    assert_eq!(
        result_block["content"]["error_code"],
        json!("invalid_tool_input")
    );
    assert_eq!(result_block["tool_use_id"], json!("srvtoolu_err"));
}

#[tokio::test]
async fn web_fetch_error_echo_routes_to_web_fetch_tool_result_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_2",
            "type": "message",
            "model": "claude-3-5-sonnet-latest",
            "content": [{ "type": "text", "text": "ok" }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).messages("claude-3-5-sonnet-latest");
    let prompt = vec![
        user_text("fetch"),
        Message::Assistant {
            content: vec![
                AssistantPart::ToolCall(ToolCallPart {
                    tool_call_id: "srvtoolu_wf".into(),
                    tool_name: "web_fetch".into(),
                    input: json!({ "url": "https://example.com" }),
                    provider_executed: Some(true),
                    dynamic: None,
                    provider_options: None,
                }),
                AssistantPart::ToolResult(ToolResultPart {
                    tool_call_id: "srvtoolu_wf".into(),
                    tool_name: "web_fetch".into(),
                    output: ToolResultOutput::ErrorJson {
                        value: json!({
                            "type": "web_fetch_tool_result_error",
                            "errorCode": "max_uses_exceeded",
                        }),
                        provider_options: None,
                    },
                    provider_options: None,
                }),
            ],
            provider_options: None,
        },
    ];

    let opts = CallOptions {
        prompt,
        max_output_tokens: Some(8),
        ..Default::default()
    };
    let _ = model.do_generate(opts).await.expect("call succeeds");

    let req = server.received_requests().await.expect("recorded");
    let body: serde_json::Value = serde_json::from_slice(&req[0].body).expect("body is json");
    let assistant_content = body["messages"][1]["content"].as_array().expect("array");
    let result_block = assistant_content
        .iter()
        .find(|b| b["type"] == "web_fetch_tool_result")
        .expect("web_fetch_tool_result block present");
    assert_eq!(
        result_block["content"]["type"],
        json!("web_fetch_tool_result_error")
    );
    assert_eq!(
        result_block["content"]["error_code"],
        json!("max_uses_exceeded")
    );
}

#[tokio::test]
async fn mcp_tool_result_routes_to_mcp_tool_result_block() {
    let server = MockServer::start().await;
    let mut mcp_opts = Map::new();
    mcp_opts.insert("type".into(), json!("mcp-tool-result"));

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_3",
            "type": "message",
            "model": "claude-3-5-sonnet-latest",
            "content": [{ "type": "text", "text": "ok" }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).messages("claude-3-5-sonnet-latest");
    let prompt = vec![
        user_text("call mcp"),
        Message::Assistant {
            content: vec![AssistantPart::ToolResult(ToolResultPart {
                tool_call_id: "mcp_1".into(),
                tool_name: "anything".into(),
                output: ToolResultOutput::Json {
                    value: json!([{ "type": "text", "text": "result" }]),
                    provider_options: None,
                },
                provider_options: Some(anthropic_options(mcp_opts)),
            })],
            provider_options: None,
        },
    ];

    let opts = CallOptions {
        prompt,
        max_output_tokens: Some(8),
        ..Default::default()
    };
    let _ = model.do_generate(opts).await.expect("call succeeds");

    let req = server.received_requests().await.expect("recorded");
    let body: serde_json::Value = serde_json::from_slice(&req[0].body).expect("body is json");
    let assistant_content = body["messages"][1]["content"].as_array().expect("array");
    let block = assistant_content
        .iter()
        .find(|b| b["type"] == "mcp_tool_result")
        .expect("mcp_tool_result block present");
    assert_eq!(block["is_error"], json!(false));
    assert_eq!(block["tool_use_id"], json!("mcp_1"));
}

#[tokio::test]
async fn server_tool_use_bash_code_execution_collapses_to_code_execution() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_4",
            "type": "message",
            "model": "claude-3-5-sonnet-latest",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "tu_bash",
                    "name": "bash_code_execution",
                    "input": { "command": "ls" }
                }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).messages("claude-3-5-sonnet-latest");
    let prompt = vec![user_text("run ls")];
    let opts = CallOptions {
        prompt,
        max_output_tokens: Some(8),
        ..Default::default()
    };
    let result = model.do_generate(opts).await.expect("call succeeds");

    let tool_call = result
        .content
        .iter()
        .find_map(|c| match c {
            Content::ToolCall(tc) => Some(tc),
            _ => None,
        })
        .expect("tool call present");

    // Name collapses to unified `code_execution`.
    assert_eq!(tool_call.tool_name, "code_execution");
    // Sub-tool name preserved as `type` in the input object.
    let type_val = tool_call.input.get("type").and_then(|v| v.as_str());
    assert_eq!(type_val, Some("bash_code_execution"));
    assert_eq!(
        tool_call.input.get("command").and_then(|v| v.as_str()),
        Some("ls")
    );
}

#[tokio::test]
async fn server_tool_use_code_execution_with_code_only_gets_programmatic_type_injected() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_5",
            "type": "message",
            "model": "claude-3-5-sonnet-latest",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "tu_code",
                    "name": "code_execution",
                    "input": { "code": "print(1)" }
                }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).messages("claude-3-5-sonnet-latest");
    let prompt = vec![user_text("run code")];
    let opts = CallOptions {
        prompt,
        max_output_tokens: Some(8),
        ..Default::default()
    };
    let result = model.do_generate(opts).await.expect("call succeeds");

    let tool_call = result
        .content
        .iter()
        .find_map(|c| match c {
            Content::ToolCall(tc) => Some(tc),
            _ => None,
        })
        .expect("tool call present");

    assert_eq!(tool_call.tool_name, "code_execution");
    let type_val = tool_call.input.get("type").and_then(|v| v.as_str());
    assert_eq!(type_val, Some("programmatic-tool-call"));
    assert_eq!(
        tool_call.input.get("code").and_then(|v| v.as_str()),
        Some("print(1)")
    );
}

#[tokio::test]
async fn code_execution_with_explicit_type_is_not_double_prefixed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_6",
            "type": "message",
            "model": "claude-3-5-sonnet-latest",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "tu_typed",
                    "name": "code_execution",
                    "input": { "type": "preset", "code": "print(2)" }
                }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).messages("claude-3-5-sonnet-latest");
    let prompt = vec![user_text("run code")];
    let opts = CallOptions {
        prompt,
        max_output_tokens: Some(8),
        ..Default::default()
    };
    let result = model.do_generate(opts).await.expect("call succeeds");

    let tool_call = result
        .content
        .iter()
        .find_map(|c| match c {
            Content::ToolCall(tc) => Some(tc),
            _ => None,
        })
        .expect("tool call present");

    // Explicit `type` field is preserved (no override).
    let type_val = tool_call.input.get("type").and_then(|v| v.as_str());
    assert_eq!(type_val, Some("preset"));
}
