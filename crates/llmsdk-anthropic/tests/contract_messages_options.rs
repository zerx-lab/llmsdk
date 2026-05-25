//! Contract tests covering the M11 review-fix-pack scope:
//!
//! - 20 versioned Anthropic server tool ids + wire `name` overrides + beta headers
//! - 11 new `anthropic` provider options (sendReasoning, structuredOutputMode,
//!   disableParallelToolUse, cacheControl, metadata.userId, mcpServers,
//!   toolStreaming, effort, taskBudget, speed, inferenceGeo, anthropicBeta)
//! - `output_config.format` driven by `responseFormat` + sanitize-json-schema
// Rust guideline compliant 2026-02-21

use llmsdk_anthropic::Anthropic;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    AssistantPart, CallOptions, FunctionTool, Message, ProviderTool, ResponseFormat, TextPart,
    Tool, UserPart,
};
use llmsdk_provider::shared::ProviderOptions;
use serde_json::{Value, json};
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

fn po_with_anthropic(value: &Value) -> ProviderOptions {
    let mut po = ProviderOptions::new();
    po.insert("anthropic".into(), value.as_object().cloned().unwrap());
    po
}

fn ok_response() -> Value {
    json!({
        "id": "msg_1",
        "type": "message",
        "model": "claude-opus-4-7",
        "content": [{ "type": "text", "text": "ok" }],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    })
}

// ---- versioned tool ids ----------------------------------------------------

#[tokio::test]
async fn computer_20251124_routes_with_beta() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("anthropic-beta", "computer-use-2025-11-24"))
        .and(body_partial_json(json!({
            "tools": [{
                "type": "computer_20251124",
                "name": "computer",
                "display_width_px": 1024,
                "display_height_px": 768
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    provider(&server)
        .messages("claude-opus-4-7")
        .do_generate(CallOptions {
            prompt: vec![user_text("use computer")],
            tools: Some(vec![Tool::Provider(ProviderTool {
                id: "anthropic.computer_20251124".into(),
                name: "computer".into(),
                args: Some(serde_json::Map::from_iter([
                    ("display_width_px".into(), json!(1024)),
                    ("display_height_px".into(), json!(768)),
                ])),
                provider_options: None,
            })]),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn memory_20250818_routes_with_context_management_beta() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("anthropic-beta", "context-management-2025-06-27"))
        .and(body_partial_json(json!({
            "tools": [{"type": "memory_20250818", "name": "memory"}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    provider(&server)
        .messages("claude-opus-4-7")
        .do_generate(CallOptions {
            prompt: vec![user_text("remember")],
            tools: Some(vec![Tool::Provider(ProviderTool {
                id: "anthropic.memory_20250818".into(),
                name: "memory".into(),
                args: None,
                provider_options: None,
            })]),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn advisor_20260301_routes_with_advisor_beta() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("anthropic-beta", "advisor-tool-2026-03-01"))
        .and(body_partial_json(json!({
            "tools": [{
                "type": "advisor_20260301",
                "name": "advisor",
                "model": "claude-opus-4-7"
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    provider(&server)
        .messages("claude-haiku-4-5")
        .do_generate(CallOptions {
            prompt: vec![user_text("advise")],
            tools: Some(vec![Tool::Provider(ProviderTool {
                id: "anthropic.advisor_20260301".into(),
                name: "advisor".into(),
                args: Some(serde_json::Map::from_iter([(
                    "model".into(),
                    json!("claude-opus-4-7"),
                )])),
                provider_options: None,
            })]),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn text_editor_20250728_uses_str_replace_based_edit_tool_name() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "tools": [{
                "type": "text_editor_20250728",
                "name": "str_replace_based_edit_tool"
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    provider(&server)
        .messages("claude-opus-4-7")
        .do_generate(CallOptions {
            prompt: vec![user_text("edit")],
            tools: Some(vec![Tool::Provider(ProviderTool {
                id: "anthropic.text_editor_20250728".into(),
                // Caller-supplied name is overridden by the wire-mandated value.
                name: "editor".into(),
                args: None,
                provider_options: None,
            })]),
            ..Default::default()
        })
        .await
        .expect("ok");
}

// ---- new provider options -------------------------------------------------

#[tokio::test]
async fn disable_parallel_tool_use_flows_to_tool_choice() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "tool_choice": { "type": "auto", "disable_parallel_tool_use": true }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let po = po_with_anthropic(&json!({"disableParallelToolUse": true}));
    provider(&server)
        .messages("claude-opus-4-7")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            tools: Some(vec![Tool::Function(FunctionTool {
                name: "add".into(),
                description: None,
                input_schema: serde_json::Map::new().into(),
                input_examples: None,
                strict: None,
                provider_options: None,
            })]),
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn send_reasoning_false_drops_reasoning_blocks() {
    let server = MockServer::start().await;
    // Wire body must NOT contain a thinking block.
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "messages": [
                { "role": "user", "content": [{ "type": "text", "text": "hi" }] },
                { "role": "assistant", "content": [{ "type": "text", "text": "hello" }] }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let po = po_with_anthropic(&json!({"sendReasoning": false}));
    provider(&server)
        .messages("claude-opus-4-7")
        .do_generate(CallOptions {
            prompt: vec![
                user_text("hi"),
                Message::Assistant {
                    content: vec![
                        AssistantPart::Reasoning {
                            text: "ponder".into(),
                            provider_options: None,
                        },
                        AssistantPart::Text(TextPart {
                            text: "hello".into(),
                            provider_options: None,
                        }),
                    ],
                    provider_options: None,
                },
            ],
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn output_config_carries_effort_and_task_budget() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "output_config": {
                "effort": "high",
                "task_budget": { "type": "tokens", "total": 50000 }
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let po = po_with_anthropic(&json!({
        "effort": "high",
        "taskBudget": {"type": "tokens", "total": 50000}
    }));
    provider(&server)
        .messages("claude-opus-4-7")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn output_config_format_from_response_format() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}},
                        "additionalProperties": false
                    }
                }
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let schema_map = json!({"type": "object", "properties": {"x": {"type": "number"}}, "additionalProperties": true})
        .as_object()
        .unwrap()
        .clone();
    provider(&server)
        .messages("claude-opus-4-7")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            response_format: Some(ResponseFormat::Json {
                schema: Some(schema_map.into()),
                name: None,
                description: None,
            }),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn speed_inference_geo_cache_control_metadata_relay() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "speed": "fast",
            "inference_geo": "us",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "metadata": {"user_id": "abc123"}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let po = po_with_anthropic(&json!({
        "speed": "fast",
        "inferenceGeo": "us",
        "cacheControl": {"type": "ephemeral", "ttl": "1h"},
        "metadata": {"userId": "abc123"}
    }));
    provider(&server)
        .messages("claude-opus-4-6")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn mcp_servers_relay_with_field_renames() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "mcp_servers": [{
                "type": "url",
                "name": "internal",
                "url": "https://example.com/mcp",
                "authorization_token": "token-xyz",
                "tool_configuration": {
                    "enabled": true,
                    "allowed_tools": ["a", "b"]
                }
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let po = po_with_anthropic(&json!({
        "mcpServers": [{
            "type": "url",
            "name": "internal",
            "url": "https://example.com/mcp",
            "authorizationToken": "token-xyz",
            "toolConfiguration": {
                "enabled": true,
                "allowedTools": ["a", "b"]
            }
        }]
    }));
    provider(&server)
        .messages("claude-opus-4-7")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn anthropic_beta_adds_extra_header_tokens() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("anthropic-beta", "experimental-feature-1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let po = po_with_anthropic(&json!({"anthropicBeta": ["experimental-feature-1"]}));
    provider(&server)
        .messages("claude-opus-4-7")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn tool_streaming_default_emits_eager_input_streaming_on_function_tools() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "tools": [{
                "name": "add",
                "eager_input_streaming": true
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    // Default tool_streaming = true (omitted in po) means eager_input_streaming=true.
    provider(&server)
        .messages("claude-opus-4-7")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            tools: Some(vec![Tool::Function(FunctionTool {
                name: "add".into(),
                description: None,
                input_schema: serde_json::Map::new().into(),
                input_examples: None,
                strict: None,
                provider_options: None,
            })]),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn context_management_camel_case_fields_rename_to_snake_case_on_wire() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "context_management": {
                "edits": [
                    {
                        "type": "clear_tool_uses_20250919",
                        "clear_at_least": { "type": "input_tokens", "value": 10000 },
                        "clear_tool_inputs": true,
                        "exclude_tools": ["important_tool"]
                    },
                    {
                        "type": "compact_20260112",
                        "pause_after_compaction": true,
                        "instructions": "summarize"
                    }
                ]
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    provider(&server)
        .messages("claude-opus-4-7")
        .do_generate(CallOptions {
            prompt: vec![user_text("hello")],
            provider_options: Some(po_with_anthropic(&json!({
                "contextManagement": {
                    "edits": [
                        {
                            "type": "clear_tool_uses_20250919",
                            "clearAtLeast": { "type": "input_tokens", "value": 10000 },
                            "clearToolInputs": true,
                            "excludeTools": ["important_tool"]
                        },
                        {
                            "type": "compact_20260112",
                            "pauseAfterCompaction": true,
                            "instructions": "summarize"
                        }
                    ]
                }
            }))),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn tool_result_with_tool_reference_emits_nested_array_content() {
    use llmsdk_provider::language_model::{
        ToolMessagePart, ToolOutputPart, ToolResultOutput, ToolResultPart,
    };

    let server = MockServer::start().await;
    let mut po_tool_ref = ProviderOptions::new();
    po_tool_ref.insert(
        "anthropic".into(),
        json!({"type": "tool-reference", "toolName": "get_weather"})
            .as_object()
            .unwrap()
            .clone(),
    );

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "hi" },
                        {
                            "type": "tool_result",
                            "tool_use_id": "srvtoolu_1",
                            "content": [{"type": "tool_reference", "tool_name": "get_weather"}]
                        }
                    ]
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    provider(&server)
        .messages("claude-opus-4-7")
        .do_generate(CallOptions {
            prompt: vec![
                user_text("hi"),
                Message::Tool {
                    content: vec![ToolMessagePart::ToolResult(ToolResultPart {
                        tool_call_id: "srvtoolu_1".into(),
                        tool_name: "tool_search_tool_regex".into(),
                        output: ToolResultOutput::Content {
                            value: vec![ToolOutputPart::Custom {
                                provider_options: Some(po_tool_ref),
                            }],
                        },
                        provider_options: None,
                    })],
                    provider_options: None,
                },
            ],
            ..Default::default()
        })
        .await
        .expect("ok");
}
