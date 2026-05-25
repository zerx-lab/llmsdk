//! Contract tests for the more involved Responses flows:
//! - MCP approval round-trip (`mcp_approval_request` → `ToolApprovalRequest`)
//! - Streaming `apply_patch` operation diff delta accumulation.
// Rust guideline compliant 2026-02-21

use futures::StreamExt;
use llmsdk_openai::OpenAi;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, Message, ProviderTool, StreamPart, TextPart, Tool, UserPart,
};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> OpenAi {
    OpenAi::builder()
        .api_key("sk-test")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
}

fn user(s: &str) -> Message {
    Message::User {
        content: vec![UserPart::Text(TextPart {
            text: s.into(),
            provider_options: None,
        })],
        provider_options: None,
    }
}

fn sse(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(body.to_owned())
}

#[tokio::test]
async fn mcp_call_in_response_emits_dynamic_tool_call_and_result() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "r",
            "model": "gpt-4o-mini",
            "output": [{
                "type": "mcp_call",
                "id": "mcp_1",
                "status": "completed",
                "arguments": "{\"q\": \"x\"}",
                "name": "search",
                "server_label": "fs",
                "output": "result"
            }],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .mount(&server)
        .await;

    let model = provider(&server).responses("gpt-4o-mini");
    let r = model
        .do_generate(CallOptions {
            prompt: vec![user("use mcp")],
            tools: Some(vec![Tool::Provider(ProviderTool {
                id: "openai.mcp".into(),
                name: "mcp".into(),
                args: json!({"serverLabel": "fs", "serverUrl": "https://mcp.example"})
                    .as_object()
                    .cloned(),
                provider_options: None,
            })]),
            ..Default::default()
        })
        .await
        .expect("call");

    let dynamic_call = r
        .content
        .iter()
        .find_map(|c| match c {
            Content::ToolCall(tc) if tc.dynamic == Some(true) => Some(tc),
            _ => None,
        })
        .expect("expected dynamic mcp tool call");
    assert!(dynamic_call.tool_name.starts_with("mcp."));
    assert!(
        r.content
            .iter()
            .any(|c| matches!(c, Content::ToolResult(tr) if tr.tool_name.starts_with("mcp.")))
    );
}

#[tokio::test]
async fn mcp_approval_request_in_stream_emits_tool_approval_request() {
    let body = concat!(
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"mcp_approval_request\",\"id\":\"appr_1\",\"server_label\":\"fs\",\"name\":\"delete\",\"arguments\":\"{}\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\n",
        "data: [DONE]\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(sse(body))
        .mount(&server)
        .await;

    let model = provider(&server).responses("gpt-4o-mini");
    let res = model
        .do_stream(CallOptions {
            prompt: vec![user("delete that")],
            ..Default::default()
        })
        .await
        .expect("stream");
    let mut parts = Vec::new();
    let mut s = res.stream;
    while let Some(p) = s.next().await {
        parts.push(p.expect("no error"));
    }
    assert!(
        parts
            .iter()
            .any(|p| matches!(p, StreamPart::ToolApprovalRequest(_)))
    );
}

#[tokio::test]
async fn apply_patch_stream_accumulates_diff_into_tool_input() {
    let body = concat!(
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"apply_patch_call\",\"id\":\"ap_1\",\"call_id\":\"call_ap\",\"status\":\"in_progress\",\"operation\":{\"type\":\"update_file\",\"path\":\"a.rs\",\"diff\":\"\"}}}\n\n",
        "data: {\"type\":\"response.apply_patch_call_operation_diff.delta\",\"item_id\":\"ap_1\",\"output_index\":0,\"delta\":\"@@\"}\n\n",
        "data: {\"type\":\"response.apply_patch_call_operation_diff.done\",\"item_id\":\"ap_1\",\"output_index\":0,\"diff\":\"@@\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"apply_patch_call\",\"id\":\"ap_1\",\"call_id\":\"call_ap\",\"status\":\"completed\",\"operation\":{\"type\":\"update_file\",\"path\":\"a.rs\",\"diff\":\"@@\"}}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\n",
        "data: [DONE]\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(sse(body))
        .mount(&server)
        .await;

    let model = provider(&server).responses("gpt-4o-mini");
    let res = model
        .do_stream(CallOptions {
            prompt: vec![user("update file")],
            tools: Some(vec![Tool::Provider(ProviderTool {
                id: "openai.apply_patch".into(),
                name: "apply_patch".into(),
                args: json!({}).as_object().cloned(),
                provider_options: None,
            })]),
            ..Default::default()
        })
        .await
        .expect("stream");
    let mut parts = Vec::new();
    let mut s = res.stream;
    while let Some(p) = s.next().await {
        parts.push(p.expect("no err"));
    }
    let deltas: Vec<&str> = parts
        .iter()
        .filter_map(|p| match p {
            StreamPart::ToolInputDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    let joined = deltas.join("");
    assert!(joined.contains("@@"));
    assert!(joined.ends_with("\"}}"));
    assert!(
        parts
            .iter()
            .any(|p| matches!(p, StreamPart::ToolInputEnd { .. }))
    );
    assert!(
        parts
            .iter()
            .any(|p| matches!(p, StreamPart::ToolCall(tc) if tc.tool_name == "openai.apply_patch"))
    );
}
