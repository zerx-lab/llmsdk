//! Contract tests for [`XaiResponsesLanguageModel`] typed-tool routing.
//!
//! Verifies that each of the seven typed tool factories serializes to the
//! wire shape xAI expects, and that the responses parser routes the
//! corresponding server-side tool-call back to the same tool name.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, Message, TextPart, ToolResultOutput, UserPart,
};
use llmsdk_xai::Xai;
use llmsdk_xai::tools::{
    FileSearchOptions, McpServerOptions, WebSearchOptions, XSearchOptions, code_execution,
    file_search, mcp_server, view_image, view_x_video, web_search, x_search,
};
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

fn user_text(text: &str) -> Message {
    Message::User {
        content: vec![UserPart::Text(TextPart {
            text: text.into(),
            provider_options: None,
        })],
        provider_options: None,
    }
}

fn empty_completed() -> serde_json::Value {
    json!({
        "id": "resp_x",
        "object": "response",
        "status": "completed",
        "output": [],
        "usage": {"input_tokens": 0, "output_tokens": 0}
    })
}

#[tokio::test]
async fn web_search_tool_serializes_with_snake_case_wire_fields() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "tools": [{
                "type": "web_search",
                "allowed_domains": ["example.com"],
                "enable_image_understanding": true
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_completed()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let _ = provider
        .responses("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            tools: Some(vec![web_search(&WebSearchOptions {
                allowed_domains: Some(vec!["example.com".into()]),
                enable_image_understanding: Some(true),
                ..Default::default()
            })]),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn x_search_tool_serializes_with_snake_case_wire_fields() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "tools": [{
                "type": "x_search",
                "from_date": "2020-01-01",
                "enable_video_understanding": true
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_completed()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let _ = provider
        .responses("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            tools: Some(vec![x_search(&XSearchOptions {
                from_date: Some("2020-01-01".into()),
                enable_video_understanding: Some(true),
                ..Default::default()
            })]),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn code_execution_emits_code_interpreter_wire_type() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "tools": [{"type": "code_interpreter"}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_completed()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let _ = provider
        .responses("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            tools: Some(vec![code_execution()]),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn view_image_and_view_x_video_emit_simple_types() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "tools": [{"type": "view_image"}, {"type": "view_x_video"}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_completed()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let _ = provider
        .responses("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            tools: Some(vec![view_image(), view_x_video()]),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn file_search_serializes_vector_store_ids() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "tools": [{
                "type": "file_search",
                "vector_store_ids": ["vs_1"],
                "max_num_results": 5
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_completed()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let _ = provider
        .responses("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            tools: Some(vec![file_search(&FileSearchOptions {
                vector_store_ids: vec!["vs_1".into()],
                max_num_results: Some(5),
            })]),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn mcp_server_emits_server_url() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "tools": [{
                "type": "mcp",
                "server_url": "https://mcp.example.com",
                "server_label": "lbl"
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_completed()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let _ = provider
        .responses("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            tools: Some(vec![mcp_server(&McpServerOptions {
                server_url: "https://mcp.example.com".into(),
                server_label: Some("lbl".into()),
                server_description: None,
                allowed_tools: None,
                headers: None,
                authorization: None,
            })]),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn file_search_call_response_emits_tool_call_and_tool_result() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_fs",
            "object": "response",
            "status": "completed",
            "output": [{
                "type": "file_search_call",
                "id": "fs_call_1",
                "status": "completed",
                "queries": ["foo"],
                "results": [{
                    "file_id": "file_1",
                    "filename": "a.txt",
                    "score": 0.9,
                    "text": "snippet"
                }]
            }],
            "usage": {"input_tokens": 0, "output_tokens": 0}
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let result = provider
        .responses("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user_text("search files")],
            tools: Some(vec![file_search(&FileSearchOptions {
                vector_store_ids: vec!["vs_1".into()],
                max_num_results: None,
            })]),
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.content.len(), 2);
    let Content::ToolCall(tc) = &result.content[0] else {
        panic!("expected ToolCall");
    };
    assert_eq!(tc.tool_name, "file_search");
    assert_eq!(tc.provider_executed, Some(true));
    let Content::ToolResult(tr) = &result.content[1] else {
        panic!("expected ToolResult");
    };
    let ToolResultOutput::Json { value, .. } = &tr.output else {
        panic!("expected Json output");
    };
    assert_eq!(value["queries"][0], "foo");
    assert_eq!(value["results"][0]["fileId"], "file_1");
}
