//! Contract tests: typed tool factories produce the right wire id / name / args.
//!
//! End-to-end check that `anthropic.tools::*` factories build a
//! `Tool::Provider` whose `id` is recognized by
//! `messages::model::resolve_anthropic_server_tool`. We send a minimal
//! request through the model and assert the upstream `tools[]` array shape.
// Rust guideline compliant 2026-02-21

use llmsdk_anthropic::Anthropic;
use llmsdk_anthropic::tools::{
    AdvisorArgs, CitationsConfig, ComputerArgs, ComputerArgsWithZoom, TextEditor20250728Args,
    UserLocation, UserLocationKind, WebFetchArgs, WebSearchArgs, advisor_20260301, bash_20250124,
    code_execution_20260120, computer_20250124, computer_20251124, memory_20250818,
    text_editor_20250728, web_fetch_20260209, web_search_20260209,
};
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{CallOptions, Message, TextPart, Tool, UserPart};
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Anthropic {
    Anthropic::builder()
        .api_key("sk-ant-test")
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

fn ok_response() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "msg_1",
        "type": "message",
        "content": [{ "type": "text", "text": "ok" }],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    }))
}

async fn send_with_tool(server: &MockServer, tool: Tool) {
    let model = provider(server).messages("claude-sonnet-4-6");
    model
        .do_generate(CallOptions {
            prompt: vec![user("ping")],
            tools: Some(vec![tool]),
            max_output_tokens: Some(8),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn advisor_factory_emits_expected_wire_shape() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "tools": [{
                "type": "advisor_20260301",
                "name": "advisor",
                "model": "claude-opus-4-7"
            }]
        })))
        .respond_with(ok_response())
        .mount(&server)
        .await;

    send_with_tool(
        &server,
        advisor_20260301(AdvisorArgs {
            model: "claude-opus-4-7".into(),
            max_uses: Some(3),
            caching: None,
        }),
    )
    .await;
}

#[tokio::test]
async fn computer_v3_emits_zoom_capable_wire_shape() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "tools": [{
                "type": "computer_20251124",
                "name": "computer",
                "display_width_px": 1920,
                "display_height_px": 1080,
                "enable_zoom": true
            }]
        })))
        .respond_with(ok_response())
        .mount(&server)
        .await;

    send_with_tool(
        &server,
        computer_20251124(ComputerArgsWithZoom {
            display_width_px: 1920,
            display_height_px: 1080,
            display_number: None,
            enable_zoom: Some(true),
        }),
    )
    .await;
}

#[tokio::test]
async fn web_search_user_location_passes_through() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "tools": [{
                "type": "web_search_20260209",
                "name": "web_search",
                "user_location": { "type": "approximate", "city": "Paris" }
            }]
        })))
        .respond_with(ok_response())
        .mount(&server)
        .await;

    send_with_tool(
        &server,
        web_search_20260209(WebSearchArgs {
            user_location: Some(UserLocation {
                kind: UserLocationKind::Approximate,
                city: Some("Paris".into()),
                ..Default::default()
            }),
            ..Default::default()
        }),
    )
    .await;
}

#[tokio::test]
async fn web_fetch_citations_pass_through() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "tools": [{
                "type": "web_fetch_20260209",
                "name": "web_fetch",
                "citations": { "enabled": true }
            }]
        })))
        .respond_with(ok_response())
        .mount(&server)
        .await;

    send_with_tool(
        &server,
        web_fetch_20260209(WebFetchArgs {
            citations: Some(CitationsConfig { enabled: true }),
            ..Default::default()
        }),
    )
    .await;
}

#[tokio::test]
async fn zero_arg_tools_emit_type_and_name_only() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "tools": [{ "type": "bash_20250124", "name": "bash" }]
        })))
        .respond_with(ok_response())
        .mount(&server)
        .await;

    send_with_tool(&server, bash_20250124()).await;
}

#[tokio::test]
async fn code_execution_latest_no_beta_required() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "tools": [{ "type": "code_execution_20260120", "name": "code_execution" }]
        })))
        .respond_with(ok_response())
        .mount(&server)
        .await;

    send_with_tool(&server, code_execution_20260120()).await;
}

#[tokio::test]
async fn memory_tool_no_args() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "tools": [{ "type": "memory_20250818", "name": "memory" }]
        })))
        .respond_with(ok_response())
        .mount(&server)
        .await;

    send_with_tool(&server, memory_20250818()).await;
}

#[tokio::test]
async fn text_editor_20250728_max_characters_passes_through() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "tools": [{
                "type": "text_editor_20250728",
                "name": "str_replace_based_edit_tool",
                "max_characters": 50000
            }]
        })))
        .respond_with(ok_response())
        .mount(&server)
        .await;

    send_with_tool(
        &server,
        text_editor_20250728(TextEditor20250728Args {
            max_characters: Some(50_000),
        }),
    )
    .await;
}

#[tokio::test]
async fn computer_basic_args_no_zoom() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "tools": [{
                "type": "computer_20250124",
                "name": "computer",
                "display_width_px": 1024,
                "display_height_px": 768
            }]
        })))
        .respond_with(ok_response())
        .mount(&server)
        .await;

    send_with_tool(
        &server,
        computer_20250124(ComputerArgs {
            display_width_px: 1024,
            display_height_px: 768,
            display_number: None,
        }),
    )
    .await;
}
