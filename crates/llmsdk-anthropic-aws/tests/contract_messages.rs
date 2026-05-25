//! Contract tests for the Messages end-to-end pipeline + base URL templating.

use llmsdk_anthropic_aws::{AnthropicAws, tools};
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{CallOptions, Message, TextPart, UserPart};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn test_prompt() -> Vec<Message> {
    vec![Message::User {
        content: vec![UserPart::Text(TextPart {
            text: "Hi".into(),
            provider_options: None,
        })],
        provider_options: None,
    }]
}

fn success_body() -> serde_json::Value {
    json!({
        "type": "message",
        "id": "msg_e2e",
        "model": "claude-sonnet-4-6",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi back"}],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 2, "output_tokens": 2}
    })
}

#[tokio::test]
async fn base_url_override_routes_to_proxy() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("anthropic-workspace-id", "wrkspc_x"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_body()))
        .mount(&server)
        .await;

    let provider = AnthropicAws::builder()
        .region("us-west-2")
        .workspace_id("wrkspc_x")
        .api_key("sk-test")
        .base_url(format!("{}/", server.uri())) // trailing slash exercised
        .build()
        .unwrap();

    let result = provider
        .language_model("claude-sonnet-4-6")
        .do_generate(CallOptions {
            prompt: test_prompt(),
            max_output_tokens: Some(64),
            ..Default::default()
        })
        .await
        .unwrap();

    let text = result.content.iter().find_map(|c| match c {
        llmsdk_provider::language_model::Content::Text(t) => Some(t.text.as_str()),
        _ => None,
    });
    assert_eq!(text, Some("Hi back"));
}

#[tokio::test]
async fn custom_extra_header_is_forwarded() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-custom-trace", "trace-123"))
        .and(header("anthropic-workspace-id", "wrkspc_h"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_body()))
        .mount(&server)
        .await;

    let provider = AnthropicAws::builder()
        .region("us-west-2")
        .workspace_id("wrkspc_h")
        .api_key("sk-test")
        .base_url(server.uri())
        .header("x-custom-trace", Some("trace-123".to_owned()))
        .build()
        .unwrap();

    provider
        .language_model("claude-sonnet-4-6")
        .do_generate(CallOptions {
            prompt: test_prompt(),
            max_output_tokens: Some(64),
            ..Default::default()
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn anthropic_tools_module_is_re_exported() {
    // Sanity: the tools re-export should expose the same typed factories
    // as `llmsdk-anthropic::tools` — calling one builder confirms wiring.
    let tool = tools::web_search_20260209(tools::WebSearchArgs::default());
    match tool {
        llmsdk_provider::language_model::Tool::Provider(p) => {
            assert!(p.id.starts_with("anthropic."), "got: {}", p.id);
        }
        llmsdk_provider::language_model::Tool::Function(_) => {
            panic!("expected provider tool, got function tool");
        }
    }
}
