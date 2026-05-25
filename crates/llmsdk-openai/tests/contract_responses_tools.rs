//! Contract tests for the Responses API tool routing.
//!
//! Covers both function tools and the provider-defined tool set
//! (`web_search`, `file_search`, `code_interpreter`, `image_generation`).
// Rust guideline compliant 2026-02-21

use llmsdk_openai::OpenAi;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::json::JsonSchema;
use llmsdk_provider::language_model::{
    CallOptions, Content, FunctionTool, Message, ProviderTool, TextPart, Tool, UserPart,
};
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
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

fn empty_schema() -> JsonSchema {
    serde_json::from_value(json!({"type":"object"})).unwrap()
}

fn happy_msg() -> serde_json::Value {
    json!({
        "id": "resp_t",
        "model": "gpt-4o-mini",
        "output": [{
            "type": "message",
            "id": "m",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}]
        }],
        "usage": {"input_tokens": 1, "output_tokens": 1}
    })
}

#[tokio::test]
async fn function_tool_is_serialized_with_parameters() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "strict": true,
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(happy_msg()))
        .mount(&server)
        .await;

    let model = provider(&server).responses("gpt-4o-mini");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user("weather?")],
            tools: Some(vec![Tool::Function(FunctionTool {
                name: "get_weather".into(),
                description: Some("Get weather".into()),
                input_schema: empty_schema(),
                input_examples: None,
                strict: Some(true),
                provider_options: None,
            })]),
            ..Default::default()
        })
        .await
        .expect("call");
}

#[tokio::test]
async fn web_search_tool_routes_and_auto_include_sources() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "tools": [{ "type": "web_search", "search_context_size": "high" }],
            "include": ["web_search_call.action.sources"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "r",
            "model": "gpt-4o-mini",
            "output": [{
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": { "type": "search", "query": "rust" }
            }],
            "usage": {"input_tokens": 0, "output_tokens": 0}
        })))
        .mount(&server)
        .await;

    let model = provider(&server).responses("gpt-4o-mini");
    let r = model
        .do_generate(CallOptions {
            prompt: vec![user("search rust")],
            tools: Some(vec![Tool::Provider(ProviderTool {
                id: "openai.web_search".into(),
                name: "web_search".into(),
                args: json!({"searchContextSize": "high"}).as_object().cloned(),
                provider_options: None,
            })]),
            ..Default::default()
        })
        .await
        .expect("call");

    assert!(
        r.content
            .iter()
            .any(|c| matches!(c, Content::ToolCall(tc) if tc.tool_name == "web_search"))
    );
    assert!(
        r.content
            .iter()
            .any(|c| matches!(c, Content::ToolResult(_)))
    );
}

#[tokio::test]
async fn file_search_tool_routes_vector_store_ids() {
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
        .respond_with(ResponseTemplate::new(200).set_body_json(happy_msg()))
        .mount(&server)
        .await;

    let model = provider(&server).responses("gpt-4o-mini");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user("find docs")],
            tools: Some(vec![Tool::Provider(ProviderTool {
                id: "openai.file_search".into(),
                name: "file_search".into(),
                args: json!({"vectorStoreIds": ["vs_1"], "maxNumResults": 5})
                    .as_object()
                    .cloned(),
                provider_options: None,
            })]),
            ..Default::default()
        })
        .await
        .expect("call");
}

#[tokio::test]
async fn code_interpreter_auto_include_outputs() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "tools": [{ "type": "code_interpreter" }],
            "include": ["code_interpreter_call.outputs"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(happy_msg()))
        .mount(&server)
        .await;

    let model = provider(&server).responses("gpt-4o-mini");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user("run code")],
            tools: Some(vec![Tool::Provider(ProviderTool {
                id: "openai.code_interpreter".into(),
                name: "code_interpreter".into(),
                args: json!({}).as_object().cloned(),
                provider_options: None,
            })]),
            ..Default::default()
        })
        .await
        .expect("call");
}

#[tokio::test]
async fn image_generation_response_emits_tool_call_and_result() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "r",
            "model": "gpt-4o-mini",
            "output": [{
                "type": "image_generation_call",
                "id": "img_1",
                "result": "BASE64DATA"
            }],
            "usage": {"input_tokens": 0, "output_tokens": 0}
        })))
        .mount(&server)
        .await;

    let model = provider(&server).responses("gpt-4o-mini");
    let r = model
        .do_generate(CallOptions {
            prompt: vec![user("draw a cat")],
            tools: Some(vec![Tool::Provider(ProviderTool {
                id: "openai.image_generation".into(),
                name: "image_generation".into(),
                args: json!({"quality": "high"}).as_object().cloned(),
                provider_options: None,
            })]),
            ..Default::default()
        })
        .await
        .expect("call");

    let has_call = r
        .content
        .iter()
        .any(|c| matches!(c, Content::ToolCall(tc) if tc.tool_name == "openai.image_generation"));
    let has_result = r
        .content
        .iter()
        .any(|c| matches!(c, Content::ToolResult(tr) if tr.tool_name == "openai.image_generation"));
    assert!(has_call && has_result);
}
