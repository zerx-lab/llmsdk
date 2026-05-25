//! Contract tests covering the 22 `provider_options.openai.*` for Responses.
// Rust guideline compliant 2026-02-21

use llmsdk_openai::OpenAi;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{CallOptions, Message, TextPart, UserPart};
use llmsdk_provider::shared::ProviderOptions;
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

fn empty_body() -> serde_json::Value {
    json!({
        "id": "r",
        "model": "gpt-5-mini",
        "output": [],
        "usage": {"input_tokens": 0, "output_tokens": 0}
    })
}

fn po(kv: &serde_json::Value) -> ProviderOptions {
    let mut po = ProviderOptions::new();
    po.insert("openai".into(), kv.as_object().unwrap().clone());
    po
}

#[tokio::test]
async fn all_simple_passthroughs_land_on_wire() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "model": "gpt-5-mini",
            "instructions": "be concise",
            "metadata": { "trace": "abc" },
            "parallel_tool_calls": true,
            "previous_response_id": "resp_prev",
            "store": false,
            "user": "u1",
            "prompt_cache_key": "k1",
            "prompt_cache_retention": "24h",
            "safety_identifier": "safe-1",
            "service_tier": "priority",
            "truncation": "auto",
            "max_tool_calls": 7
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_body()))
        .mount(&server)
        .await;

    let model = provider(&server).responses("gpt-5-mini");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            provider_options: Some(po(&json!({
                "instructions": "be concise",
                "metadata": {"trace": "abc"},
                "parallelToolCalls": true,
                "previousResponseId": "resp_prev",
                "store": false,
                "user": "u1",
                "promptCacheKey": "k1",
                "promptCacheRetention": "24h",
                "safetyIdentifier": "safe-1",
                "serviceTier": "priority",
                "truncation": "auto",
                "maxToolCalls": 7,
            }))),
            ..Default::default()
        })
        .await
        .expect("call");
}

#[tokio::test]
async fn logprobs_count_emits_top_logprobs_and_auto_include() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "top_logprobs": 5,
            "include": ["message.output_text.logprobs"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_body()))
        .mount(&server)
        .await;
    let model = provider(&server).responses("gpt-4o-mini");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            provider_options: Some(po(&json!({"logprobs": 5}))),
            ..Default::default()
        })
        .await
        .expect("call");
}

#[tokio::test]
async fn text_verbosity_serializes_under_text_block() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "text": { "verbosity": "high" }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_body()))
        .mount(&server)
        .await;
    let model = provider(&server).responses("gpt-4o-mini");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            provider_options: Some(po(&json!({"textVerbosity": "high"}))),
            ..Default::default()
        })
        .await
        .expect("call");
}

#[tokio::test]
async fn allowed_tools_overrides_tool_choice() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": "a"}, {"type": "function", "name": "b"}]
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_body()))
        .mount(&server)
        .await;
    let model = provider(&server).responses("gpt-4o-mini");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            tools: Some(vec![
                llmsdk_provider::language_model::Tool::Function(
                    llmsdk_provider::language_model::FunctionTool {
                        name: "a".into(),
                        description: None,
                        input_schema: serde_json::from_value(json!({"type":"object"})).unwrap(),
                        input_examples: None,
                        strict: None,
                        provider_options: None,
                    },
                ),
                llmsdk_provider::language_model::Tool::Function(
                    llmsdk_provider::language_model::FunctionTool {
                        name: "b".into(),
                        description: None,
                        input_schema: serde_json::from_value(json!({"type":"object"})).unwrap(),
                        input_examples: None,
                        strict: None,
                        provider_options: None,
                    },
                ),
            ]),
            provider_options: Some(po(&json!({
                "allowedTools": {"toolNames": ["a", "b"], "mode": "required"}
            }))),
            ..Default::default()
        })
        .await
        .expect("call");
}

#[tokio::test]
async fn flex_on_unsupported_model_warns_and_drops() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_body()))
        .mount(&server)
        .await;
    let model = provider(&server).responses("gpt-4o-mini");
    let r = model
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            provider_options: Some(po(&json!({"serviceTier": "flex"}))),
            ..Default::default()
        })
        .await
        .expect("call");
    assert!(r.warnings.iter().any(|w| matches!(
        w,
        llmsdk_provider::shared::Warning::UnsupportedSetting { setting, .. } if setting == "serviceTier"
    )));
}

#[tokio::test]
async fn context_management_compaction_serialized() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "context_management": [{ "type": "compaction", "compact_threshold": 4096 }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_body()))
        .mount(&server)
        .await;
    let model = provider(&server).responses("gpt-4o-mini");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            provider_options: Some(po(&json!({
                "contextManagement": [{"type": "compaction", "compactThreshold": 4096}]
            }))),
            ..Default::default()
        })
        .await
        .expect("call");
}

#[tokio::test]
async fn store_false_on_reasoning_model_auto_includes_encrypted_reasoning() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "include": ["reasoning.encrypted_content"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_body()))
        .mount(&server)
        .await;
    let model = provider(&server).responses("o3-mini");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            provider_options: Some(po(&json!({"store": false}))),
            ..Default::default()
        })
        .await
        .expect("call");
}
