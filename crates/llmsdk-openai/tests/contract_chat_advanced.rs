//! Contract tests for M7 capability extensions.
//!
//! Covers:
//!
//! - reasoning-model parameter stripping + `max_completion_tokens` mapping
//! - `reasoning_effort` plumbing (top-level + provider option override)
//! - `developer` role for system messages on reasoning models
//! - `gpt-4o-search-preview*` `temperature` stripping
//! - `logprobs` provider option transport and `provider_metadata` collection
//! - `url_citation` annotations → `Content::Source`
//! - streaming counterparts where applicable
//!
//! Each test boots a `wiremock` server, captures the outgoing request body
//! by setting it as the wiremock matcher, and asserts both the request
//! shape and the response mapping.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use futures::StreamExt;
use llmsdk_openai::OpenAi;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, Message, ReasoningEffort, Source, StreamPart, TextPart, UserPart,
};
use llmsdk_provider::shared::{ProviderOptions, Warning};
use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> OpenAi {
    OpenAi::builder()
        .api_key("sk-test")
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

fn system(text: &str) -> Message {
    Message::System {
        content: text.into(),
        provider_options: None,
    }
}

fn provider_options_with_openai(value: &Value) -> ProviderOptions {
    let mut po = ProviderOptions::new();
    po.insert("openai".into(), value.as_object().cloned().unwrap());
    po
}

fn empty_choice_response(model: &str) -> Value {
    json!({
        "id": "chatcmpl-test",
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    })
}

// ---------- reasoning models ------------------------------------------

#[tokio::test]
async fn reasoning_model_strips_temperature_and_top_p_and_warns() {
    let server = MockServer::start().await;

    // The outgoing body must NOT contain `temperature` / `top_p`, but it
    // MUST carry `reasoning_effort` (top-level field) and
    // `max_completion_tokens` (mapped from `max_output_tokens`).
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "model": "o3-mini",
            "max_completion_tokens": 256,
            "reasoning_effort": "medium",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_choice_response("o3-mini")))
        .mount(&server)
        .await;

    let model = provider(&server).chat("o3-mini");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_output_tokens: Some(256),
            reasoning: Some(ReasoningEffort::Medium),
            ..Default::default()
        })
        .await
        .expect("ok");

    // Two `UnsupportedSetting` warnings: temperature + topP.
    let kinds: Vec<&str> = result
        .warnings
        .iter()
        .filter_map(|w| match w {
            Warning::Unsupported { feature, .. } => Some(feature.as_str()),
            _ => None,
        })
        .collect();
    assert!(kinds.contains(&"temperature"));
    assert!(kinds.contains(&"topP"));
}

#[tokio::test]
async fn reasoning_model_uses_developer_role_for_system_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "messages": [
                {"role": "developer", "content": "be brief"},
                {"role": "user", "content": "hi"}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_choice_response("o1-mini")))
        .mount(&server)
        .await;

    let model = provider(&server).chat("o1-mini");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![system("be brief"), user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn reasoning_effort_provider_option_overrides_top_level() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({"reasoning_effort": "high"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_choice_response("o3")))
        .mount(&server)
        .await;

    let model = provider(&server).chat("o3");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            reasoning: Some(ReasoningEffort::Low),
            provider_options: Some(provider_options_with_openai(&json!(
                {"reasoningEffort": "high"}
            ))),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn force_reasoning_treats_unknown_model_as_reasoning() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "messages": [{"role": "developer", "content": "x"}, {"role":"user","content":"hi"}]
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(empty_choice_response("custom-alias")),
        )
        .mount(&server)
        .await;

    let model = provider(&server).chat("custom-alias");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![system("x"), user_text("hi")],
            provider_options: Some(provider_options_with_openai(&json!(
                {"forceReasoning": true}
            ))),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn gpt_5_1_none_effort_keeps_temperature() {
    let server = MockServer::start().await;
    // `reasoning_effort = "none"` on gpt-5.1+ keeps temperature on the wire.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "model": "gpt-5.1",
            "reasoning_effort": "none",
            "temperature": 0.5,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_choice_response("gpt-5.1")))
        .mount(&server)
        .await;

    let model = provider(&server).chat("gpt-5.1");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            temperature: Some(0.5),
            reasoning: Some(ReasoningEffort::None),
            ..Default::default()
        })
        .await
        .expect("ok");

    assert!(
        !result.warnings.iter().any(|w| matches!(
            w,
            Warning::Unsupported { feature, .. } if feature == "temperature"
        )),
        "temperature should not be stripped for gpt-5.1 with effort=none"
    );
}

// ---------- search-preview models -------------------------------------

#[tokio::test]
async fn search_preview_strips_temperature() {
    let server = MockServer::start().await;
    // search-preview model: temperature must be absent in the outgoing body.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(empty_choice_response("gpt-4o-search-preview")),
        )
        .mount(&server)
        .await;

    let model = provider(&server).chat("gpt-4o-search-preview");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            temperature: Some(0.5),
            ..Default::default()
        })
        .await
        .expect("ok");

    assert!(result.warnings.iter().any(|w| matches!(
        w,
        Warning::Unsupported { feature, .. } if feature == "temperature"
    )));
}

// ---------- logprobs ---------------------------------------------------

#[tokio::test]
async fn logprobs_flag_relays_and_metadata_collected() {
    let server = MockServer::start().await;
    let logprobs_payload = json!([{
        "token": "hi",
        "logprob": -0.1,
        "top_logprobs": []
    }]);
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "logprobs": true,
            "top_logprobs": 0,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "r-1",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "logprobs": {"content": logprobs_payload},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .mount(&server)
        .await;

    let model = provider(&server).chat("gpt-4o-mini");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(provider_options_with_openai(&json!({"logprobs": true}))),
            ..Default::default()
        })
        .await
        .expect("ok");

    let pm = result.provider_metadata.expect("provider_metadata present");
    let openai = pm.get("openai").expect("openai entry");
    let logprobs = openai.get("logprobs").expect("logprobs entry");
    assert!(logprobs.is_array(), "logprobs must be an array");
    assert_eq!(logprobs[0]["token"], "hi");
}

#[tokio::test]
async fn logprobs_count_relays_top_logprobs_n() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(
            json!({"logprobs": true, "top_logprobs": 5}),
        ))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(empty_choice_response("gpt-4o-mini")),
        )
        .mount(&server)
        .await;

    let model = provider(&server).chat("gpt-4o-mini");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(provider_options_with_openai(&json!({"logprobs": 5}))),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn logprobs_false_omits_field_on_wire() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(empty_choice_response("gpt-4o-mini")),
        )
        .mount(&server)
        .await;

    let model = provider(&server).chat("gpt-4o-mini");
    model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(provider_options_with_openai(&json!({"logprobs": false}))),
            ..Default::default()
        })
        .await
        .expect("ok");

    // Mirrors ai-sdk openai-chat-language-model.ts:149-153: logprobs is only
    // sent when the option is true / a number; false ⇒ omitted entirely.
    let request = &server.received_requests().await.unwrap()[0];
    let body: serde_json::Value =
        serde_json::from_slice(&request.body).expect("wire body should parse");
    assert!(
        body.get("logprobs").is_none(),
        "logprobs:false must omit the wire field, got {body:?}"
    );
    assert!(
        body.get("top_logprobs").is_none(),
        "logprobs:false must omit top_logprobs too, got {body:?}"
    );
}

#[tokio::test]
async fn logprobs_dropped_for_reasoning_model_with_warning() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_choice_response("o3-mini")))
        .mount(&server)
        .await;

    let model = provider(&server).chat("o3-mini");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(provider_options_with_openai(&json!({"logprobs": 3}))),
            ..Default::default()
        })
        .await
        .expect("ok");

    let messages: Vec<&str> = result
        .warnings
        .iter()
        .filter_map(|w| match w {
            Warning::Other { message } => Some(message.as_str()),
            _ => None,
        })
        .collect();
    assert!(messages.iter().any(|m| m.contains("logprobs")));
    assert!(messages.iter().any(|m| m.contains("topLogprobs")));
}

// ---------- annotations ------------------------------------------------

#[tokio::test]
async fn url_citation_annotation_emits_source_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-cite",
            "model": "gpt-4o-search-preview",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Capital of Denmark is Copenhagen.",
                    "annotations": [{
                        "type": "url_citation",
                        "url_citation": {
                            "start_index": 0,
                            "end_index": 5,
                            "url": "https://example.com/dk",
                            "title": "Denmark Facts"
                        }
                    }]
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .mount(&server)
        .await;

    let model = provider(&server).chat("gpt-4o-search-preview");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("What is the capital of Denmark?")],
            ..Default::default()
        })
        .await
        .expect("ok");

    let sources: Vec<&Source> = result
        .content
        .iter()
        .filter_map(|c| match c {
            Content::Source(s) => Some(s),
            _ => None,
        })
        .collect();
    assert_eq!(sources.len(), 1);
    if let Source::Url { url, title, id, .. } = sources[0] {
        assert_eq!(url, "https://example.com/dk");
        assert_eq!(title.as_deref(), Some("Denmark Facts"));
        assert!(id.starts_with("chatcmpl-cite:citation:"));
    } else {
        panic!("expected Source::Url");
    }
}

// ---------- streaming annotations + logprobs --------------------------

#[tokio::test]
async fn stream_relays_annotations_and_logprobs_metadata() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-stream\",\"model\":\"gpt-4o-search-preview\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi \"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"there\",\"annotations\":[{\"type\":\"url_citation\",\"url_citation\":{\"start_index\":0,\"end_index\":2,\"url\":\"https://example.com/a\",\"title\":\"A\"}}]},\"logprobs\":{\"content\":[{\"token\":\"hi\",\"logprob\":-0.5,\"top_logprobs\":[]}]}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n",
        "data: [DONE]\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&server)
        .await;

    let model = provider(&server).chat("gpt-4o-search-preview");
    let result = model
        .do_stream(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("stream ok");

    let mut stream = result.stream;
    let mut got_source = false;
    let mut finish_metadata: Option<HashMap<String, _>> = None;
    while let Some(item) = stream.next().await {
        let part = item.expect("stream part");
        match part {
            StreamPart::Source(Source::Url { url, id, .. }) => {
                got_source = true;
                assert_eq!(url, "https://example.com/a");
                assert!(id.starts_with("chatcmpl-stream:citation:"));
            }
            StreamPart::Finish {
                provider_metadata, ..
            } => finish_metadata = provider_metadata,
            _ => {}
        }
    }

    assert!(got_source, "did not see Source frame");
    let pm = finish_metadata.expect("provider_metadata on Finish");
    let openai = pm.get("openai").expect("openai entry");
    let logprobs = openai.get("logprobs").expect("logprobs entry");
    assert!(logprobs.is_array());
}

// ---------- new option transport (review fix-pack) --------------------

#[tokio::test]
async fn prompt_cache_retention_relays() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "model": "gpt-5.1",
            "prompt_cache_retention": "24h",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_choice_response("gpt-5.1")))
        .mount(&server)
        .await;

    let po = provider_options_with_openai(&json!({"promptCacheRetention": "24h"}));
    provider(&server)
        .chat("gpt-5.1")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("relays prompt_cache_retention");
}

#[tokio::test]
async fn prompt_cache_retention_invalid_emits_warning() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_choice_response("gpt-4o")))
        .mount(&server)
        .await;

    let po = provider_options_with_openai(&json!({"promptCacheRetention": "bogus"}));
    let res = provider(&server)
        .chat("gpt-4o")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("ok");
    assert!(res.warnings.iter().any(|w| matches!(
        w,
        Warning::Unsupported { feature, .. } if feature == "openai.promptCacheRetention"
    )));
}

#[tokio::test]
async fn system_message_mode_remove_drops_system_with_warning() {
    let server = MockServer::start().await;
    // The outgoing body must NOT contain any system or developer role messages.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(
            json!({"messages": [{"role": "user", "content": "hi"}]}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_choice_response("gpt-4o")))
        .mount(&server)
        .await;

    let po = provider_options_with_openai(&json!({"systemMessageMode": "remove"}));
    let res = provider(&server)
        .chat("gpt-4o")
        .do_generate(CallOptions {
            prompt: vec![system("be helpful"), user_text("hi")],
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("ok");
    assert!(res.warnings.iter().any(|w| matches!(
        w,
        Warning::Other { message } if message.contains("system message removed")
    )));
}

#[tokio::test]
async fn system_message_mode_developer_forces_role() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "messages": [
                {"role": "developer", "content": "be helpful"},
                {"role": "user", "content": "hi"}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_choice_response("gpt-4o")))
        .mount(&server)
        .await;

    let po = provider_options_with_openai(&json!({"systemMessageMode": "developer"}));
    provider(&server)
        .chat("gpt-4o")
        .do_generate(CallOptions {
            prompt: vec![system("be helpful"), user_text("hi")],
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("forces developer role");
}

#[tokio::test]
async fn explicit_max_completion_tokens_takes_precedence() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "model": "o3",
            "max_completion_tokens": 999,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_choice_response("o3")))
        .mount(&server)
        .await;

    let po = provider_options_with_openai(&json!({"maxCompletionTokens": 999}));
    provider(&server)
        .chat("o3")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            // Even with max_output_tokens, the provider option wins.
            max_output_tokens: Some(64),
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn flex_processing_dropped_for_unsupported_model() {
    let server = MockServer::start().await;
    // gpt-4o does not support flex processing — `service_tier` must be omitted.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_choice_response("gpt-4o")))
        .mount(&server)
        .await;

    let po = provider_options_with_openai(&json!({"serviceTier": "flex"}));
    let res = provider(&server)
        .chat("gpt-4o")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("ok");
    assert!(res.warnings.iter().any(|w| matches!(
        w,
        Warning::Unsupported { feature, details } if feature == "serviceTier"
            && details.as_deref().unwrap_or("").contains("flex processing")
    )));
}

#[tokio::test]
async fn priority_processing_rejected_for_gpt5_nano() {
    let server = MockServer::start().await;
    // gpt-5-nano is on the priority denylist.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_choice_response("gpt-5-nano")))
        .mount(&server)
        .await;

    let po = provider_options_with_openai(&json!({"serviceTier": "priority"}));
    let res = provider(&server)
        .chat("gpt-5-nano")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("ok");
    assert!(res.warnings.iter().any(|w| matches!(
        w,
        Warning::Unsupported { feature, details } if feature == "serviceTier"
            && details.as_deref().unwrap_or("").contains("priority")
    )));
}

#[tokio::test]
async fn flex_processing_accepted_for_o3() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({"service_tier": "flex"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_choice_response("o3")))
        .mount(&server)
        .await;

    let po = provider_options_with_openai(&json!({"serviceTier": "flex"}));
    provider(&server)
        .chat("o3")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("flex relays for o3");
}
