//! Contract tests for Anthropic extended-thinking support.
//!
//! Covers:
//!
//! - request: `thinking.type` + `budget_tokens` relayed, sampling settings
//!   stripped with warnings, `max_tokens` raised by the budget
//! - response: `thinking` / `redacted_thinking` blocks become
//!   `Content::Reasoning` with `signature` / `redactedData` carried on
//!   `provider_options.anthropic`
//! - outbound: `AssistantPart::Reasoning` round-trips through the
//!   `thinking` wire block, preserving signature + redacted data
//! - streaming: `thinking_delta` + `signature_delta` emit the matching
//!   `StreamPart::ReasoningStart/Delta/End` frames
// Rust guideline compliant 2026-02-21

use futures::StreamExt;
use llmsdk_anthropic::Anthropic;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    AssistantPart, CallOptions, Content, Message, StreamPart, TextPart, UserPart,
};
use llmsdk_provider::shared::{ProviderOptions, Warning};
use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, method, path};
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

fn provider_options_with_anthropic(value: &Value) -> ProviderOptions {
    let mut po = ProviderOptions::new();
    po.insert("anthropic".into(), value.as_object().cloned().unwrap());
    po
}

fn ok_response() -> Value {
    json!({
        "id": "msg_t",
        "type": "message",
        "model": "claude-3-7-sonnet",
        "content": [{ "type": "text", "text": "ok" }],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    })
}

#[tokio::test]
async fn thinking_enabled_strips_sampling_and_inflates_max_tokens() {
    let server = MockServer::start().await;
    // max_tokens = 256 (caller) + 1024 (budget) = 1280.
    // Sampling settings (`temperature`, `top_p`, `top_k`) must be ABSENT.
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "max_tokens": 1280,
            "thinking": { "type": "enabled", "budget_tokens": 1024 }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let model = provider(&server).messages("claude-3-7-sonnet");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("think hard")],
            max_output_tokens: Some(256),
            temperature: Some(0.5),
            top_p: Some(0.9),
            top_k: Some(40),
            provider_options: Some(provider_options_with_anthropic(&json!({
                "thinking": { "type": "enabled", "budgetTokens": 1024 }
            }))),
            ..Default::default()
        })
        .await
        .expect("ok");

    let stripped: Vec<&str> = result
        .warnings
        .iter()
        .filter_map(|w| match w {
            Warning::UnsupportedSetting { setting, .. } => Some(setting.as_str()),
            _ => None,
        })
        .collect();
    assert!(stripped.contains(&"temperature"));
    assert!(stripped.contains(&"topP"));
    assert!(stripped.contains(&"topK"));
}

#[tokio::test]
async fn thinking_disabled_round_trips_disabled_marker() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(
            json!({ "thinking": { "type": "disabled" } }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let model = provider(&server).messages("claude-3-7-sonnet");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(provider_options_with_anthropic(&json!({
                "thinking": { "type": "disabled" }
            }))),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn response_thinking_becomes_reasoning_with_signature() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_t2",
            "type": "message",
            "model": "claude-3-7-sonnet",
            "content": [
                { "type": "thinking", "thinking": "step 1...", "signature": "sig_abc" },
                { "type": "redacted_thinking", "data": "OPAQUE" },
                { "type": "text", "text": "answer" }
            ],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 5, "output_tokens": 9 }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).messages("claude-3-7-sonnet");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    // Expect: Reasoning("step 1...") with signature; Reasoning("") with
    // redactedData; Text("answer").
    assert_eq!(result.content.len(), 3);
    if let Content::Reasoning(r) = &result.content[0] {
        assert_eq!(r.text, "step 1...");
        let po = r.provider_options.as_ref().expect("provider_options");
        assert_eq!(po["anthropic"]["signature"], "sig_abc");
    } else {
        panic!("expected Reasoning at index 0, got {:?}", result.content[0]);
    }
    if let Content::Reasoning(r) = &result.content[1] {
        assert!(r.text.is_empty());
        let po = r.provider_options.as_ref().expect("provider_options");
        assert_eq!(po["anthropic"]["redactedData"], "OPAQUE");
    } else {
        panic!("expected Reasoning at index 1");
    }
    if let Content::Text(t) = &result.content[2] {
        assert_eq!(t.text, "answer");
    } else {
        panic!("expected Text at index 2");
    }
}

#[tokio::test]
async fn outbound_reasoning_message_emits_thinking_block_with_signature() {
    let server = MockServer::start().await;
    // Confirm the wire shape: the assistant turn carries a thinking block
    // with the embedded signature.
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "messages": [
                { "role": "user", "content": [{"type":"text","text":"hi"}] },
                { "role": "assistant", "content": [
                    {"type":"thinking","thinking":"prior step","signature":"sig_xyz"},
                    {"type":"text","text":"answer"}
                ]},
                { "role": "user", "content": [{"type":"text","text":"again"}] }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let model = provider(&server).messages("claude-3-7-sonnet");
    let mut anthropic = serde_json::Map::new();
    anthropic.insert("signature".into(), json!("sig_xyz"));
    let mut po = ProviderOptions::new();
    po.insert("anthropic".into(), anthropic);

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![
                user_text("hi"),
                Message::Assistant {
                    content: vec![
                        AssistantPart::Reasoning {
                            text: "prior step".into(),
                            provider_options: Some(po),
                        },
                        AssistantPart::Text(TextPart {
                            text: "answer".into(),
                            provider_options: None,
                        }),
                    ],
                    provider_options: None,
                },
                user_text("again"),
            ],
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn stream_thinking_and_signature_delta_emit_reasoning_frames() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s\",\"model\":\"claude-3-7-sonnet\",\"usage\":{\"input_tokens\":1}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"musing...\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_stream\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&server)
        .await;

    let model = provider(&server).messages("claude-3-7-sonnet");
    let mut stream = model
        .do_stream(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("stream ok")
        .stream;

    let mut got_start = false;
    let mut got_delta = false;
    let mut got_signature = false;
    let mut got_end = false;
    while let Some(item) = stream.next().await {
        let part = item.expect("stream part");
        match part {
            StreamPart::ReasoningStart { .. } => got_start = true,
            StreamPart::ReasoningDelta {
                delta,
                provider_metadata,
                ..
            } => {
                if delta == "musing..." {
                    got_delta = true;
                } else if delta.is_empty()
                    && provider_metadata
                        .as_ref()
                        .and_then(|m| m.get("anthropic"))
                        .and_then(|a| a.get("signature"))
                        .and_then(|v| v.as_str())
                        == Some("sig_stream")
                {
                    got_signature = true;
                }
            }
            StreamPart::ReasoningEnd { .. } => got_end = true,
            _ => {}
        }
    }
    assert!(got_start, "missing ReasoningStart");
    assert!(got_delta, "missing thinking_delta -> ReasoningDelta");
    assert!(got_signature, "missing signature_delta -> ReasoningDelta");
    assert!(got_end, "missing ReasoningEnd");
}

#[tokio::test]
async fn stream_redacted_thinking_attaches_metadata_on_start() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_r\",\"model\":\"claude-3-7-sonnet\",\"usage\":{\"input_tokens\":1}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"OPAQUE\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&server)
        .await;

    let model = provider(&server).messages("claude-3-7-sonnet");
    let mut stream = model
        .do_stream(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("stream ok")
        .stream;

    let mut redacted_data = None;
    while let Some(item) = stream.next().await {
        let part = item.expect("stream part");
        if let StreamPart::ReasoningStart {
            provider_metadata, ..
        } = part
        {
            redacted_data = provider_metadata
                .as_ref()
                .and_then(|m| m.get("anthropic"))
                .and_then(|a| a.get("redactedData"))
                .and_then(|v| v.as_str())
                .map(str::to_owned);
        }
    }
    assert_eq!(redacted_data.as_deref(), Some("OPAQUE"));
}
