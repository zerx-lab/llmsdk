//! Contract tests for [`XaiResponsesLanguageModel::do_generate`].
// Rust guideline compliant 2026-05-25

use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, FinishReasonKind, Message, TextPart, UserPart,
};
use llmsdk_xai::Xai;
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
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

#[tokio::test]
async fn happy_path_message_output() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer xai-test"))
        .and(body_partial_json(json!({
            "model": "grok-4.3",
            "input": [{
                "role": "user",
                "content": [{ "type": "input_text", "text": "hi" }]
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_1",
            "created_at": 1_700_000_000_u64,
            "model": "grok-4.3",
            "object": "response",
            "status": "completed",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": "hello from grok"
                }]
            }],
            "usage": {
                "input_tokens": 5,
                "output_tokens": 3
            }
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.responses("grok-4.3");

    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.content.len(), 1);
    let Content::Text(t) = &result.content[0] else {
        panic!("expected text");
    };
    assert_eq!(t.text, "hello from grok");
    assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
    assert_eq!(result.usage.input_tokens.total, Some(5));
    assert_eq!(result.usage.output_tokens.total, Some(3));
}

#[tokio::test]
async fn function_call_output_marks_tool_calls() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_2",
            "object": "response",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "weather",
                "arguments": "{\"city\":\"NYC\"}"
            }],
            "usage": {"input_tokens": 0, "output_tokens": 0}
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let result = provider
        .responses("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user_text("weather?")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.finish_reason.unified, FinishReasonKind::ToolCalls);
    assert_eq!(result.content.len(), 1);
    let Content::ToolCall(tc) = &result.content[0] else {
        panic!("expected ToolCall");
    };
    assert_eq!(tc.tool_call_id, "call_1");
    assert_eq!(tc.tool_name, "weather");
    assert_eq!(tc.input["city"], "NYC");
}

#[tokio::test]
async fn reasoning_block_with_encrypted_content_passes_through_metadata() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_3",
            "object": "response",
            "status": "completed",
            "output": [{
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{ "type": "summary_text", "text": "thinking" }],
                "status": "completed",
                "encrypted_content": "enc_xyz"
            }],
            "usage": {"input_tokens": 0, "output_tokens": 0}
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let result = provider
        .responses("grok-4.20-reasoning")
        .do_generate(CallOptions {
            prompt: vec![user_text("hello")],
            ..Default::default()
        })
        .await
        .expect("ok");

    let Content::Reasoning(rp) = &result.content[0] else {
        panic!("expected Reasoning");
    };
    assert_eq!(rp.text, "thinking");
    let po = rp.provider_options.as_ref().unwrap();
    assert_eq!(po["xai"]["itemId"], "rs_1");
    assert_eq!(po["xai"]["reasoningEncryptedContent"], "enc_xyz");
}

#[tokio::test]
async fn url_citation_annotation_emits_source() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_4",
            "object": "response",
            "status": "completed",
            "output": [{
                "type": "message",
                "id": "msg_2",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": "see [1]",
                    "annotations": [{
                        "type": "url_citation",
                        "url": "https://example.com/a",
                        "title": "A"
                    }]
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
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert!(
        result
            .content
            .iter()
            .any(|c| matches!(c, Content::Source(_)))
    );
}

#[tokio::test]
async fn cost_in_usd_ticks_passes_to_provider_metadata() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_5",
            "object": "response",
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_in_usd_ticks": 42
            }
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let result = provider
        .responses("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    let pm = result.provider_metadata.unwrap();
    assert_eq!(pm["xai"]["costInUsdTicks"], 42);
}
