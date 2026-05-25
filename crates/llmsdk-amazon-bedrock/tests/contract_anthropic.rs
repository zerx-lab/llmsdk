//! Contract tests for the Anthropic-on-Bedrock `InvokeModel` path.
// Rust guideline compliant 2026-05-25

use llmsdk_amazon_bedrock::AmazonBedrock;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, FinishReasonKind, Message, TextPart, UserPart,
};
use serde_json::json;
use wiremock::matchers::{body_partial_json, header_exists, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> AmazonBedrock {
    AmazonBedrock::builder()
        .region("us-east-1")
        .api_key("bearer-test")
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
async fn invoke_endpoint_with_anthropic_version_injection() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(
            "/model/anthropic.claude-3-5-haiku-20241022-v1%3A0/invoke",
        ))
        .and(header_exists("authorization"))
        .and(body_partial_json(json!({
            "anthropic_version": "bedrock-2023-05-31"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "anthropic.claude-3-5-haiku-20241022-v1:0",
            "content": [{ "type": "text", "text": "hello from bedrock anthropic" }],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": { "input_tokens": 7, "output_tokens": 4 }
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider
        .anthropic("anthropic.claude-3-5-haiku-20241022-v1:0")
        .expect("model builds");

    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            max_output_tokens: Some(32),
            ..Default::default()
        })
        .await
        .expect("ok");

    let Content::Text(t) = &result.content[0] else {
        panic!("expected text");
    };
    assert_eq!(t.text, "hello from bedrock anthropic");
    assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
}
