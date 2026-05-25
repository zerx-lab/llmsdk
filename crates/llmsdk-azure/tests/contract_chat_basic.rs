//! Happy-path contract test for Azure `OpenAi::chat`.
//!
//! Mirrors `llmsdk-openai` contract chat basic but routes through the
//! Azure URL strategy + `api-key` auth header.
// Rust guideline compliant 2026-02-21

use llmsdk_azure::AzureOpenAi;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, FinishReasonKind, Message, TextPart, UserPart,
};
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn happy_path_returns_text() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(query_param("api-version", "v1"))
        .and(header("api-key", "az-key"))
        .and(body_partial_json(json!({
            "model": "gpt-4o-mini-deployment",
            "messages": [{ "role": "user", "content": "ping" }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-az-1",
            "created": 1_700_000_000_u64,
            "model": "gpt-4o-mini-deployment",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "pong" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3 }
        })))
        .mount(&server)
        .await;

    let provider = AzureOpenAi::builder()
        .api_key("az-key")
        .base_url(server.uri())
        .build()
        .expect("provider builds");

    let model = provider.chat("gpt-4o-mini-deployment");

    let result = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserPart::Text(TextPart {
                    text: "ping".into(),
                    provider_options: None,
                })],
                provider_options: None,
            }],
            ..Default::default()
        })
        .await
        .expect("call succeeds");

    assert_eq!(model.provider(), "azure.chat");
    assert_eq!(model.model_id(), "gpt-4o-mini-deployment");

    assert_eq!(result.content.len(), 1);
    let Content::Text(text) = &result.content[0] else {
        panic!("expected text content");
    };
    assert_eq!(text.text, "pong");
    assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
    assert_eq!(result.usage.input_tokens.total, Some(2));
}

#[tokio::test]
async fn missing_api_key_errors_when_env_unset() {
    // We never set AZURE_API_KEY in this binary, so build() without an
    // explicit key must fail.
    let err = AzureOpenAi::builder()
        .resource_name("ignored-since-build-fails-first")
        .build()
        .expect_err("builder should fail without api key");
    assert!(format!("{err}").contains("Azure OpenAI"));
}
