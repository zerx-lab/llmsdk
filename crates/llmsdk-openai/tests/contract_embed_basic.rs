//! Contract tests for [`OpenAiEmbeddingModel::do_embed`].
// Rust guideline compliant 2026-02-21

use llmsdk_openai::OpenAi;
use llmsdk_provider::embedding_model::{EmbedOptions, EmbeddingModel};
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> OpenAi {
    OpenAi::builder()
        .api_key("sk-test")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
}

#[tokio::test]
async fn happy_path_returns_embeddings_and_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("authorization", "Bearer sk-test"))
        .and(body_partial_json(json!({
            "model": "text-embedding-3-small",
            "input": ["hello", "world"],
            "encoding_format": "float"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                { "embedding": [0.1, 0.2, 0.3] },
                { "embedding": [0.4, 0.5, 0.6] }
            ],
            "usage": { "prompt_tokens": 2 }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).embedding("text-embedding-3-small");
    assert_eq!(model.provider(), "openai");
    assert_eq!(model.model_id(), "text-embedding-3-small");
    assert_eq!(model.max_embeddings_per_call().await, Some(2048));

    let result = model
        .do_embed(EmbedOptions {
            values: vec!["hello".into(), "world".into()],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.embeddings.len(), 2);
    assert_eq!(result.embeddings[0], vec![0.1, 0.2, 0.3]);
    assert_eq!(result.embeddings[1], vec![0.4, 0.5, 0.6]);
    assert_eq!(result.usage.unwrap().tokens, Some(2));
}

#[tokio::test]
async fn relays_dimensions_provider_option() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(body_partial_json(json!({ "dimensions": 256 })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{ "embedding": [0.0] }]
        })))
        .mount(&server)
        .await;

    let mut provider_options = llmsdk_provider::shared::ProviderOptions::new();
    let mut openai = serde_json::Map::new();
    openai.insert("dimensions".into(), json!(256));
    provider_options.insert("openai".into(), openai);

    let model = provider(&server).embedding("text-embedding-3-small");
    let result = model
        .do_embed(EmbedOptions {
            values: vec!["x".into()],
            provider_options: Some(provider_options),
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(result.embeddings.len(), 1);
}

#[tokio::test]
async fn rejects_too_many_inputs() {
    let server = MockServer::start().await;
    let model = provider(&server).embedding("text-embedding-3-small");
    let values = (0..2049_u32).map(|i| format!("v{i}")).collect();
    let err = model
        .do_embed(EmbedOptions {
            values,
            ..Default::default()
        })
        .await
        .expect_err("should reject");
    assert!(format!("{err}").contains("too many embedding values"));
}

#[tokio::test]
async fn maps_429_with_openai_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": { "message": "embedding rate limit", "type": "requests" }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).embedding("text-embedding-3-small");
    let err = model
        .do_embed(EmbedOptions {
            values: vec!["x".into()],
            ..Default::default()
        })
        .await
        .expect_err("should error");
    assert!(err.is_api_call());
    assert!(err.is_retryable());
    assert_eq!(err.status_code(), Some(429));
    assert!(format!("{err}").contains("embedding rate limit"));
}

#[tokio::test]
async fn handles_response_without_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{ "embedding": [1.0] }]
        })))
        .mount(&server)
        .await;

    let model = provider(&server).embedding("text-embedding-3-small");
    let result = model
        .do_embed(EmbedOptions {
            values: vec!["x".into()],
            ..Default::default()
        })
        .await
        .expect("ok");
    assert!(result.usage.is_none());
}
