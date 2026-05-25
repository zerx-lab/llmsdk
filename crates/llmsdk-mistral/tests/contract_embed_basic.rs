//! Contract tests for [`MistralEmbeddingModel::do_embed`].
// Rust guideline compliant 2026-05-25

use llmsdk_mistral::Mistral;
use llmsdk_provider::EmbeddingModel;
use llmsdk_provider::embedding_model::EmbedOptions;
use llmsdk_provider::shared::ProviderOptions;
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Mistral {
    Mistral::builder()
        .api_key("test-api-key")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
}

#[tokio::test]
async fn happy_path_returns_embeddings_and_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("authorization", "Bearer test-api-key"))
        .and(body_partial_json(json!({
            "model": "mistral-embed",
            "input": ["a sunny day", "a rainy day"],
            "encoding_format": "float"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "embed-1",
            "object": "list",
            "data": [
                { "object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0 },
                { "object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1 }
            ],
            "model": "mistral-embed",
            "usage": { "prompt_tokens": 8, "total_tokens": 8 }
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let model = provider.embedding("mistral-embed");
    let result = model
        .do_embed(EmbedOptions {
            values: vec!["a sunny day".into(), "a rainy day".into()],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.embeddings.len(), 2);
    assert_eq!(result.embeddings[0], vec![0.1, 0.2, 0.3]);
    assert_eq!(result.embeddings[1], vec![0.4, 0.5, 0.6]);
    assert_eq!(result.usage.unwrap().tokens, Some(8));
}

#[tokio::test]
async fn output_dimension_and_dtype_forwarded() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(body_partial_json(json!({
            "model": "codestral-embed",
            "output_dimension": 256,
            "output_dtype": "int8"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{ "object": "embedding", "embedding": [0.1], "index": 0 }],
            "model": "codestral-embed",
            "usage": { "prompt_tokens": 1, "total_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let mut po = ProviderOptions::new();
    po.insert(
        "mistral".into(),
        json!({"outputDimension": 256, "outputDtype": "int8"})
            .as_object()
            .cloned()
            .unwrap(),
    );
    let provider = provider(&server);
    let _ = provider
        .embedding("codestral-embed")
        .do_embed(EmbedOptions {
            values: vec!["hi".into()],
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn too_many_values_returns_error() {
    let server = MockServer::start().await;
    let provider = provider(&server);
    let model = provider.embedding("mistral-embed");
    let values: Vec<String> = (0..33).map(|i| format!("v{i}")).collect();
    let err = model
        .do_embed(EmbedOptions {
            values,
            ..Default::default()
        })
        .await
        .expect_err("must error");
    assert!(format!("{err}").contains("32"));
}

#[tokio::test]
async fn http_error_yields_provider_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "object": "error",
            "message": "unauthorized",
            "type": "invalid_api_key",
            "param": null,
            "code": "invalid_api_key"
        })))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let err = provider
        .embedding("mistral-embed")
        .do_embed(EmbedOptions {
            values: vec!["hi".into()],
            ..Default::default()
        })
        .await
        .expect_err("must error");
    assert!(format!("{err}").contains("401"));
}

#[tokio::test]
async fn embedding_model_provider_and_id_match() {
    let server = MockServer::start().await;
    let provider = provider(&server);
    let model = provider.embedding("mistral-embed");
    assert_eq!(model.provider(), "mistral");
    assert_eq!(model.model_id(), "mistral-embed");
    assert_eq!(model.max_embeddings_per_call().await, Some(32));
    assert!(!model.supports_parallel_calls().await);
}
