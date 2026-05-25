//! Happy-path contract test for Azure `OpenAi::embedding`.
// Rust guideline compliant 2026-02-21

use llmsdk_azure::AzureOpenAi;
use llmsdk_provider::EmbeddingModel;
use llmsdk_provider::embedding_model::EmbedOptions;
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn happy_path_embeddings() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(query_param("api-version", "v1"))
        .and(header("api-key", "az-key"))
        .and(body_partial_json(json!({
            "model": "text-embedding-3-small",
            "input": ["hello", "world"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [
                { "object": "embedding", "index": 0, "embedding": [0.1, 0.2] },
                { "object": "embedding", "index": 1, "embedding": [0.3, 0.4] }
            ],
            "usage": { "prompt_tokens": 4, "total_tokens": 4 }
        })))
        .mount(&server)
        .await;

    let provider = AzureOpenAi::builder()
        .api_key("az-key")
        .base_url(server.uri())
        .build()
        .expect("provider builds");

    let model = provider.embedding("text-embedding-3-small");

    let result = model
        .do_embed(EmbedOptions {
            values: vec!["hello".into(), "world".into()],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(model.provider(), "azure.embeddings");
    assert_eq!(model.model_id(), "text-embedding-3-small");
    assert_eq!(result.embeddings.len(), 2);
    assert_eq!(result.embeddings[0], vec![0.1_f32, 0.2_f32]);
    assert_eq!(result.embeddings[1], vec![0.3_f32, 0.4_f32]);
    assert_eq!(result.usage.and_then(|u| u.tokens), Some(4));
}

#[tokio::test]
async fn embedding_model_alias_matches_embedding() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [
                { "object": "embedding", "index": 0, "embedding": [0.0] }
            ],
            "usage": { "prompt_tokens": 1, "total_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let provider = AzureOpenAi::builder()
        .api_key("az-key")
        .base_url(server.uri())
        .build()
        .expect("provider builds");

    let a = provider.embedding("text-embedding-3-small");
    let b = provider.embedding_model("text-embedding-3-small");

    assert_eq!(a.provider(), b.provider());
    assert_eq!(a.model_id(), b.model_id());
}
