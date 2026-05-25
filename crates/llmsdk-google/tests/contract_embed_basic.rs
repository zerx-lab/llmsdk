//! Contract tests for [`llmsdk_google::GoogleEmbeddingModel`].
// Rust guideline compliant 2026-05-25

use llmsdk_google::Google;
use llmsdk_provider::EmbeddingModel;
use llmsdk_provider::embedding_model::EmbedOptions;
use llmsdk_provider::shared::ProviderOptions;
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Google {
    Google::builder()
        .api_key("k")
        .base_url(server.uri())
        .build()
        .expect("ok")
}

#[tokio::test]
#[allow(non_snake_case, reason = "endpoint name in test ident")]
async fn single_embed_uses_embedContent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/gemini-embedding-001:embedContent"))
        .and(body_partial_json(json!({
            "model": "models/gemini-embedding-001",
            "content": { "parts": [{"text": "hi"}] }
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"embedding": {"values": [0.1, 0.2, 0.3]}})),
        )
        .mount(&server)
        .await;
    let model = provider(&server).embedding("gemini-embedding-001");
    let r = model
        .do_embed(EmbedOptions {
            values: vec!["hi".into()],
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(r.embeddings.len(), 1);
    assert_eq!(r.embeddings[0].len(), 3);
}

#[tokio::test]
#[allow(non_snake_case, reason = "endpoint name in test ident")]
async fn batch_uses_batchEmbedContents() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/gemini-embedding-001:batchEmbedContents"))
        .and(body_partial_json(json!({
            "requests": [
                {"model": "models/gemini-embedding-001"},
                {"model": "models/gemini-embedding-001"}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": [
                {"values": [0.1, 0.2]},
                {"values": [0.3, 0.4]}
            ]
        })))
        .mount(&server)
        .await;
    let model = provider(&server).embedding("gemini-embedding-001");
    let r = model
        .do_embed(EmbedOptions {
            values: vec!["a".into(), "b".into()],
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(r.embeddings.len(), 2);
}

#[tokio::test]
async fn task_type_passthrough() {
    let server = MockServer::start().await;
    let mut opts = ProviderOptions::new();
    opts.insert(
        "google".into(),
        json!({"taskType":"RETRIEVAL_QUERY","outputDimensionality":256})
            .as_object()
            .unwrap()
            .clone(),
    );
    Mock::given(method("POST"))
        .and(path("/models/gemini-embedding-001:embedContent"))
        .and(body_partial_json(json!({
            "outputDimensionality": 256,
            "taskType": "RETRIEVAL_QUERY"
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"embedding":{"values":[0.0]}})),
        )
        .mount(&server)
        .await;
    let model = provider(&server).embedding("gemini-embedding-001");
    let _ = model
        .do_embed(EmbedOptions {
            values: vec!["hi".into()],
            provider_options: Some(opts),
            ..Default::default()
        })
        .await
        .expect("ok");
}
