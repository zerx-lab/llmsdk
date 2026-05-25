//! Vertex embedding contract tests.
// Rust guideline compliant 2026-05-25

use llmsdk_google_vertex::GoogleVertex;
use llmsdk_provider::EmbeddingModel;
use llmsdk_provider::embedding_model::EmbedOptions;
use llmsdk_provider::shared::ProviderOptions;
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn provider(server: &MockServer) -> GoogleVertex {
    GoogleVertex::builder()
        .api_key("k")
        .language_base_url(server.uri())
        .build()
        .await
        .expect("ok")
}

#[tokio::test]
async fn single_value_uses_predict_endpoint_with_instances_array() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/text-embedding-005:predict"))
        .and(body_partial_json(json!({
            "instances": [{"content": "hello"}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "predictions": [
                {"embeddings": {"values": [1.0, 2.0, 3.0], "statistics": {"token_count": 3}}}
            ]
        })))
        .mount(&server)
        .await;
    let p = provider(&server).await;
    let m = p.embedding("text-embedding-005");
    let r = m
        .do_embed(EmbedOptions {
            values: vec!["hello".into()],
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(r.embeddings.len(), 1);
    assert_eq!(r.embeddings[0], vec![1.0, 2.0, 3.0]);
    assert_eq!(r.usage.and_then(|u| u.tokens), Some(3));
}

#[tokio::test]
async fn batch_value_collects_all_predictions() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/text-embedding-005:predict"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "predictions": [
                {"embeddings": {"values": [0.1], "statistics": {"token_count": 1}}},
                {"embeddings": {"values": [0.2], "statistics": {"token_count": 2}}},
                {"embeddings": {"values": [0.3], "statistics": {"token_count": 4}}}
            ]
        })))
        .mount(&server)
        .await;
    let p = provider(&server).await;
    let m = p.embedding("text-embedding-005");
    let r = m
        .do_embed(EmbedOptions {
            values: vec!["a".into(), "b".into(), "c".into()],
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(r.embeddings.len(), 3);
    assert_eq!(r.usage.and_then(|u| u.tokens), Some(1 + 2 + 4));
}

#[tokio::test]
async fn task_type_and_title_routed_into_instances() {
    let server = MockServer::start().await;
    let mut opts = ProviderOptions::new();
    opts.insert(
        "googleVertex".into(),
        json!({
            "taskType": "RETRIEVAL_DOCUMENT",
            "title": "doc",
            "outputDimensionality": 64,
            "autoTruncate": false
        })
        .as_object()
        .unwrap()
        .clone(),
    );
    Mock::given(method("POST"))
        .and(path("/models/text-embedding-005:predict"))
        .and(body_partial_json(json!({
            "instances": [{"content": "x", "task_type": "RETRIEVAL_DOCUMENT", "title": "doc"}],
            "parameters": {"outputDimensionality": 64, "autoTruncate": false}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "predictions": [
                {"embeddings": {"values": [0.0], "statistics": {"token_count": 1}}}
            ]
        })))
        .mount(&server)
        .await;
    let p = provider(&server).await;
    let m = p.embedding("text-embedding-005");
    let _ = m
        .do_embed(EmbedOptions {
            values: vec!["x".into()],
            provider_options: Some(opts),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn too_many_values_returns_error_before_call() {
    let server = MockServer::start().await;
    // No mock — request should not reach the server.
    let p = provider(&server).await;
    let m = p.embedding("text-embedding-005");
    let values: Vec<String> = (0..2049).map(|i| format!("v{i}")).collect();
    let err = m
        .do_embed(EmbedOptions {
            values,
            ..Default::default()
        })
        .await
        .expect_err("over the limit");
    assert!(format!("{err}").contains("2048"));
}
