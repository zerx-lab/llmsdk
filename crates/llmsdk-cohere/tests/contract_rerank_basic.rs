//! Contract tests for [`CohereRerankingModel::do_rerank`].
//!
//! Cohere is the first `RerankingModel` implementation in the workspace; these
//! tests exercise both `text` and `object` document variants plus the typed
//! provider options.
// Rust guideline compliant 2026-05-25

use llmsdk_cohere::Cohere;
use llmsdk_provider::RerankingModel;
use llmsdk_provider::reranking_model::{RerankingDocuments, RerankingOptions};
use llmsdk_provider::shared::ProviderOptions;
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Cohere {
    Cohere::builder()
        .api_key("k")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
}

#[tokio::test]
async fn happy_path_text_documents() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .and(header("authorization", "Bearer k"))
        .and(body_partial_json(json!({
            "model": "rerank-v3.5",
            "query": "best restaurant",
            "documents": ["pizza place", "sushi bar", "burger joint"],
            "top_n": 2
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "rerank-1",
            "results": [
                { "index": 1, "relevance_score": 0.92 },
                { "index": 0, "relevance_score": 0.41 }
            ],
            "meta": {}
        })))
        .mount(&server)
        .await;

    let result = provider(&server)
        .reranking("rerank-v3.5")
        .do_rerank(RerankingOptions {
            documents: RerankingDocuments::Text {
                values: vec![
                    "pizza place".into(),
                    "sushi bar".into(),
                    "burger joint".into(),
                ],
            },
            query: "best restaurant".into(),
            top_n: Some(2),
            headers: None,
            provider_options: None,
        })
        .await
        .expect("ok");

    assert_eq!(result.ranking.len(), 2);
    assert_eq!(result.ranking[0].index, 1);
    assert!((result.ranking[0].relevance_score - 0.92).abs() < 1e-6);
    assert!(result.warnings.is_empty());
    assert_eq!(result.response.unwrap().id.as_deref(), Some("rerank-1"));
}

#[tokio::test]
async fn object_documents_stringify_and_warn() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "rerank-2",
            "results": [
                { "index": 0, "relevance_score": 0.5 }
            ]
        })))
        .mount(&server)
        .await;

    let doc: serde_json::Map<String, serde_json::Value> = json!({"title": "x", "body": "y"})
        .as_object()
        .cloned()
        .unwrap();

    let result = provider(&server)
        .reranking("rerank-v3.5")
        .do_rerank(RerankingOptions {
            documents: RerankingDocuments::Object { values: vec![doc] },
            query: "q".into(),
            top_n: Some(1),
            headers: None,
            provider_options: None,
        })
        .await
        .expect("ok");

    assert_eq!(result.ranking.len(), 1);
    assert!(!result.warnings.is_empty());
}

#[tokio::test]
async fn forwards_max_tokens_per_doc_and_priority() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .and(body_partial_json(json!({
            "max_tokens_per_doc": 1024,
            "priority": 2
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "rerank-3",
            "results": []
        })))
        .mount(&server)
        .await;

    let mut po = ProviderOptions::new();
    po.insert(
        "cohere".into(),
        json!({"maxTokensPerDoc": 1024, "priority": 2})
            .as_object()
            .cloned()
            .unwrap(),
    );

    let _ = provider(&server)
        .reranking("rerank-v3.5")
        .do_rerank(RerankingOptions {
            documents: RerankingDocuments::Text {
                values: vec!["x".into()],
            },
            query: "q".into(),
            top_n: None,
            headers: None,
            provider_options: Some(po),
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn http_error_propagates() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({"message": "boom"})))
        .mount(&server)
        .await;

    let err = provider(&server)
        .reranking("rerank-v3.5")
        .do_rerank(RerankingOptions {
            documents: RerankingDocuments::Text {
                values: vec!["x".into()],
            },
            query: "q".into(),
            top_n: None,
            headers: None,
            provider_options: None,
        })
        .await
        .expect_err("must error");
    assert!(format!("{err}").contains("500"));
}

#[tokio::test]
async fn provider_id_and_model_id_correct() {
    let provider = Cohere::builder().api_key("k").build().expect("ok");
    let model = provider.reranking("rerank-v3.5");
    assert_eq!(model.provider(), "cohere");
    assert_eq!(model.model_id(), "rerank-v3.5");
    assert_eq!(model.specification_version(), "v4");
}
