//! Contract tests for the Bedrock Rerank API.
// Rust guideline compliant 2026-05-25

use llmsdk_amazon_bedrock::AmazonBedrock;
use llmsdk_provider::RerankingModel;
use llmsdk_provider::reranking_model::{RerankingDocuments, RerankingOptions};
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> AmazonBedrock {
    AmazonBedrock::builder()
        .region("us-east-1")
        .api_key("bearer-test")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
}

#[tokio::test]
async fn rerank_text_documents() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .and(body_partial_json(json!({
            "queries": [{ "type": "TEXT", "textQuery": { "text": "best fruit" } }],
            "sources": [
                { "type": "INLINE", "inlineDocumentSource": { "type": "TEXT", "textDocument": { "text": "apples" } } },
                { "type": "INLINE", "inlineDocumentSource": { "type": "TEXT", "textDocument": { "text": "bananas" } } }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                { "index": 1, "relevanceScore": 0.92 },
                { "index": 0, "relevanceScore": 0.31 }
            ]
        })))
        .mount(&server)
        .await;

    let result = provider(&server)
        .reranking("cohere.rerank-v3-5:0")
        .do_rerank(RerankingOptions {
            documents: RerankingDocuments::Text {
                values: vec!["apples".into(), "bananas".into()],
            },
            query: "best fruit".into(),
            top_n: Some(2),
            headers: None,
            provider_options: None,
        })
        .await
        .expect("ok");

    assert_eq!(result.ranking.len(), 2);
    assert_eq!(result.ranking[0].index, 1);
    assert!((result.ranking[0].relevance_score - 0.92).abs() < f64::EPSILON);
}
