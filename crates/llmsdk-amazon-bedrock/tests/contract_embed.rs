//! Contract tests for Bedrock embedding families.
// Rust guideline compliant 2026-05-25

use llmsdk_amazon_bedrock::AmazonBedrock;
use llmsdk_provider::EmbeddingModel;
use llmsdk_provider::embedding_model::EmbedOptions;
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
async fn titan_embed_round_trip() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/model/amazon.titan-embed-text-v2%3A0/invoke"))
        .and(body_partial_json(json!({ "inputText": "hello world" })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embedding": [0.1, 0.2, 0.3],
            "inputTextTokenCount": 4
        })))
        .mount(&server)
        .await;

    let model = provider(&server).embedding("amazon.titan-embed-text-v2:0");
    let result = model
        .do_embed(EmbedOptions {
            values: vec!["hello world".into()],
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(result.embeddings[0].len(), 3);
    assert_eq!(result.usage.unwrap().tokens, Some(4));
}

#[tokio::test]
async fn cohere_v3_embed_round_trip() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/model/cohere.embed-english-v3/invoke"))
        .and(body_partial_json(json!({
            "input_type": "search_query",
            "texts": ["hi"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": [[0.5, 0.5]]
        })))
        .mount(&server)
        .await;
    let result = provider(&server)
        .embedding("cohere.embed-english-v3")
        .do_embed(EmbedOptions {
            values: vec!["hi".into()],
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(result.embeddings[0].len(), 2);
    assert!(result.usage.is_none());
}
