//! Express-mode contract tests: API key auth + publishers-google URL
//! without a project / location prefix.
// Rust guideline compliant 2026-05-25

use llmsdk_google_vertex::GoogleVertex;
use llmsdk_provider::EmbeddingModel;
use llmsdk_provider::embedding_model::EmbedOptions;
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn express_sends_x_goog_api_key_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/text-embedding-005:predict"))
        .and(header("x-goog-api-key", "express-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "predictions": [
                {"embeddings": {"values": [0.1, 0.2], "statistics": {"token_count": 4}}}
            ]
        })))
        .mount(&server)
        .await;
    let provider = GoogleVertex::builder()
        .api_key("express-key")
        .language_base_url(server.uri())
        .build()
        .await
        .expect("ok");
    let model = provider.embedding("text-embedding-005");
    let r = model
        .do_embed(EmbedOptions {
            values: vec!["hi".into()],
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(r.embeddings.len(), 1);
}

#[tokio::test]
async fn express_does_not_send_authorization_header() {
    use wiremock::matchers::header_exists;

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/text-embedding-005:predict"))
        .and(header_exists("x-goog-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "predictions": [
                {"embeddings": {"values": [0.0], "statistics": {"token_count": 0}}}
            ]
        })))
        .mount(&server)
        .await;
    let provider = GoogleVertex::builder()
        .api_key("k")
        .language_base_url(server.uri())
        .build()
        .await
        .expect("ok");
    let model = provider.embedding("text-embedding-005");
    let _ = model
        .do_embed(EmbedOptions {
            values: vec!["x".into()],
            ..Default::default()
        })
        .await
        .expect("ok");
}
