//! Contract tests for [`CohereEmbeddingModel::do_embed`].
// Rust guideline compliant 2026-05-25

use llmsdk_cohere::Cohere;
use llmsdk_provider::EmbeddingModel;
use llmsdk_provider::embedding_model::EmbedOptions;
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
async fn happy_path_embeddings_float() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .and(header("authorization", "Bearer k"))
        .and(body_partial_json(json!({
            "model": "embed-english-v3.0",
            "texts": ["hello", "world"],
            "embedding_types": ["float"],
            "input_type": "search_query"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": {
                "float": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            },
            "meta": {
                "billed_units": { "input_tokens": 2 }
            }
        })))
        .mount(&server)
        .await;

    let result = provider(&server)
        .embedding("embed-english-v3.0")
        .do_embed(EmbedOptions {
            values: vec!["hello".into(), "world".into()],
            headers: None,
            provider_options: None,
        })
        .await
        .expect("ok");

    assert_eq!(result.embeddings.len(), 2);
    assert_eq!(result.embeddings[0], vec![1.0, 2.0, 3.0]);
    assert_eq!(result.usage.unwrap().tokens, Some(2));
}

#[tokio::test]
async fn forwards_input_type_truncate_output_dimension() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .and(body_partial_json(json!({
            "input_type": "search_document",
            "truncate": "END",
            "output_dimension": 1024
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": { "float": [] }
        })))
        .mount(&server)
        .await;

    let mut po = ProviderOptions::new();
    po.insert(
        "cohere".into(),
        json!({
            "inputType": "search_document",
            "truncate": "END",
            "outputDimension": 1024
        })
        .as_object()
        .cloned()
        .unwrap(),
    );

    let _ = provider(&server)
        .embedding("embed-v4.0")
        .do_embed(EmbedOptions {
            values: vec!["doc".into()],
            headers: None,
            provider_options: Some(po),
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn rejects_more_than_96_inputs() {
    let server = MockServer::start().await;
    // Endpoint should not be hit.
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"embeddings":{"float":[]}})))
        .expect(0)
        .mount(&server)
        .await;

    let too_many: Vec<String> = (0..97).map(|i| format!("doc-{i}")).collect();
    let err = provider(&server)
        .embedding("embed-english-v3.0")
        .do_embed(EmbedOptions {
            values: too_many,
            headers: None,
            provider_options: None,
        })
        .await
        .expect_err("must error");
    assert!(format!("{err}").contains("96"));
}

#[tokio::test]
async fn surfaces_int8_in_provider_metadata() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": {
                "float": [],
                "int8": [[1, 2, 3]]
            }
        })))
        .mount(&server)
        .await;

    let mut po = ProviderOptions::new();
    po.insert(
        "cohere".into(),
        json!({"embeddingTypes": ["float", "int8"]})
            .as_object()
            .cloned()
            .unwrap(),
    );

    let result = provider(&server)
        .embedding("embed-english-v3.0")
        .do_embed(EmbedOptions {
            values: vec!["x".into()],
            headers: None,
            provider_options: Some(po),
        })
        .await
        .expect("ok");
    let pm = result.provider_metadata.expect("provider_metadata");
    let cohere = pm.get("cohere").unwrap();
    assert_eq!(cohere["embeddings"]["int8"], json!([[1, 2, 3]]));
}
