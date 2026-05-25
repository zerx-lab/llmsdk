//! Contract tests for [`llmsdk_google::GoogleImageModel`] (Imagen path).
// Rust guideline compliant 2026-05-25

use llmsdk_google::Google;
use llmsdk_provider::ImageModel;
use llmsdk_provider::image_model::ImageOptions;
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
async fn imagen_predict_returns_decoded_bytes() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/imagen-4.0-generate-001:predict"))
        .and(body_partial_json(json!({
            "instances": [{"prompt":"a cat"}],
            "parameters": { "sampleCount": 2, "aspectRatio": "1:1" }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "predictions":[
                {"bytesBase64Encoded":"AAA="},
                {"bytesBase64Encoded":"AAA="}
            ]
        })))
        .mount(&server)
        .await;
    let model = provider(&server).image("imagen-4.0-generate-001");
    let r = model
        .do_generate(ImageOptions {
            prompt: "a cat".into(),
            n: Some(2),
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(r.images.len(), 2);
}

#[tokio::test]
async fn imagen_max_images_per_call_is_4() {
    let server = MockServer::start().await;
    let model = provider(&server).image("imagen-4.0-generate-001");
    assert_eq!(model.max_images_per_call().await, Some(4));
}

#[tokio::test]
async fn gemini_image_max_per_call_is_10() {
    let server = MockServer::start().await;
    let model = provider(&server).image("gemini-2.5-flash-image");
    assert_eq!(model.max_images_per_call().await, Some(10));
}
