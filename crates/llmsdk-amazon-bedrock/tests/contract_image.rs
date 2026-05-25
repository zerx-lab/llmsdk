//! Contract tests for the Bedrock image generation surface.
// Rust guideline compliant 2026-05-25

use llmsdk_amazon_bedrock::AmazonBedrock;
use llmsdk_provider::ImageModel;
use llmsdk_provider::image_model::ImageOptions;
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
async fn nova_canvas_text_image_round_trip() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/model/amazon.nova-canvas-v1%3A0/invoke"))
        .and(body_partial_json(json!({
            "taskType": "TEXT_IMAGE",
            "textToImageParams": { "text": "a cat" }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            // "iVBORw0KGgo=" is a 1-pixel PNG header — round-trip is base64.
            "images": ["iVBORw0KGgo="]
        })))
        .mount(&server)
        .await;
    let result = provider(&server)
        .image("amazon.nova-canvas-v1:0")
        .do_generate(ImageOptions {
            prompt: "a cat".into(),
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(result.images.len(), 1);
    assert!(result.images[0].bytes.len() >= 4); // PNG magic-ish
    assert_eq!(result.images[0].media_type, "image/png");
}

#[tokio::test]
async fn moderation_response_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "Request Moderated",
            "details": {
                "Moderation Reasons": ["Safety"]
            }
        })))
        .mount(&server)
        .await;
    let err = provider(&server)
        .image("amazon.nova-canvas-v1:0")
        .do_generate(ImageOptions {
            prompt: "x".into(),
            ..Default::default()
        })
        .await
        .expect_err("moderated");
    assert!(format!("{err}").contains("moderated"));
}
