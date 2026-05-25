//! Vertex Imagen contract tests.
// Rust guideline compliant 2026-05-25

use llmsdk_google_vertex::GoogleVertex;
use llmsdk_provider::ImageModel;
use llmsdk_provider::image_model::ImageOptions;
use llmsdk_provider::language_model::FilePart;
use llmsdk_provider::shared::{FileBytes, FileData, ProviderOptions};
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
async fn imagen_generate_returns_decoded_bytes() {
    let server = MockServer::start().await;
    // Base64 for "PNG"
    let b64 = "UE5H";
    Mock::given(method("POST"))
        .and(path("/models/imagen-4.0-generate-001:predict"))
        .and(body_partial_json(json!({
            "instances": [{"prompt": "a cat"}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "predictions": [
                {"bytesBase64Encoded": b64, "mimeType": "image/png", "prompt": "revised: a cat"}
            ]
        })))
        .mount(&server)
        .await;
    let p = provider(&server).await;
    let m = p.image("imagen-4.0-generate-001");
    let r = m
        .do_generate(ImageOptions {
            prompt: "a cat".into(),
            n: Some(1),
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(r.images.len(), 1);
    assert_eq!(r.images[0].bytes.as_ref(), b"PNG");
    assert_eq!(r.images[0].media_type, "image/png");
    let meta = r.provider_metadata.unwrap();
    let gv = meta.get("googleVertex").unwrap();
    assert!(
        gv.get("images")
            .and_then(|v| v.as_array())
            .is_some_and(|arr| arr.iter().any(|o| o.get("revisedPrompt").is_some()))
    );
}

#[tokio::test]
async fn imagen_passes_through_negative_prompt_and_aspect_ratio() {
    let server = MockServer::start().await;
    let mut opts = ProviderOptions::new();
    opts.insert(
        "googleVertex".into(),
        json!({"negativePrompt": "blur"})
            .as_object()
            .unwrap()
            .clone(),
    );
    Mock::given(method("POST"))
        .and(path("/models/imagen-4.0-generate-001:predict"))
        .and(body_partial_json(json!({
            "parameters": {"aspectRatio": "16:9", "negativePrompt": "blur", "sampleCount": 2}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "predictions": [
                {"bytesBase64Encoded": "UE5H"}, {"bytesBase64Encoded": "UE5H"}
            ]
        })))
        .mount(&server)
        .await;
    let p = provider(&server).await;
    let m = p.image("imagen-4.0-generate-001");
    let r = m
        .do_generate(ImageOptions {
            prompt: "x".into(),
            n: Some(2),
            aspect_ratio: Some("16:9".into()),
            provider_options: Some(opts),
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(r.images.len(), 2);
}

#[tokio::test]
async fn imagen_edit_mode_emits_reference_images_with_mask() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/imagen-4.0-generate-001:predict"))
        .and(body_partial_json(json!({
            "instances": [{
                "prompt": "edit",
                "referenceImages": [
                    {"referenceType": "REFERENCE_TYPE_RAW", "referenceId": 1},
                    {"referenceType": "REFERENCE_TYPE_MASK", "referenceId": 2,
                     "maskImageConfig": {"maskMode": "MASK_MODE_USER_PROVIDED"}}
                ]
            }],
            "parameters": {"editMode": "EDIT_MODE_INPAINT_INSERTION"}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "predictions": [{"bytesBase64Encoded": "UE5H"}]
        })))
        .mount(&server)
        .await;
    let p = provider(&server).await;
    let m = p.image("imagen-4.0-generate-001");

    let src = FilePart {
        filename: None,
        data: FileData::Data {
            data: FileBytes::Base64("SU1H".into()),
        },
        media_type: "image/png".into(),
        provider_options: None,
    };
    let mask = FilePart {
        filename: None,
        data: FileData::Data {
            data: FileBytes::Base64("TQ==".into()),
        },
        media_type: "image/png".into(),
        provider_options: None,
    };

    let _ = m
        .do_generate(ImageOptions {
            prompt: "edit".into(),
            files: Some(vec![src]),
            mask: Some(mask),
            ..Default::default()
        })
        .await
        .expect("ok");
}
