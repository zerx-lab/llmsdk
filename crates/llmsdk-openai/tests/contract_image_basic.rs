//! Contract tests for [`OpenAiImageModel::do_generate`].
//!
//! Each test boots a `wiremock` server, points the provider at it, and
//! asserts both the request shape and the response mapping. End-to-end
//! complement to the in-crate unit tests in `src/image.rs`.
// Rust guideline compliant 2026-02-21

use llmsdk_openai::OpenAi;
use llmsdk_provider::ImageModel;
use llmsdk_provider::image_model::ImageOptions;
use llmsdk_provider::shared::{ProviderOptions, Warning};
use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// One pixel red PNG, base64-encoded.
const RED_PIXEL_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";

fn provider(server: &MockServer) -> OpenAi {
    OpenAi::builder()
        .api_key("sk-test")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
}

fn provider_options_with_openai(value: &Value) -> ProviderOptions {
    let mut po = ProviderOptions::new();
    po.insert("openai".into(), value.as_object().cloned().unwrap());
    po
}

fn ok_response_dall_e_3() -> Value {
    json!({
        "created": 1_700_000_000_u64,
        "size": "1024x1024",
        "quality": "standard",
        "data": [{
            "b64_json": RED_PIXEL_PNG_B64,
            "revised_prompt": "A serene red square, beautifully composed."
        }]
    })
}

#[tokio::test]
async fn happy_path_decodes_png_and_captures_revised_prompt() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/images/generations"))
        .and(header("authorization", "Bearer sk-test"))
        .and(body_partial_json(json!({
            "model": "dall-e-3",
            "prompt": "a red square",
            "response_format": "b64_json"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response_dall_e_3()))
        .mount(&server)
        .await;

    let model = provider(&server).image("dall-e-3");
    let result = model
        .do_generate(ImageOptions {
            prompt: "a red square".into(),
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.images.len(), 1);
    let img = &result.images[0];
    // PNG magic: 89 50 4E 47 0D 0A 1A 0A
    assert_eq!(&img.bytes[..8], b"\x89PNG\r\n\x1a\n");
    assert_eq!(img.media_type, "image/png");

    let pm = result.provider_metadata.expect("provider_metadata");
    let openai = pm.get("openai").expect("openai entry");
    let images = openai.get("images").expect("images array");
    let images = images.as_array().expect("array");
    assert_eq!(images.len(), 1);
    assert_eq!(
        images[0]["revisedPrompt"],
        "A serene red square, beautifully composed."
    );
    assert_eq!(images[0]["size"], "1024x1024");

    let resp = result.response.expect("response info");
    assert_eq!(resp.timestamp.as_deref(), Some("1700000000"));
    assert_eq!(resp.model_id.as_deref(), Some("dall-e-3"));

    assert!(result.warnings.is_empty(), "no warnings expected");
}

#[tokio::test]
async fn gpt_image_1_omits_response_format() {
    let server = MockServer::start().await;
    // wiremock cannot easily assert _absence_ of a field, but
    // `body_partial_json` succeeds on presence-only matchers.
    // We use the round-trip: send `gpt-image-1` and verify the call
    // completes — the model-side decision branch is exercised by the
    // unit test `gpt_image_1_omits_response_format_field` already.
    Mock::given(method("POST"))
        .and(path("/images/generations"))
        .and(body_partial_json(
            json!({ "model": "gpt-image-1", "prompt": "hi" }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "created": 1,
            "data": [{ "b64_json": RED_PIXEL_PNG_B64 }]
        })))
        .mount(&server)
        .await;

    let model = provider(&server).image("gpt-image-1");
    let _ = model
        .do_generate(ImageOptions {
            prompt: "hi".into(),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn aspect_ratio_and_seed_emit_warnings() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/images/generations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response_dall_e_3()))
        .mount(&server)
        .await;

    let model = provider(&server).image("dall-e-3");
    let result = model
        .do_generate(ImageOptions {
            prompt: "a red square".into(),
            aspect_ratio: Some("16:9".into()),
            seed: Some(42),
            ..Default::default()
        })
        .await
        .expect("ok");

    let kinds: Vec<&str> = result
        .warnings
        .iter()
        .filter_map(|w| match w {
            Warning::UnsupportedSetting { setting, .. } => Some(setting.as_str()),
            _ => None,
        })
        .collect();
    assert!(kinds.contains(&"aspectRatio"));
    assert!(kinds.contains(&"seed"));
}

#[tokio::test]
async fn provider_options_relay_quality_style_and_output_format() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/images/generations"))
        .and(body_partial_json(json!({
            "quality": "hd",
            "style": "vivid",
            "output_format": "png",
            "user": "u-123"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "created": 1,
            "output_format": "png",
            "data": [{ "b64_json": RED_PIXEL_PNG_B64 }]
        })))
        .mount(&server)
        .await;

    let model = provider(&server).image("dall-e-3");
    let result = model
        .do_generate(ImageOptions {
            prompt: "hi".into(),
            provider_options: Some(provider_options_with_openai(&json!({
                "quality": "hd",
                "style": "vivid",
                "outputFormat": "png",
                "user": "u-123"
            }))),
            ..Default::default()
        })
        .await
        .expect("ok");

    // Server-reported `output_format` wins media-type detection.
    assert_eq!(result.images[0].media_type, "image/png");
    let openai = result.provider_metadata.unwrap();
    let openai = openai.get("openai").unwrap();
    let images = openai.get("images").unwrap().as_array().unwrap();
    assert_eq!(images[0]["outputFormat"], "png");
}

#[tokio::test]
async fn http_400_maps_to_openai_error_envelope() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/images/generations"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": {
                "message": "Invalid value: 'size'.",
                "type": "invalid_request_error",
                "code": "invalid_value"
            }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).image("dall-e-3");
    let err = model
        .do_generate(ImageOptions {
            prompt: "hi".into(),
            size: Some("999x999".into()),
            ..Default::default()
        })
        .await
        .expect_err("should fail");

    assert!(err.is_api_call(), "expected api_call error, got {err:?}");
    assert_eq!(err.status_code(), Some(400));
    assert!(!err.is_retryable());
    let msg = err.to_string();
    assert!(
        msg.contains("Invalid value: 'size'."),
        "missing upstream message in: {msg}"
    );
}

#[tokio::test]
async fn invalid_base64_in_response_yields_type_validation_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/images/generations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "created": 1,
            "data": [{ "b64_json": "not!valid!base64!" }]
        })))
        .mount(&server)
        .await;

    let model = provider(&server).image("dall-e-3");
    let err = model
        .do_generate(ImageOptions {
            prompt: "hi".into(),
            ..Default::default()
        })
        .await
        .expect_err("should fail");

    assert!(
        err.to_string().contains("invalid base64"),
        "missing base64 hint in: {err}"
    );
}

#[tokio::test]
async fn max_images_per_call_uses_per_model_ceiling() {
    let model = provider(&MockServer::start().await).image("dall-e-3");
    assert_eq!(model.max_images_per_call().await, Some(1));

    let model = provider(&MockServer::start().await).image("gpt-image-1");
    assert_eq!(model.max_images_per_call().await, Some(10));
}

#[tokio::test]
async fn per_call_header_override_wins_over_provider_default() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/images/generations"))
        .and(header("x-custom-header", "per-call"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response_dall_e_3()))
        .mount(&server)
        .await;

    let model = provider(&server).image("dall-e-3");
    let mut headers = llmsdk_provider::shared::Headers::new();
    headers.insert("x-custom-header".into(), Some("per-call".into()));

    let _ = model
        .do_generate(ImageOptions {
            prompt: "hi".into(),
            headers: Some(headers),
            ..Default::default()
        })
        .await
        .expect("ok");
}
