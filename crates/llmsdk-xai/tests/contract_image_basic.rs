//! Contract tests for [`XaiImageModel::do_generate`].
//!
//! Each test boots a `wiremock` server, points the provider at it, and
//! asserts both the outgoing request shape and the response mapping.
//! End-to-end complement to the in-crate unit tests under `src/image/*`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::ImageModel;
use llmsdk_provider::image_model::ImageOptions;
use llmsdk_provider::language_model::FilePart;
use llmsdk_provider::shared::{FileBytes, FileData, ProviderOptions};
use llmsdk_xai::Xai;
use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// One-pixel red PNG, base64-encoded.
const RED_PIXEL_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";

fn provider(server: &MockServer) -> Xai {
    Xai::builder()
        .api_key("xai-test")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
}

fn provider_options_with_xai(value: &Value) -> ProviderOptions {
    let mut po = ProviderOptions::new();
    po.insert("xai".into(), value.as_object().cloned().unwrap());
    po
}

fn ok_b64_response() -> Value {
    json!({
        "data": [{
            "b64_json": RED_PIXEL_PNG_B64,
            "revised_prompt": "A serene red square."
        }],
        "usage": { "cost_in_usd_ticks": 1234_u64 }
    })
}

#[tokio::test]
async fn generations_happy_path_decodes_png_and_collects_metadata() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/images/generations"))
        .and(header("authorization", "Bearer xai-test"))
        .and(body_partial_json(json!({
            "model": "grok-imagine-image",
            "prompt": "a red square",
            "response_format": "b64_json"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_b64_response()))
        .mount(&server)
        .await;

    let model = provider(&server).image("grok-imagine-image");
    let result = model
        .do_generate(ImageOptions {
            prompt: "a red square".into(),
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.images.len(), 1);
    let img = &result.images[0];
    // PNG magic header confirms the base64 was decoded into real bytes.
    assert_eq!(&img.bytes[..8], b"\x89PNG\r\n\x1a\n");
    assert_eq!(img.media_type, "image/png");

    let pm = result.provider_metadata.expect("provider_metadata");
    let xai = pm.get("xai").expect("xai entry");
    let images = xai
        .get("images")
        .and_then(|v| v.as_array())
        .expect("images array");
    assert_eq!(images.len(), 1);
    assert_eq!(images[0]["revisedPrompt"], "A serene red square.");
    assert_eq!(xai["costInUsdTicks"], 1234);

    let resp = result.response.expect("response info");
    assert_eq!(resp.model_id.as_deref(), Some("grok-imagine-image"));
    assert!(result.warnings.is_empty(), "no warnings expected");
}

#[tokio::test]
async fn provider_options_are_forwarded_to_wire() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/images/generations"))
        // All provider-option-derived fields must land on the wire body.
        .and(body_partial_json(json!({
            "model": "grok-imagine-image",
            "prompt": "a cat",
            "response_format": "b64_json",
            "aspect_ratio": "16:9",
            "output_format": "png",
            "sync_mode": true,
            "resolution": "2k",
            "quality": "high",
            "user": "alice@example.com"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_b64_response()))
        .mount(&server)
        .await;

    let model = provider(&server).image("grok-imagine-image");
    let opts = ImageOptions {
        prompt: "a cat".into(),
        provider_options: Some(provider_options_with_xai(&json!({
            "aspect_ratio": "16:9",
            "output_format": "png",
            "sync_mode": true,
            "resolution": "2k",
            "quality": "high",
            "user": "alice@example.com"
        }))),
        ..Default::default()
    };
    let result = model.do_generate(opts).await.expect("ok");
    assert_eq!(result.images.len(), 1);
}

#[tokio::test]
async fn size_seed_mask_emit_unsupported_warnings_but_call_still_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/images/generations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_b64_response()))
        .mount(&server)
        .await;

    let model = provider(&server).image("grok-imagine-image");
    let opts = ImageOptions {
        prompt: "hi".into(),
        size: Some("1024x1024".into()),
        seed: Some(7),
        mask: Some(FilePart {
            filename: None,
            data: FileData::Data {
                data: FileBytes::Bytes(vec![0xFF]),
            },
            media_type: "image/png".into(),
            provider_options: None,
        }),
        ..Default::default()
    };
    let result = model.do_generate(opts).await.expect("ok");
    let settings: Vec<&str> = result
        .warnings
        .iter()
        .filter_map(|w| match w {
            llmsdk_provider::shared::Warning::UnsupportedSetting { setting, .. } => {
                Some(setting.as_str())
            }
            _ => None,
        })
        .collect();
    assert!(settings.contains(&"size"));
    assert!(settings.contains(&"seed"));
    assert!(settings.contains(&"mask"));
}

#[tokio::test]
async fn files_route_to_edits_endpoint_with_single_image_field() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/images/edits"))
        .and(body_partial_json(json!({
            "model": "grok-imagine-image",
            "prompt": "make it blue",
            "response_format": "b64_json",
            "image": { "url": "data:image/png;base64,Zm9v", "type": "image_url" }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_b64_response()))
        .mount(&server)
        .await;

    let model = provider(&server).image("grok-imagine-image");
    let opts = ImageOptions {
        prompt: "make it blue".into(),
        files: Some(vec![FilePart {
            filename: None,
            data: FileData::Data {
                data: FileBytes::Bytes(b"foo".to_vec()),
            },
            media_type: "image/png".into(),
            provider_options: None,
        }]),
        ..Default::default()
    };
    let result = model.do_generate(opts).await.expect("ok");
    assert_eq!(result.images.len(), 1);
}

#[tokio::test]
async fn url_only_response_triggers_download_fallback() {
    let server = MockServer::start().await;
    // 1) /images/generations returns a URL pointing back at the same mock.
    let download_path = "/cdn/red.png";
    let download_url = format!("{}{download_path}", server.uri());
    Mock::given(method("POST"))
        .and(path("/images/generations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{ "url": download_url }]
        })))
        .mount(&server)
        .await;
    // 2) The downloader will GET that URL — return a real (one-pixel) PNG.
    let png_bytes: Vec<u8> = decode_b64_for_test(RED_PIXEL_PNG_B64);
    Mock::given(method("GET"))
        .and(path(download_path))
        .respond_with(ResponseTemplate::new(200).set_body_raw(png_bytes.clone(), "image/png"))
        .mount(&server)
        .await;

    let model = provider(&server).image("grok-imagine-image");
    let result = model
        .do_generate(ImageOptions {
            prompt: "url me".into(),
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(result.images.len(), 1);
    let img = &result.images[0];
    assert_eq!(img.bytes.as_ref(), png_bytes.as_slice());
    assert_eq!(img.media_type, "image/png");
}

#[tokio::test]
async fn max_images_per_call_reports_three() {
    let server = MockServer::start().await;
    let model = provider(&server).image("grok-imagine-image-pro");
    assert_eq!(model.max_images_per_call().await, Some(3));
    assert_eq!(model.provider(), "xai");
    assert_eq!(model.model_id(), "grok-imagine-image-pro");
}

// ---- helpers ---------------------------------------------------------

/// Tiny in-test base64 decoder so we can verify byte-for-byte equality on
/// the download path without re-exporting the model's internal decoder.
fn decode_b64_for_test(input: &str) -> Vec<u8> {
    fn dec(b: u8) -> Option<(u8, bool)> {
        Some(match b {
            b'A'..=b'Z' => (b - b'A', false),
            b'a'..=b'z' => (b - b'a' + 26, false),
            b'0'..=b'9' => (b - b'0' + 52, false),
            b'+' => (62, false),
            b'/' => (63, false),
            b'=' => (0, true),
            _ => return None,
        })
    }
    let bytes = input.as_bytes();
    assert!(bytes.len().is_multiple_of(4), "test base64 padded");
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks_exact(4) {
        let (b0, _) = dec(chunk[0]).expect("alphabet");
        let (b1, _) = dec(chunk[1]).expect("alphabet");
        let (b2, p2) = dec(chunk[2]).expect("alphabet");
        let (b3, p3) = dec(chunk[3]).expect("alphabet");
        let n =
            (u32::from(b0) << 18) | (u32::from(b1) << 12) | (u32::from(b2) << 6) | u32::from(b3);
        out.push(((n >> 16) & 0xFF) as u8);
        if !p2 {
            out.push(((n >> 8) & 0xFF) as u8);
        }
        if !p3 {
            out.push((n & 0xFF) as u8);
        }
    }
    out
}
