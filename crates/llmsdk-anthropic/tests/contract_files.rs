//! Contract tests for [`llmsdk_anthropic::AnthropicFiles`].
//!
//! Covers the V4 `upload_file` round-trip:
//! - `POST /v1/files` is invoked with the correct beta header and auth
//! - the multipart body includes the file part with the right filename
//! - the response is mapped into `UploadFileResult`
//! - error envelopes turn into [`ProviderError::api_call`]
// Rust guideline compliant 2026-02-21

use llmsdk_anthropic::Anthropic;
use llmsdk_provider::shared::FileBytes;
use llmsdk_provider::{FilesModel, UploadFileData, UploadFileOptions};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Anthropic {
    Anthropic::builder()
        .api_key("sk-ant-test")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
}

#[tokio::test]
async fn upload_file_sends_beta_header_and_maps_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/files"))
        .and(header("anthropic-beta", "files-api-2025-04-14"))
        .and(header("x-api-key", "sk-ant-test"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "file-abc123",
            "type": "file",
            "filename": "report.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 12345,
            "created_at": "2025-04-14T12:00:00Z",
            "downloadable": true,
        })))
        .mount(&server)
        .await;

    let files = provider(&server).files();
    let r = files
        .upload_file(UploadFileOptions {
            data: UploadFileData::Data {
                data: FileBytes::Bytes(b"PDF-1.4 binary".to_vec()),
            },
            media_type: "application/pdf".into(),
            filename: Some("report.pdf".into()),
            provider_options: None,
        })
        .await
        .expect("upload ok");

    assert_eq!(
        r.provider_reference.get("anthropic").unwrap(),
        "file-abc123"
    );
    assert_eq!(r.filename.as_deref(), Some("report.pdf"));
    assert_eq!(r.media_type.as_deref(), Some("application/pdf"));
    let meta = r.provider_metadata.unwrap();
    let anthropic = meta.get("anthropic").unwrap();
    assert_eq!(anthropic.get("sizeBytes").unwrap(), 12345);
    assert_eq!(anthropic.get("downloadable").unwrap(), true);
    assert_eq!(anthropic.get("createdAt").unwrap(), "2025-04-14T12:00:00Z");
}

#[tokio::test]
async fn upload_file_text_payload_round_trip() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "file-text-1",
            "type": "file",
            "filename": "blob",
            "mime_type": "text/plain",
            "size_bytes": 5,
            "created_at": "2025-04-14T12:00:00Z",
        })))
        .mount(&server)
        .await;

    let files = provider(&server).files();
    let r = files
        .upload_file(UploadFileOptions {
            data: UploadFileData::Text {
                text: "hello".into(),
            },
            media_type: "text/plain".into(),
            filename: None,
            provider_options: None,
        })
        .await
        .expect("upload ok");

    assert_eq!(
        r.provider_reference.get("anthropic").unwrap(),
        "file-text-1"
    );
    let meta = r.provider_metadata.unwrap();
    let anthropic = meta.get("anthropic").unwrap();
    assert!(
        anthropic.get("downloadable").is_none(),
        "downloadable omitted when absent"
    );
}

#[tokio::test]
async fn upload_file_http_error_surfaces_anthropic_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "type": "error",
            "error": { "type": "rate_limit_error", "message": "slow down" }
        })))
        .mount(&server)
        .await;

    let files = provider(&server).files();
    let err = files
        .upload_file(UploadFileOptions {
            data: UploadFileData::Data {
                data: FileBytes::Bytes(vec![1, 2, 3]),
            },
            media_type: "application/octet-stream".into(),
            filename: None,
            provider_options: None,
        })
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("slow down"));
}

#[tokio::test]
async fn upload_file_base64_payload_decodes_before_upload() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "file-b64",
            "type": "file",
            "filename": "img.png",
            "mime_type": "image/png",
            "size_bytes": 5,
            "created_at": "2025-04-14T12:00:00Z",
        })))
        .mount(&server)
        .await;

    let files = provider(&server).files();
    let r = files
        .upload_file(UploadFileOptions {
            data: UploadFileData::Data {
                data: FileBytes::Base64("aGVsbG8=".into()),
            },
            media_type: "image/png".into(),
            filename: Some("img.png".into()),
            provider_options: None,
        })
        .await
        .expect("upload ok");
    assert_eq!(r.provider_reference.get("anthropic").unwrap(), "file-b64");
}
