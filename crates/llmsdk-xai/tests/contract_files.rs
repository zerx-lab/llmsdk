//! Contract tests for [`llmsdk_xai::XaiFiles`].
//!
//! Covers the V4 `upload_file` round-trip against a wiremock server:
//!
//! - `POST {base_url}/files` is invoked with bearer auth + multipart body
//! - the multipart body includes the file part and (optionally) `team_id`
//! - the JSON response is mapped into `UploadFileResult` per ai-sdk rules
//! - HTTP errors surface the xAI error message
//! - explicit `null` fields in the response are omitted from
//!   `provider_metadata.xai.*`
// Rust guideline compliant 2026-05-25

use llmsdk_provider::shared::FileBytes;
use llmsdk_provider::{FilesModel, UploadFileData, UploadFileOptions};
use llmsdk_xai::Xai;
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn provider(server: &MockServer) -> Xai {
    Xai::builder()
        .api_key("xai-test")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
}

#[tokio::test]
async fn upload_file_sends_multipart_post_with_bearer_auth() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/files"))
        .and(header("authorization", "Bearer xai-test"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "file-abc123",
            "object": "file",
            "bytes": 3_u64,
            "created_at": 1_234_567_890_u64,
            "filename": "upload",
        })))
        .mount(&server)
        .await;

    let files = provider(&server).files();
    let r = files
        .upload_file(UploadFileOptions {
            data: UploadFileData::Data {
                data: FileBytes::Bytes(vec![1, 2, 3]),
            },
            media_type: "application/octet-stream".into(),
            filename: None,
            provider_options: None,
        })
        .await
        .expect("upload ok");

    assert_eq!(r.provider_reference.get("xai").unwrap(), "file-abc123");
    assert_eq!(r.media_type.as_deref(), Some("application/octet-stream"));
    // filename echoed from server (overrides absent input).
    assert_eq!(r.filename.as_deref(), Some("upload"));
    let meta = r.provider_metadata.unwrap();
    let xai_meta = meta.get("xai").unwrap();
    assert_eq!(xai_meta.get("filename").unwrap(), "upload");
    assert_eq!(xai_meta.get("bytes").unwrap(), 3);
    assert_eq!(xai_meta.get("createdAt").unwrap(), 1_234_567_890_u64);
    assert!(r.warnings.is_empty());
}

#[tokio::test]
async fn upload_file_includes_filename_and_team_id_in_multipart_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "file-xyz789",
            "object": "file",
            "bytes": 14_u64,
            "created_at": 1_700_000_000_u64,
            "filename": "report.pdf",
        })))
        .mount(&server)
        .await;

    let files = provider(&server).files();
    let mut po = llmsdk_provider::shared::ProviderOptions::default();
    po.insert(
        "xai".into(),
        json!({ "teamId": "team-123" }).as_object().unwrap().clone(),
    );
    let r = files
        .upload_file(UploadFileOptions {
            data: UploadFileData::Data {
                data: FileBytes::Bytes(b"PDF-1.4 binary".to_vec()),
            },
            media_type: "application/pdf".into(),
            filename: Some("report.pdf".into()),
            provider_options: Some(po),
        })
        .await
        .expect("upload ok");

    assert_eq!(r.provider_reference.get("xai").unwrap(), "file-xyz789");
    assert_eq!(r.filename.as_deref(), Some("report.pdf"));

    // Inspect the captured multipart body for correctness.
    let received = server.received_requests().await.expect("requests recorded");
    let req: &Request = received.first().expect("got a request");
    let ct = req
        .headers
        .get("content-type")
        .expect("content-type header")
        .to_str()
        .unwrap();
    assert!(
        ct.starts_with("multipart/form-data; boundary="),
        "expected multipart content-type, got: {ct}"
    );
    let body = std::str::from_utf8(&req.body).expect("utf8 body");
    assert!(
        body.contains("name=\"file\"; filename=\"report.pdf\""),
        "missing file part header in body: {body}"
    );
    assert!(
        body.contains("Content-Type: application/pdf"),
        "missing file part content-type in body: {body}"
    );
    assert!(
        body.contains("PDF-1.4 binary"),
        "missing file bytes in body: {body}"
    );
    assert!(
        body.contains("name=\"team_id\""),
        "missing team_id form field in body: {body}"
    );
    assert!(
        body.contains("team-123"),
        "missing team_id value in body: {body}"
    );
}

#[tokio::test]
async fn upload_file_uses_default_filename_blob_when_none_supplied() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "file-1",
            "object": "file",
            // server returns no filename → result.filename falls back to None
        })))
        .mount(&server)
        .await;

    let files = provider(&server).files();
    files
        .upload_file(UploadFileOptions {
            data: UploadFileData::Data {
                data: FileBytes::Bytes(vec![9]),
            },
            media_type: "application/octet-stream".into(),
            filename: None,
            provider_options: None,
        })
        .await
        .expect("upload ok");

    let received = server.received_requests().await.expect("requests recorded");
    let body = std::str::from_utf8(&received[0].body).expect("utf8 body");
    // Upstream's `formData.append('file', blob)` (no filename) sets
    // "blob" as the filename. We mirror that explicitly.
    assert!(
        body.contains("name=\"file\"; filename=\"blob\""),
        "expected default filename=\"blob\" in body: {body}"
    );
}

#[tokio::test]
async fn upload_file_does_not_include_team_id_when_absent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "file-1",
            "object": "file",
        })))
        .mount(&server)
        .await;

    let files = provider(&server).files();
    files
        .upload_file(UploadFileOptions {
            data: UploadFileData::Data {
                data: FileBytes::Bytes(vec![1]),
            },
            media_type: "application/octet-stream".into(),
            filename: None,
            provider_options: None,
        })
        .await
        .expect("upload ok");

    let received = server.received_requests().await.expect("requests recorded");
    let body = std::str::from_utf8(&received[0].body).expect("utf8 body");
    assert!(
        !body.contains("name=\"team_id\""),
        "team_id must be omitted when not supplied: {body}"
    );
}

#[tokio::test]
async fn upload_file_base64_payload_is_decoded_before_upload() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "file-b64",
            "object": "file",
            "bytes": 4_u64,
            "filename": "blob",
        })))
        .mount(&server)
        .await;

    let files = provider(&server).files();
    let r = files
        .upload_file(UploadFileOptions {
            data: UploadFileData::Data {
                // "test" → "dGVzdA==" (RFC 4648)
                data: FileBytes::Base64("dGVzdA==".into()),
            },
            media_type: "application/octet-stream".into(),
            filename: None,
            provider_options: None,
        })
        .await
        .expect("upload ok");
    assert_eq!(r.provider_reference.get("xai").unwrap(), "file-b64");

    let received = server.received_requests().await.expect("requests recorded");
    let body = &received[0].body;
    // The bytes "test" (0x74 0x65 0x73 0x74) must appear in the multipart body.
    assert!(
        body.windows(4).any(|w| w == b"test"),
        "decoded base64 payload missing from body"
    );
}

#[tokio::test]
async fn upload_file_omits_null_response_fields_from_provider_metadata() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "file-abc123",
            "object": "file",
            "bytes": null,
            "created_at": null,
            "filename": null,
        })))
        .mount(&server)
        .await;

    let files = provider(&server).files();
    let r = files
        .upload_file(UploadFileOptions {
            data: UploadFileData::Data {
                data: FileBytes::Bytes(vec![1]),
            },
            media_type: "application/octet-stream".into(),
            filename: None,
            provider_options: None,
        })
        .await
        .expect("upload ok");

    let meta = r.provider_metadata.unwrap();
    let xai_meta = meta.get("xai").unwrap();
    assert!(xai_meta.get("filename").is_none());
    assert!(xai_meta.get("bytes").is_none());
    assert!(xai_meta.get("createdAt").is_none());
    // filename in the result also falls back to None when neither server
    // nor caller supplied one.
    assert!(r.filename.is_none());
}

#[tokio::test]
async fn upload_file_falls_back_to_input_filename_when_server_omits_it() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "file-2",
            "object": "file",
            // No `filename` in response.
        })))
        .mount(&server)
        .await;

    let files = provider(&server).files();
    let r = files
        .upload_file(UploadFileOptions {
            data: UploadFileData::Data {
                data: FileBytes::Bytes(vec![1, 2, 3]),
            },
            media_type: "image/png".into(),
            filename: Some("input-name.png".into()),
            provider_options: None,
        })
        .await
        .expect("upload ok");

    assert_eq!(r.filename.as_deref(), Some("input-name.png"));
}

#[tokio::test]
async fn upload_file_http_error_surfaces_xai_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": {
                "message": "slow down",
                "type": "rate_limit_error",
                "code": "rate_limited"
            }
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
    let msg = format!("{err}");
    assert!(
        msg.contains("slow down") || msg.contains("rate_limit") || msg.contains("429"),
        "expected xAI error context in: {msg}"
    );
}

#[tokio::test]
async fn upload_file_specification_version_and_provider_id() {
    // Pure builder check — does not hit the server.
    let server = MockServer::start().await;
    let files = provider(&server).files();
    assert_eq!(files.specification_version(), "v4");
    assert_eq!(files.provider(), "xai.files");
}
