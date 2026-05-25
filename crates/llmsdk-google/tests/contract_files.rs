//! Contract tests for the Gemini Files API.
// Rust guideline compliant 2026-05-25

use llmsdk_google::Google;
use llmsdk_provider::FilesModel;
use llmsdk_provider::shared::FileBytes;
use llmsdk_provider::{UploadFileData, UploadFileOptions};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Google {
    Google::builder()
        .api_key("test-key")
        .base_url(format!("{}/v1beta", server.uri()))
        .build()
        .expect("ok")
}

#[tokio::test]
async fn resumable_upload_two_step() {
    let server = MockServer::start().await;
    // Step 1: init returns x-goog-upload-url.
    Mock::given(method("POST"))
        .and(path("/upload/v1beta/files"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header(
                    "x-goog-upload-url",
                    format!("{}/upload/v1beta/files/session-123", server.uri()).as_str(),
                )
                .set_body_json(json!({})),
        )
        .mount(&server)
        .await;
    // Step 2: finalize returns file resource.
    Mock::given(method("POST"))
        .and(path("/upload/v1beta/files/session-123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "file": {
                "name": "files/abc-123",
                "displayName": "report.pdf",
                "mimeType": "application/pdf",
                "sizeBytes": "1024",
                "createTime": "2026-01-01T00:00:00Z",
                "uri": "https://example.com/files/abc-123",
                "state": "ACTIVE"
            }
        })))
        .mount(&server)
        .await;

    let files = provider(&server).files();
    let r = files
        .upload_file(UploadFileOptions {
            data: UploadFileData::Data {
                data: FileBytes::Bytes(vec![1, 2, 3, 4]),
            },
            media_type: "application/pdf".into(),
            filename: Some("local.pdf".into()),
            provider_options: None,
        })
        .await
        .expect("ok");
    assert!(r.provider_reference.contains_key("google"));
    assert_eq!(
        r.provider_reference.get("google").unwrap(),
        "https://example.com/files/abc-123"
    );
    assert_eq!(r.media_type.as_deref(), Some("application/pdf"));
}

#[tokio::test]
async fn processing_state_polls_until_active() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/upload/v1beta/files"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header(
                    "x-goog-upload-url",
                    format!("{}/upload/v1beta/files/session-x", server.uri()).as_str(),
                )
                .set_body_json(json!({})),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/upload/v1beta/files/session-x"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "file": {
                "name": "files/proc",
                "mimeType": "image/png",
                "uri": "https://example.com/files/proc",
                "state": "PROCESSING"
            }
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1beta/files/proc"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "name": "files/proc",
            "mimeType": "image/png",
            "uri": "https://example.com/files/proc",
            "state": "ACTIVE"
        })))
        .mount(&server)
        .await;
    let mut opts = llmsdk_provider::shared::ProviderOptions::new();
    opts.insert(
        "google".into(),
        json!({"pollIntervalMs":10,"pollTimeoutMs":5000})
            .as_object()
            .unwrap()
            .clone(),
    );

    let files = provider(&server).files();
    let r = files
        .upload_file(UploadFileOptions {
            data: UploadFileData::Data {
                data: FileBytes::Bytes(vec![0, 1]),
            },
            media_type: "image/png".into(),
            filename: None,
            provider_options: Some(opts),
        })
        .await
        .expect("ok");
    assert_eq!(
        r.provider_reference.get("google").unwrap(),
        "https://example.com/files/proc"
    );
}
