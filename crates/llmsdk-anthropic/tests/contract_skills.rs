//! Contract tests for [`llmsdk_anthropic::AnthropicSkills`].
// Rust guideline compliant 2026-02-21

use llmsdk_anthropic::Anthropic;
use llmsdk_provider::shared::FileBytes;
use llmsdk_provider::{SkillFile, SkillsModel, UploadFileData, UploadSkillOptions};
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
async fn upload_skill_basic_round_trip_with_version_metadata_fetch() {
    let server = MockServer::start().await;

    // POST /skills
    Mock::given(method("POST"))
        .and(path("/skills"))
        .and(header("anthropic-beta", "skills-2025-10-02"))
        .and(header("x-api-key", "sk-ant-test"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "skill-abc",
            "display_title": "Greeter",
            "name": null,
            "description": null,
            "latest_version": "v3",
            "source": "user",
            "created_at": "2025-10-02T00:00:00Z",
            "updated_at": "2025-10-02T00:00:00Z",
        })))
        .mount(&server)
        .await;

    // GET /skills/skill-abc/versions/v3
    Mock::given(method("GET"))
        .and(path("/skills/skill-abc/versions/v3"))
        .and(header("anthropic-beta", "skills-2025-10-02"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "type": "skill_version",
            "skill_id": "skill-abc",
            "name": "greeter-v3",
            "description": "Says hi (v3)",
        })))
        .mount(&server)
        .await;

    let skills = provider(&server).skills();
    let r = skills
        .upload_skill(UploadSkillOptions {
            files: vec![SkillFile {
                path: "main.py".into(),
                data: UploadFileData::Text {
                    text: "print('hi')".into(),
                },
            }],
            display_title: Some("Greeter".into()),
            provider_options: None,
        })
        .await
        .expect("upload ok");

    assert_eq!(r.provider_reference.get("anthropic").unwrap(), "skill-abc");
    assert_eq!(r.display_title.as_deref(), Some("Greeter"));
    assert_eq!(r.latest_version.as_deref(), Some("v3"));
    // Name / description come from the version metadata, not the upload body.
    assert_eq!(r.name.as_deref(), Some("greeter-v3"));
    assert_eq!(r.description.as_deref(), Some("Says hi (v3)"));
    let meta = r.provider_metadata.unwrap();
    let anthropic = meta.get("anthropic").unwrap();
    assert_eq!(anthropic.get("source").unwrap(), "user");
    assert_eq!(anthropic.get("createdAt").unwrap(), "2025-10-02T00:00:00Z");
}

#[tokio::test]
async fn upload_skill_without_latest_version_skips_followup_get() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/skills"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "skill-no-version",
            "name": "fallback-name",
            "description": "fallback-desc",
            "source": "user",
            "created_at": "2025-10-02T00:00:00Z",
            "updated_at": "2025-10-02T00:00:00Z",
        })))
        .mount(&server)
        .await;

    let skills = provider(&server).skills();
    let r = skills
        .upload_skill(UploadSkillOptions {
            files: vec![SkillFile {
                path: "a.bin".into(),
                data: UploadFileData::Data {
                    data: FileBytes::Bytes(vec![0u8; 4]),
                },
            }],
            display_title: None,
            provider_options: None,
        })
        .await
        .expect("upload ok");

    // Falls back to the upload response's name/description.
    assert_eq!(r.name.as_deref(), Some("fallback-name"));
    assert_eq!(r.description.as_deref(), Some("fallback-desc"));
    assert!(r.latest_version.is_none());
}

#[tokio::test]
async fn upload_skill_multi_file_bundle() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/skills"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "skill-multi",
            "source": "user",
            "created_at": "2025-10-02T00:00:00Z",
            "updated_at": "2025-10-02T00:00:00Z",
        })))
        .mount(&server)
        .await;

    let skills = provider(&server).skills();
    let r = skills
        .upload_skill(UploadSkillOptions {
            files: vec![
                SkillFile {
                    path: "main.py".into(),
                    data: UploadFileData::Text {
                        text: "print(1)".into(),
                    },
                },
                SkillFile {
                    path: "lib/helpers.py".into(),
                    data: UploadFileData::Text {
                        text: "def x(): pass".into(),
                    },
                },
                SkillFile {
                    path: "assets/data.bin".into(),
                    data: UploadFileData::Data {
                        data: FileBytes::Bytes(vec![1, 2, 3]),
                    },
                },
            ],
            display_title: None,
            provider_options: None,
        })
        .await
        .expect("upload ok");
    assert_eq!(
        r.provider_reference.get("anthropic").unwrap(),
        "skill-multi"
    );
}

#[tokio::test]
async fn upload_skill_http_error_surfaces_anthropic_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/skills"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "type": "error",
            "error": { "type": "invalid_request_error", "message": "files[] required" }
        })))
        .mount(&server)
        .await;

    let skills = provider(&server).skills();
    let err = skills
        .upload_skill(UploadSkillOptions {
            files: vec![SkillFile {
                path: "x.txt".into(),
                data: UploadFileData::Text { text: "x".into() },
            }],
            display_title: None,
            provider_options: None,
        })
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("files[] required"));
}
