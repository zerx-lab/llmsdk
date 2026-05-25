//! Advanced contract tests: warnings on unsupported settings, max-token mapping,
//! prefix-duplicate dropping, file references.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    AssistantPart, CallOptions, Content, FilePart, Message, TextPart, UserPart,
};
use llmsdk_provider::shared::{FileData, Warning};
use llmsdk_xai::Xai;
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Xai {
    Xai::builder()
        .api_key("xai-test")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
}

fn user(text: &str) -> Message {
    Message::User {
        content: vec![UserPart::Text(TextPart {
            text: text.into(),
            provider_options: None,
        })],
        provider_options: None,
    }
}

fn ok_response() -> serde_json::Value {
    json!({
        "id": "r1",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop"
        }]
    })
}

#[tokio::test]
async fn unsupported_settings_emit_warnings() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let result = provider(&server)
        .chat("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            top_k: Some(5),
            frequency_penalty: Some(0.1),
            presence_penalty: Some(0.1),
            stop_sequences: Some(vec!["END".into()]),
            ..Default::default()
        })
        .await
        .expect("ok");

    let names: Vec<&str> = result
        .warnings
        .iter()
        .filter_map(|w| match w {
            Warning::UnsupportedSetting { setting, .. } => Some(setting.as_str()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"topK"));
    assert!(names.contains(&"frequencyPenalty"));
    assert!(names.contains(&"presencePenalty"));
    assert!(names.contains(&"stopSequences"));
}

#[tokio::test]
async fn max_output_tokens_maps_to_max_completion_tokens() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({ "max_completion_tokens": 256 })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let _ = provider(&server)
        .chat("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            max_output_tokens: Some(256),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn assistant_prefix_duplicate_text_is_dropped() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "r1",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "PREFIX " },
                "finish_reason": "stop"
            }]
        })))
        .mount(&server)
        .await;

    let result = provider(&server)
        .chat("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![
                user("hi"),
                Message::Assistant {
                    content: vec![AssistantPart::Text(TextPart {
                        text: "PREFIX ".into(),
                        provider_options: None,
                    })],
                    provider_options: None,
                },
            ],
            ..Default::default()
        })
        .await
        .expect("ok");

    // duplicate prefix dropped → empty content list
    assert!(result.content.is_empty() || !matches!(result.content[0], Content::Text(_)));
}

#[tokio::test]
async fn file_reference_serializes_as_file_id_part() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "messages": [
                { "role": "user", "content": [
                    { "type": "text", "text": "describe it" },
                    { "type": "file", "file": { "file_id": "file_abc123" } }
                ]}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let mut reference = serde_json::Map::new();
    reference.insert("xai".into(), json!("file_abc123"));
    let _ = provider(&server)
        .chat("grok-4.3")
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![
                    UserPart::Text(TextPart {
                        text: "describe it".into(),
                        provider_options: None,
                    }),
                    UserPart::File(FilePart {
                        filename: None,
                        data: FileData::Reference { reference },
                        media_type: "application/pdf".into(),
                        provider_options: None,
                    }),
                ],
                provider_options: None,
            }],
            ..Default::default()
        })
        .await
        .expect("ok");
}
