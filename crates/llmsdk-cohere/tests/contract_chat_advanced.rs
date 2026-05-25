//! Advanced contract tests: RAG documents, image inputs, response format.
// Rust guideline compliant 2026-05-25

use llmsdk_cohere::Cohere;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, FilePart, Message, ResponseFormat, TextPart, UserPart,
};
use llmsdk_provider::shared::FileData;
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Cohere {
    Cohere::builder()
        .api_key("k")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
}

fn ok_response() -> serde_json::Value {
    json!({
        "generation_id": "g",
        "message": {
            "role": "assistant",
            "content": [{ "type": "text", "text": "ok" }]
        },
        "finish_reason": "COMPLETE"
    })
}

#[tokio::test]
async fn non_image_file_part_routes_to_documents_array() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat"))
        .and(body_partial_json(json!({
            "documents": [
                { "data": { "text": "hello", "title": "notes.txt" } }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let _ = provider(&server)
        .chat("command-a-03-2025")
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![
                    UserPart::Text(TextPart {
                        text: "summarize".into(),
                        provider_options: None,
                    }),
                    UserPart::File(FilePart {
                        filename: Some("notes.txt".into()),
                        data: FileData::Text {
                            text: "hello".into(),
                        },
                        media_type: "text/plain".into(),
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

#[tokio::test]
async fn image_url_keeps_parts_array() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat"))
        .and(body_partial_json(json!({
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "look" },
                    {
                        "type": "image_url",
                        "image_url": { "url": "https://example.com/a.png" }
                    }
                ]
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let _ = provider(&server)
        .chat("command-a-vision-07-2025")
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![
                    UserPart::Text(TextPart {
                        text: "look".into(),
                        provider_options: None,
                    }),
                    UserPart::File(FilePart {
                        filename: None,
                        data: FileData::Url {
                            url: "https://example.com/a.png".into(),
                        },
                        media_type: "image/png".into(),
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

#[tokio::test]
async fn json_response_format_emits_json_object_wire() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat"))
        .and(body_partial_json(json!({
            "response_format": {
                "type": "json_object",
                "json_schema": { "type": "object" }
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()))
        .mount(&server)
        .await;

    let _ = provider(&server)
        .chat("command-a-03-2025")
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserPart::Text(TextPart {
                    text: "json".into(),
                    provider_options: None,
                })],
                provider_options: None,
            }],
            response_format: Some(ResponseFormat::Json {
                schema: Some(serde_json::from_value(json!({"type": "object"})).expect("schema")),
                name: None,
                description: None,
            }),
            ..Default::default()
        })
        .await
        .expect("ok");
}
