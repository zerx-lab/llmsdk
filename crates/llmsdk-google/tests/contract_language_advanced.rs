//! Advanced response shape tests (sources, response format, file outputs).
// Rust guideline compliant 2026-05-25

use llmsdk_google::Google;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Content, Message, ResponseFormat, Source, TextPart, UserPart,
};
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Google {
    Google::builder()
        .api_key("k")
        .base_url(server.uri())
        .build()
        .expect("ok")
}
fn user_text(text: &str) -> Message {
    Message::User {
        content: vec![UserPart::Text(TextPart {
            text: text.into(),
            provider_options: None,
        })],
        provider_options: None,
    }
}

#[tokio::test]
async fn grounding_chunks_emit_url_sources() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates":[{
                "content":{"role":"model","parts":[{"text":"hi"}]},
                "finishReason":"STOP",
                "groundingMetadata": {
                    "groundingChunks": [
                        {"web": {"uri":"https://example.com/a","title":"A"}},
                        {"web": {"uri":"https://example.com/b","title":"B"}}
                    ]
                }
            }]
        })))
        .mount(&server)
        .await;
    let model = provider(&server).chat("gemini-2.5-flash");
    let r = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");
    let sources: Vec<&Source> = r
        .content
        .iter()
        .filter_map(|c| {
            if let Content::Source(s) = c {
                Some(s)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(sources.len(), 2);
    assert!(matches!(sources[0], Source::Url { url, .. } if url.contains("example.com/a")));
}

#[tokio::test]
async fn response_format_json_sets_mime_type() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:generateContent"))
        .and(body_partial_json(json!({
            "generationConfig": { "responseMimeType": "application/json" }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates":[{"content":{"role":"model","parts":[{"text":"{}"}]},"finishReason":"STOP"}]
        })))
        .mount(&server)
        .await;
    let model = provider(&server).chat("gemini-2.5-flash");
    let _ = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            response_format: Some(ResponseFormat::Json {
                schema: None,
                name: None,
                description: None,
            }),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn inline_data_to_file_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash-image:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates":[{
                "content":{"role":"model","parts":[
                    {"inlineData":{"mimeType":"image/png","data":"AAA="}}
                ]},
                "finishReason":"STOP"
            }]
        })))
        .mount(&server)
        .await;
    let model = provider(&server).chat("gemini-2.5-flash-image");
    let r = model
        .do_generate(CallOptions {
            prompt: vec![user_text("draw")],
            ..Default::default()
        })
        .await
        .expect("ok");
    let file_count = r
        .content
        .iter()
        .filter(|c| matches!(c, Content::File(_)))
        .count();
    assert_eq!(file_count, 1);
}
