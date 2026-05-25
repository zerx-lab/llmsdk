//! Contract tests for Gemini streamGenerateContent (SSE).
// Rust guideline compliant 2026-05-25

use futures::StreamExt;
use llmsdk_google::Google;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, FinishReasonKind, Message, StreamPart, TextPart, UserPart,
};
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Google {
    Google::builder()
        .api_key("test-key")
        .base_url(server.uri())
        .build()
        .expect("builds")
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

fn sse(chunks: &[&str]) -> String {
    let mut out = String::new();
    for chunk in chunks {
        out.push_str("data: ");
        out.push_str(chunk);
        out.push_str("\n\n");
    }
    out
}

#[tokio::test]
async fn happy_path_text_stream() {
    let server = MockServer::start().await;
    let body = sse(&[
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"hel"}]}}]}"#,
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]}}]}"#,
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":2}}"#,
    ]);
    Mock::given(method("POST"))
        .and(path_regex(
            r"^/models/gemini-2\.5-flash:streamGenerateContent$",
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let model = provider(&server).chat("gemini-2.5-flash");
    let result = model
        .do_stream(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    let parts: Vec<_> = result
        .stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|r| r.expect("part ok"))
        .collect();

    // First should be StreamStart, last should be Finish.
    assert!(matches!(
        parts.first(),
        Some(StreamPart::StreamStart { .. })
    ));
    let last = parts.last().expect("non-empty");
    assert!(matches!(
        last,
        StreamPart::Finish {
            finish_reason,
            ..
        } if finish_reason.unified == FinishReasonKind::Stop
    ));

    let mut text = String::new();
    for p in &parts {
        if let StreamPart::TextDelta { delta, .. } = p {
            text.push_str(delta);
        }
    }
    assert_eq!(text, "hello");
}

#[tokio::test]
async fn reasoning_block_split() {
    let server = MockServer::start().await;
    let body = sse(&[
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"thinking","thought":true}]}}]}"#,
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"answer"}]}}]}"#,
        r#"{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}]}"#,
    ]);
    Mock::given(method("POST"))
        .and(path_regex(
            r"^/models/gemini-2\.5-flash:streamGenerateContent$",
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let model = provider(&server).chat("gemini-2.5-flash");
    let result = model
        .do_stream(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    let parts: Vec<_> = result
        .stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|r| r.expect("part ok"))
        .collect();

    let mut saw_reasoning = false;
    let mut saw_reasoning_end = false;
    for p in &parts {
        match p {
            StreamPart::ReasoningStart { .. } => saw_reasoning = true,
            StreamPart::ReasoningEnd { .. } => saw_reasoning_end = true,
            _ => {}
        }
    }
    assert!(saw_reasoning && saw_reasoning_end);
}
