//! Contract tests for the dynamic Microsoft Entra ID `token_provider`.
//!
//! Mirrors the upstream `azure-openai-provider.test.ts` cases
//! "should call tokenProvider for every request" and the mutual-exclusion
//! guard against combining `apiKey` with `tokenProvider`.
// Rust guideline compliant 2026-06-04

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use llmsdk_azure::AzureOpenAi;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{CallOptions, Message, TextPart, UserPart};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn user_prompt() -> Vec<Message> {
    vec![Message::User {
        content: vec![UserPart::Text(TextPart {
            text: "ping".into(),
            provider_options: None,
        })],
        provider_options: None,
    }]
}

fn ok_body() -> serde_json::Value {
    json!({
        "id": "chatcmpl-az-1",
        "created": 1_700_000_000_u64,
        "model": "gpt-4o-mini-deployment",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "pong" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3 }
    })
}

#[tokio::test]
async fn token_provider_is_called_for_every_request() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_body()))
        .mount(&server)
        .await;

    // Each call yields a fresh token: token-1, token-2, ...
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_for_provider = Arc::clone(&counter);

    let provider = AzureOpenAi::builder()
        .base_url(server.uri())
        .token_provider(move || {
            let counter = Arc::clone(&counter_for_provider);
            async move {
                let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
                Ok(format!("token-{n}"))
            }
        })
        .build()
        .expect("provider builds");

    let model = provider.chat("gpt-4o-mini-deployment");

    model
        .do_generate(CallOptions {
            prompt: user_prompt(),
            ..Default::default()
        })
        .await
        .expect("first call succeeds");

    model
        .do_generate(CallOptions {
            prompt: user_prompt(),
            ..Default::default()
        })
        .await
        .expect("second call succeeds");

    // Provider invoked exactly once per request.
    assert_eq!(counter.load(Ordering::SeqCst), 2);

    // Each request carried the freshly minted bearer token, and no `api-key`
    // header was sent.
    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(requests.len(), 2);
    let auth0 = requests[0]
        .headers
        .get("authorization")
        .expect("authorization header present on request 0");
    assert_eq!(auth0.to_str().unwrap(), "Bearer token-1");
    let auth1 = requests[1]
        .headers
        .get("authorization")
        .expect("authorization header present on request 1");
    assert_eq!(auth1.to_str().unwrap(), "Bearer token-2");
    assert!(
        requests[0].headers.get("api-key").is_none(),
        "no api-key header expected when using token_provider"
    );
}

#[tokio::test]
async fn api_key_and_token_provider_are_mutually_exclusive() {
    let err = AzureOpenAi::builder()
        .resource_name("my-resource")
        .api_key("az-key")
        .token_provider(|| async { Ok("aad-token".to_owned()) })
        .build()
        .expect_err("combining api_key with token_provider must fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("only one authentication method"),
        "unexpected error message: {msg}"
    );
}
