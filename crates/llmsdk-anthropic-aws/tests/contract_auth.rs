//! Contract tests for authentication header injection.
//!
//! Mirrors the JS suite `anthropic-aws-provider.test.ts` (auth +
//! `workspaceId` + auth-precedence). The internal Anthropic pipeline is
//! exercised through `wiremock` — every test issues a real `do_generate`
//! call against a mock server and asserts that the right auth headers
//! landed on the wire.

use std::sync::Arc;

use llmsdk_anthropic_aws::AnthropicAws;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{CallOptions, Message, TextPart, UserPart};
use llmsdk_provider_utils::aws_sigv4::{
    AwsCredentials, AwsCredentialsProvider, StaticCredentialsProvider,
};
use serde_json::json;
use wiremock::matchers::{header, header_exists, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(Debug)]
struct CountingProvider {
    inner: StaticCredentialsProvider,
    count: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl AwsCredentialsProvider for CountingProvider {
    async fn get_credentials(&self) -> Result<AwsCredentials, llmsdk_provider::ProviderError> {
        self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.get_credentials().await
    }
}

fn test_prompt() -> Vec<Message> {
    vec![Message::User {
        content: vec![UserPart::Text(TextPart {
            text: "Hello".into(),
            provider_options: None,
        })],
        provider_options: None,
    }]
}

fn success_body() -> serde_json::Value {
    json!({
        "type": "message",
        "id": "msg_123",
        "model": "claude-sonnet-4-6",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi"}],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1}
    })
}

#[tokio::test]
async fn api_key_path_sends_xapikey_header_and_workspace_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", "sk-aws-platform-key"))
        .and(header("anthropic-workspace-id", "wrkspc_test"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_body()))
        .mount(&server)
        .await;

    let provider = AnthropicAws::builder()
        .region("us-west-2")
        .workspace_id("wrkspc_test")
        .api_key("sk-aws-platform-key")
        .base_url(server.uri())
        .build()
        .unwrap();

    let result = provider
        .language_model("claude-sonnet-4-6")
        .do_generate(CallOptions {
            prompt: test_prompt(),
            max_output_tokens: Some(64),
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(!result.content.is_empty());
}

#[tokio::test]
async fn sigv4_path_signs_with_authorization_and_x_amz_date() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header_exists("authorization"))
        .and(header_exists("x-amz-date"))
        .and(header("anthropic-workspace-id", "wrkspc_test"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_body()))
        .mount(&server)
        .await;

    let provider = AnthropicAws::builder()
        .region("us-west-2")
        .workspace_id("wrkspc_test")
        .access_key_id("AKIDEXAMPLE")
        .secret_access_key("wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")
        .base_url(server.uri())
        .build()
        .unwrap();

    provider
        .language_model("claude-sonnet-4-6")
        .do_generate(CallOptions {
            prompt: test_prompt(),
            max_output_tokens: Some(64),
            ..Default::default()
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn sigv4_path_with_session_token_propagates_x_amz_security_token() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header_exists("authorization"))
        .and(header("x-amz-security-token", "sess-tok"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_body()))
        .mount(&server)
        .await;

    let creds_provider: Arc<dyn AwsCredentialsProvider> = Arc::new(StaticCredentialsProvider::new(
        AwsCredentials::new("AKID", "SECRET").with_session_token("sess-tok"),
    ));

    let provider = AnthropicAws::builder()
        .region("us-east-1")
        .workspace_id("wrkspc_test")
        .credentials_provider(creds_provider)
        .base_url(server.uri())
        .build()
        .unwrap();

    provider
        .language_model("claude-sonnet-4-6")
        .do_generate(CallOptions {
            prompt: test_prompt(),
            max_output_tokens: Some(64),
            ..Default::default()
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn api_key_takes_precedence_over_sigv4_credentials() {
    // Both apiKey and AWS creds provided → x-api-key path runs, no SigV4.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", "sk-explicit"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_body()))
        .mount(&server)
        .await;

    let provider = AnthropicAws::builder()
        .region("us-west-2")
        .workspace_id("wrkspc_test")
        .api_key("sk-explicit")
        .access_key_id("should-be-ignored")
        .secret_access_key("should-be-ignored")
        .base_url(server.uri())
        .build()
        .unwrap();

    provider
        .language_model("claude-sonnet-4-6")
        .do_generate(CallOptions {
            prompt: test_prompt(),
            max_output_tokens: Some(64),
            ..Default::default()
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn custom_credentials_provider_is_called_on_each_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header_exists("authorization"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_body()))
        .expect(2)
        .mount(&server)
        .await;

    // Static provider returns deterministic creds; we use an atomic counter
    // wrapper to confirm it was invoked per request.
    let counter = Arc::new(CountingProvider {
        inner: StaticCredentialsProvider::new(AwsCredentials::new("AKID", "SECRET")),
        count: std::sync::atomic::AtomicUsize::new(0),
    });
    let counter_arc: Arc<dyn AwsCredentialsProvider> = Arc::clone(&counter) as _;

    let provider = AnthropicAws::builder()
        .region("us-west-2")
        .workspace_id("wrkspc_test")
        .credentials_provider(counter_arc)
        .base_url(server.uri())
        .build()
        .unwrap();

    for _ in 0..2 {
        provider
            .language_model("claude-sonnet-4-6")
            .do_generate(CallOptions {
                prompt: test_prompt(),
                max_output_tokens: Some(64),
                ..Default::default()
            })
            .await
            .unwrap();
    }

    assert_eq!(
        counter.count.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "credentials provider should be invoked once per request"
    );
}
