//! Provider-options namespace dispatch for Azure.
//!
//! Mirrors `@ai-sdk/openai`'s
//! `openai-responses-language-model.ts:180-181` switch:
//! `provider.includes('azure') ? 'azure' : 'openai'`. Responses and Completion
//! endpoints must read from `provider_options["azure"]` and write
//! `provider_metadata["azure"]`; Chat / Embedding remain on `"openai"` since
//! upstream also hardcodes that namespace there.
// Rust guideline compliant 2026-02-21

use llmsdk_azure::AzureOpenAi;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{CallOptions, FinishReasonKind, Message, TextPart, UserPart};
use llmsdk_provider::shared::ProviderOptions;
use serde_json::{Map, json};
use wiremock::matchers::{body_partial_json, header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn user(text: &str) -> Message {
    Message::User {
        content: vec![UserPart::Text(TextPart {
            text: text.into(),
            provider_options: None,
        })],
        provider_options: None,
    }
}

fn po_with(namespace: &str, body: &serde_json::Value) -> ProviderOptions {
    let mut map = ProviderOptions::new();
    let obj = body
        .as_object()
        .cloned()
        .expect("test fixture must pass an object body");
    map.insert(namespace.into(), obj);
    map
}

#[tokio::test]
async fn responses_reads_azure_namespace_provider_options() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(query_param("api-version", "v1"))
        .and(header("api-key", "az-key"))
        // `reasoning.effort` only appears when `provider_options["azure"]
        // .reasoningEffort` is read; if Azure namespace dispatch is broken,
        // the field is silently dropped and the body matcher fails.
        .and(body_partial_json(json!({
            "model": "o3-mini",
            "reasoning": { "effort": "high" }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp-az-1",
            "model": "o3-mini",
            "created_at": 1_700_000_000_u64,
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": "ok", "annotations": [] }]
                }
            ],
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let provider = AzureOpenAi::builder()
        .api_key("az-key")
        .base_url(server.uri())
        .build()
        .expect("provider builds");

    let model = provider.responses("o3-mini");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            provider_options: Some(po_with("azure", &json!({ "reasoningEffort": "high" }))),
            ..Default::default()
        })
        .await
        .expect("responses call succeeds");

    assert_eq!(model.provider(), "azure.responses");
    assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
}

#[tokio::test]
async fn responses_falls_back_to_openai_namespace_on_azure() {
    // Upstream `@ai-sdk/openai` falls back to the `"openai"` namespace
    // when the Azure-flavoured one is empty (see
    // `openai-responses-language-model.ts:189-195`). Rust mirrors this
    // — passing `provider_options.openai.*` to an Azure Responses model
    // must still take effect.
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(body_partial_json(json!({
            "model": "o3-mini",
            "reasoning": { "effort": "high" }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp-az-2",
            "model": "o3-mini",
            "created_at": 1_700_000_000_u64,
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": "ok", "annotations": [] }]
                }
            ],
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let provider = AzureOpenAi::builder()
        .api_key("az-key")
        .base_url(server.uri())
        .build()
        .expect("provider builds");

    let result = provider
        .responses("o3-mini")
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            provider_options: Some(po_with("openai", &json!({ "reasoningEffort": "high" }))),
            ..Default::default()
        })
        .await
        .expect("fallback call succeeds");

    assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
}

#[tokio::test]
async fn chat_reads_openai_namespace_on_azure() {
    // Chat endpoint must stay on the "openai" namespace, matching upstream
    // `openai-chat-language-model.ts:396` which hardcodes
    // `{ openai: {} }` regardless of Azure routing.
    let server = MockServer::start().await;

    let mut po = Map::new();
    po.insert("user".into(), json!("alice"));
    let _ = po;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({ "user": "alice" })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-az-ns",
            "created": 1_700_000_000_u64,
            "model": "gpt-4o-mini-deployment",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "ok" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
        })))
        .mount(&server)
        .await;

    let provider = AzureOpenAi::builder()
        .api_key("az-key")
        .base_url(server.uri())
        .build()
        .expect("provider builds");

    let result = provider
        .chat("gpt-4o-mini-deployment")
        .do_generate(CallOptions {
            prompt: vec![user("hi")],
            provider_options: Some(po_with("openai", &json!({ "user": "alice" }))),
            ..Default::default()
        })
        .await
        .expect("call succeeds");

    assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
}
