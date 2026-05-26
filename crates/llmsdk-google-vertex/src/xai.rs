//! xAI (Grok) on Vertex.
//!
//! Mirrors `xai/google-vertex-xai-provider.ts`. Grok partner models on
//! Vertex speak the OpenAI-compatible Chat Completions wire at
//! `{base}/endpoints/openapi/chat/completions`. We reuse
//! `llmsdk-openai`'s [`OpenAiChatModel`] via its `internal` module and
//! flip in a per-call OAuth header (or Express-mode `x-goog-api-key`).
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use llmsdk_openai::internal::{Inner as OpenAiInner, OpenAiChatModel, UrlStrategy};
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    CallOptions, GenerateResult, LanguageModel, StreamPart, StreamResult, SupportedUrls, Usage,
};

use crate::PROVIDER_ID_XAI;
use crate::auth::cloud_platform_token;
use crate::config::{VertexAuthMode, VertexInner};

const XAI_PROVIDER_ID: &str = PROVIDER_ID_XAI;

/// Vertex xAI sub-provider handle.
#[derive(Debug, Clone)]
pub struct GoogleVertexXai {
    inner: Arc<VertexInner>,
}

impl GoogleVertexXai {
    pub(crate) fn new(inner: Arc<VertexInner>) -> Self {
        Self { inner }
    }

    /// Construct a Grok-on-Vertex chat model handle.
    #[must_use]
    pub fn language_model(&self, model_id: impl Into<String>) -> GoogleVertexXaiChatModel {
        GoogleVertexXaiChatModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::language_model`].
    #[must_use]
    pub fn chat(&self, model_id: impl Into<String>) -> GoogleVertexXaiChatModel {
        self.language_model(model_id)
    }

    /// Alias of [`Self::language_model`] — matches upstream `chatModel(id)`.
    #[must_use]
    pub fn chat_model(&self, model_id: impl Into<String>) -> GoogleVertexXaiChatModel {
        self.language_model(model_id)
    }
}

/// Vertex xAI (Grok) chat-model handle.
#[derive(Debug, Clone)]
pub struct GoogleVertexXaiChatModel {
    inner: Arc<VertexInner>,
    model_id: String,
}

impl GoogleVertexXaiChatModel {
    pub(crate) fn new(inner: Arc<VertexInner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    async fn delegate(&self) -> Result<OpenAiChatModel, ProviderError> {
        let inner = build_openai_inner(&self.inner, XAI_PROVIDER_ID).await?;
        Ok(OpenAiChatModel::new(Arc::new(inner), self.model_id.clone()))
    }
}

#[async_trait]
impl LanguageModel for GoogleVertexXaiChatModel {
    fn provider(&self) -> &str {
        XAI_PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn supported_urls(&self) -> SupportedUrls {
        // Mirrors upstream's `{ 'image/*': [/^https?:\/\/.*$/] }`.
        [(
            "image/*".to_owned(),
            vec![llmsdk_provider::language_model::UrlPattern(
                r"^https?://.*$".into(),
            )],
        )]
        .into_iter()
        .collect()
    }

    async fn do_generate(&self, mut options: CallOptions) -> Result<GenerateResult, ProviderError> {
        strip_xai_unsupported_fields(&mut options);
        let mut result = self.delegate().await?.do_generate(options).await?;
        result.usage = reconvert_xai_usage(result.usage);
        Ok(result)
    }

    async fn do_stream(&self, mut options: CallOptions) -> Result<StreamResult, ProviderError> {
        strip_xai_unsupported_fields(&mut options);
        let result = self.delegate().await?.do_stream(options).await?;
        let stream = result.stream.map(|item| match item {
            Ok(StreamPart::Finish {
                usage,
                finish_reason,
                provider_metadata,
            }) => Ok(StreamPart::Finish {
                usage: reconvert_xai_usage(usage),
                finish_reason,
                provider_metadata,
            }),
            other => other,
        });
        Ok(StreamResult {
            stream: Box::pin(stream),
            request: result.request,
            response: result.response,
        })
    }
}

/// Mirrors upstream `transformGoogleVertexXaiRequestBody`
/// (`google-vertex/xai/google-vertex-xai-provider.ts:131-134`): Vertex's Grok
/// endpoint rejects `reasoning_effort`, so we strip both the canonical
/// `CallOptions::reasoning` field and the OpenAI-specific provider option
/// before the OpenAI core builds the wire request.
fn strip_xai_unsupported_fields(options: &mut CallOptions) {
    options.reasoning = None;
    if let Some(po) = options.provider_options.as_mut()
        && let Some(openai) = po.get_mut("openai")
    {
        openai.remove("reasoningEffort");
        openai.remove("reasoning_effort");
    }
}

/// Mirrors upstream `convertGoogleVertexXaiUsage`
/// (`google-vertex/xai/google-vertex-xai-provider.ts:89-129`): xAI on Vertex
/// returns `completion_tokens` and `completion_tokens_details.reasoning_tokens`
/// as *disjoint* counters, whereas mainline OpenAI treats `completion_tokens`
/// as already including reasoning. The OpenAI core converter uses the mainline
/// formula (`text = completion - reasoning`); we re-derive the totals from the
/// preserved raw usage object so callers see the per-modality counters the
/// Vertex Grok backend actually billed for.
fn reconvert_xai_usage(mut usage: Usage) -> Usage {
    let Some(raw) = usage.raw.as_ref() else {
        return usage;
    };
    let completion = raw.get("completion_tokens").and_then(|v| v.as_u64());
    let reasoning = raw
        .get("completion_tokens_details")
        .and_then(|v| v.get("reasoning_tokens"))
        .and_then(|v| v.as_u64());
    match (completion, reasoning) {
        (Some(c), Some(r)) => {
            usage.output_tokens.total = Some(c + r);
            usage.output_tokens.text = Some(c);
            usage.output_tokens.reasoning = Some(r);
        }
        (Some(c), None) => {
            usage.output_tokens.total = Some(c);
            usage.output_tokens.text = Some(c);
            usage.output_tokens.reasoning = None;
        }
        _ => {}
    }
    usage
}

pub(crate) async fn build_openai_inner(
    state: &VertexInner,
    provider_id: &'static str,
) -> Result<OpenAiInner, ProviderError> {
    let base_url = state.sub_provider_base();
    let strategy = UrlStrategy::Standard { base_url };
    let mut headers: HashMap<String, Option<String>> = state.extra_headers.clone();
    match &state.auth {
        VertexAuthMode::Express { api_key } => {
            headers.insert("x-goog-api-key".into(), Some(api_key.clone()));
            // Vertex Express mode does not use OpenAI bearer auth.
        }
        VertexAuthMode::OAuth { token_provider, .. } => {
            let token = cloud_platform_token(token_provider.as_ref()).await?;
            headers.insert("Authorization".into(), Some(format!("Bearer {token}")));
        }
    }
    Ok(OpenAiInner::new(
        strategy,
        headers,
        state.http.clone(),
        provider_id,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GoogleVertex;
    use llmsdk_provider::language_model::ReasoningEffort;
    use serde_json::json;

    #[tokio::test]
    async fn xai_handle_reports_vertex_provider() {
        let p = GoogleVertex::builder().api_key("k").build().await.unwrap();
        let m = p.xai().chat("xai/grok-4.20-reasoning");
        assert_eq!(m.provider(), PROVIDER_ID_XAI);
        assert_eq!(m.model_id(), "xai/grok-4.20-reasoning");
    }

    #[tokio::test]
    async fn xai_supported_urls_includes_https_image() {
        let p = GoogleVertex::builder().api_key("k").build().await.unwrap();
        let m = p.xai().chat("xai/grok-4.20-reasoning");
        let urls = m.supported_urls().await;
        let v = urls.get("image/*").expect("image/* key");
        assert!(v.iter().any(|p| p.0.contains("https?")));
    }

    #[test]
    fn strip_drops_reasoning_effort_from_both_sources() {
        let mut po = HashMap::new();
        let mut openai = serde_json::Map::new();
        openai.insert("reasoningEffort".into(), json!("high"));
        openai.insert("temperature".into(), json!(0.5));
        po.insert("openai".to_owned(), openai);
        let mut options = CallOptions {
            provider_options: Some(po),
            reasoning: Some(ReasoningEffort::High),
            ..CallOptions::default()
        };
        strip_xai_unsupported_fields(&mut options);
        assert!(options.reasoning.is_none());
        let openai = options
            .provider_options
            .as_ref()
            .and_then(|po| po.get("openai"))
            .expect("openai bag preserved");
        assert!(!openai.contains_key("reasoningEffort"));
        assert!(openai.contains_key("temperature"));
    }

    #[test]
    fn reconvert_xai_usage_splits_reasoning_from_completion() {
        // Mirrors upstream test fixture
        // `xai/google-vertex-xai-provider.test.ts:140-182`:
        // completion=24, reasoning=124 → total=148, text=24, reasoning=124.
        let mut raw = serde_json::Map::new();
        raw.insert("prompt_tokens".into(), json!(10));
        raw.insert("completion_tokens".into(), json!(24));
        raw.insert(
            "completion_tokens_details".into(),
            json!({"reasoning_tokens": 124}),
        );
        let usage = Usage {
            raw: Some(raw),
            ..Usage::default()
        };
        let out = reconvert_xai_usage(usage);
        assert_eq!(out.output_tokens.total, Some(148));
        assert_eq!(out.output_tokens.text, Some(24));
        assert_eq!(out.output_tokens.reasoning, Some(124));
    }

    #[test]
    fn reconvert_xai_usage_handles_missing_reasoning_tokens() {
        let mut raw = serde_json::Map::new();
        raw.insert("completion_tokens".into(), json!(40));
        let usage = Usage {
            raw: Some(raw),
            ..Usage::default()
        };
        let out = reconvert_xai_usage(usage);
        assert_eq!(out.output_tokens.total, Some(40));
        assert_eq!(out.output_tokens.text, Some(40));
        assert!(out.output_tokens.reasoning.is_none());
    }

    #[test]
    fn reconvert_xai_usage_noop_when_raw_absent() {
        let usage = Usage::default();
        let out = reconvert_xai_usage(usage.clone());
        assert_eq!(out, usage);
    }
}
