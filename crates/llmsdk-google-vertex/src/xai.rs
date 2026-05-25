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
use llmsdk_openai::internal::{Inner as OpenAiInner, OpenAiChatModel, UrlStrategy};
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    CallOptions, GenerateResult, LanguageModel, StreamResult, SupportedUrls,
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

    async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult, ProviderError> {
        self.delegate().await?.do_generate(options).await
    }

    async fn do_stream(&self, options: CallOptions) -> Result<StreamResult, ProviderError> {
        self.delegate().await?.do_stream(options).await
    }
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
}
