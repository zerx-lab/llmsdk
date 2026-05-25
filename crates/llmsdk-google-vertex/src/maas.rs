//! MaaS (Model-as-a-Service) on Vertex.
//!
//! Mirrors `maas/google-vertex-maas-provider.ts`. Partner / open models
//! published to Vertex (DeepSeek, Llama, Mistral, Qwen, Kimi, ...) speak
//! the OpenAI-compatible Chat Completions wire at
//! `{base}/endpoints/openapi/chat/completions`. We reuse
//! `llmsdk-openai`'s [`OpenAiChatModel`] via its `internal` module.
// Rust guideline compliant 2026-05-25

use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_openai::internal::OpenAiChatModel;
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    CallOptions, GenerateResult, LanguageModel, StreamResult, SupportedUrls,
};

use crate::PROVIDER_ID_MAAS;
use crate::config::VertexInner;
use crate::xai::build_openai_inner;

/// Vertex MaaS sub-provider handle.
#[derive(Debug, Clone)]
pub struct GoogleVertexMaas {
    inner: Arc<VertexInner>,
}

impl GoogleVertexMaas {
    pub(crate) fn new(inner: Arc<VertexInner>) -> Self {
        Self { inner }
    }

    /// Construct a MaaS chat model handle.
    ///
    /// `model_id` follows the `publisher/model-id` convention upstream
    /// uses for the `MaaSModelId` type, e.g.
    /// `"deepseek-ai/deepseek-v3.2-maas"` or
    /// `"meta/llama-4-scout-17b-16e-instruct-maas"`.
    #[must_use]
    pub fn language_model(&self, model_id: impl Into<String>) -> GoogleVertexMaasChatModel {
        GoogleVertexMaasChatModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::language_model`].
    #[must_use]
    pub fn chat(&self, model_id: impl Into<String>) -> GoogleVertexMaasChatModel {
        self.language_model(model_id)
    }
}

/// Vertex MaaS chat-model handle.
#[derive(Debug, Clone)]
pub struct GoogleVertexMaasChatModel {
    inner: Arc<VertexInner>,
    model_id: String,
}

impl GoogleVertexMaasChatModel {
    pub(crate) fn new(inner: Arc<VertexInner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    async fn delegate(&self) -> Result<OpenAiChatModel, ProviderError> {
        let inner = build_openai_inner(&self.inner, PROVIDER_ID_MAAS).await?;
        Ok(OpenAiChatModel::new(Arc::new(inner), self.model_id.clone()))
    }
}

#[async_trait]
impl LanguageModel for GoogleVertexMaasChatModel {
    fn provider(&self) -> &str {
        PROVIDER_ID_MAAS
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn supported_urls(&self) -> SupportedUrls {
        SupportedUrls::default()
    }

    async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult, ProviderError> {
        self.delegate().await?.do_generate(options).await
    }

    async fn do_stream(&self, options: CallOptions) -> Result<StreamResult, ProviderError> {
        self.delegate().await?.do_stream(options).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GoogleVertex;

    #[tokio::test]
    async fn maas_handle_reports_vertex_provider() {
        let p = GoogleVertex::builder().api_key("k").build().await.unwrap();
        let m = p.maas().chat("deepseek-ai/deepseek-v3.2-maas");
        assert_eq!(m.provider(), PROVIDER_ID_MAAS);
        assert_eq!(m.model_id(), "deepseek-ai/deepseek-v3.2-maas");
    }
}
