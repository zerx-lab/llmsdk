//! Gemini language model on Vertex.
//!
//! Mirrors the chat-model branch of `google-vertex-provider-base.ts` →
//! `createChatModel`. Internally constructs a [`GoogleLanguageModel`]
//! handle (re-exported from `llmsdk-google::internal`) per call so that
//! OAuth tokens can be minted just-in-time without mutating shared state.
// Rust guideline compliant 2026-05-25

use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_google::internal::{GoogleLanguageModel, Inner as GoogleInner};
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    CallOptions, GenerateResult, LanguageModel, StreamResult, SupportedUrls, UrlPattern,
};
use llmsdk_provider_utils::http::HttpClient;

use crate::PROVIDER_ID_CHAT;
use crate::auth::cloud_platform_token;
use crate::config::{VertexAuthMode, VertexInner};

/// Vertex Gemini language-model handle.
///
/// Cheap to clone; the shared provider state is held in an [`Arc`].
#[derive(Debug, Clone)]
pub struct GoogleVertexLanguageModel {
    pub(crate) inner: Arc<VertexInner>,
    pub(crate) model_id: String,
}

impl GoogleVertexLanguageModel {
    pub(crate) fn new(inner: Arc<VertexInner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    /// Mint a per-call [`GoogleLanguageModel`] with the current auth /
    /// base URL state.
    async fn delegate(&self) -> Result<GoogleLanguageModel, ProviderError> {
        let inner = build_gemini_inner(
            &self.inner,
            self.inner.publishers_google_base(),
            PROVIDER_ID_CHAT,
            self.inner.http.clone(),
        )
        .await?;
        Ok(GoogleLanguageModel::new(
            Arc::new(inner),
            self.model_id.clone(),
        ))
    }
}

#[async_trait]
impl LanguageModel for GoogleVertexLanguageModel {
    fn provider(&self) -> &str {
        PROVIDER_ID_CHAT
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn supported_urls(&self) -> SupportedUrls {
        [(
            "*".to_owned(),
            vec![
                UrlPattern(r"^https?://.*$".into()),
                UrlPattern(r"^gs://.*$".into()),
            ],
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

/// Build a fully populated `llmsdk_google::internal::Inner` carrying the
/// caller-supplied base URL + per-call auth headers.
pub(crate) async fn build_gemini_inner(
    state: &VertexInner,
    base_url: String,
    provider: &str,
    http: HttpClient,
) -> Result<GoogleInner, ProviderError> {
    let mut builder = GoogleInner::builder()
        .provider(provider)
        .base_url(base_url)
        .http_client(http);
    for (k, v) in &state.extra_headers {
        builder = builder.header(k.clone(), v.clone());
    }
    match &state.auth {
        VertexAuthMode::Express { api_key } => {
            builder = builder.header("x-goog-api-key", Some(api_key.clone()));
        }
        VertexAuthMode::OAuth { token_provider, .. } => {
            let token = cloud_platform_token(token_provider.as_ref()).await?;
            builder = builder.header("Authorization", Some(format!("Bearer {token}")));
        }
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GoogleVertex;

    #[tokio::test]
    async fn provider_string_is_vertex_chat() {
        let p = GoogleVertex::builder().api_key("k").build().await.unwrap();
        let m = p.language_model("gemini-2.5-flash");
        assert_eq!(m.provider(), PROVIDER_ID_CHAT);
        assert_eq!(m.model_id(), "gemini-2.5-flash");
    }

    #[tokio::test]
    async fn supported_urls_includes_gs_and_http() {
        let p = GoogleVertex::builder().api_key("k").build().await.unwrap();
        let m = p.language_model("gemini-2.5-flash");
        let urls = m.supported_urls().await;
        let v = urls.get("*").expect("default key");
        assert!(v.iter().any(|p| p.0.contains("gs:")));
        assert!(v.iter().any(|p| p.0.contains("https?")));
    }
}
