//! Provider configuration and entry point.
//!
//! Mirrors `@ai-sdk/cohere/src/cohere-provider.ts`.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;

use llmsdk_provider::ProviderError;
use llmsdk_provider_utils::http::HttpClient;

use crate::chat::CohereChatModel;
use crate::embedding::CohereEmbeddingModel;
use crate::reranking::CohereRerankingModel;
use crate::{API_KEY_ENV_VAR, DEFAULT_BASE_URL};

/// Cohere provider handle — entry point for model construction.
///
/// Cheap to clone; the underlying HTTP client and headers are shared.
#[derive(Debug, Clone)]
pub struct Cohere {
    inner: Arc<Inner>,
}

#[derive(Debug)]
pub(crate) struct Inner {
    pub(crate) base_url: String,
    pub(crate) headers: HashMap<String, Option<String>>,
    pub(crate) http: HttpClient,
}

impl Cohere {
    /// Open a [`CohereBuilder`].
    #[must_use]
    pub fn builder() -> CohereBuilder {
        CohereBuilder::default()
    }

    /// Build with defaults: API key from `COHERE_API_KEY`, default base URL.
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError::load_api_key`] when the env var is unset
    /// or empty.
    pub fn from_env() -> Result<Self, ProviderError> {
        Self::builder().build()
    }

    /// Construct a Chat model handle.
    ///
    /// `model_id` is the Cohere chat model name, e.g. `"command-a-03-2025"`.
    #[must_use]
    pub fn chat(&self, model_id: impl Into<String>) -> CohereChatModel {
        CohereChatModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::chat`] — mirrors ai-sdk's `provider.languageModel(id)`.
    #[must_use]
    pub fn language_model(&self, model_id: impl Into<String>) -> CohereChatModel {
        self.chat(model_id)
    }

    /// Construct an Embeddings model handle.
    ///
    /// `model_id` is the Cohere embedding model name, e.g. `"embed-english-v3.0"`.
    #[must_use]
    pub fn embedding(&self, model_id: impl Into<String>) -> CohereEmbeddingModel {
        CohereEmbeddingModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::embedding`].
    #[must_use]
    pub fn embedding_model(&self, model_id: impl Into<String>) -> CohereEmbeddingModel {
        self.embedding(model_id)
    }

    /// Alias of [`Self::embedding`] — mirrors ai-sdk's legacy `textEmbedding`.
    #[must_use]
    pub fn text_embedding(&self, model_id: impl Into<String>) -> CohereEmbeddingModel {
        self.embedding(model_id)
    }

    /// Alias of [`Self::embedding`] — mirrors ai-sdk's legacy `textEmbeddingModel`.
    #[must_use]
    pub fn text_embedding_model(&self, model_id: impl Into<String>) -> CohereEmbeddingModel {
        self.embedding(model_id)
    }

    /// Construct a Reranking model handle.
    ///
    /// `model_id` is the Cohere reranker, e.g. `"rerank-v3.5"`.
    #[must_use]
    pub fn reranking(&self, model_id: impl Into<String>) -> CohereRerankingModel {
        CohereRerankingModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::reranking`].
    #[must_use]
    pub fn reranking_model(&self, model_id: impl Into<String>) -> CohereRerankingModel {
        self.reranking(model_id)
    }
}

/// Builder for [`Cohere`].
///
/// All setters are optional; `build()` falls back to env / library defaults.
#[derive(Debug, Default, Clone)]
pub struct CohereBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    extra_headers: HashMap<String, Option<String>>,
    http: Option<HttpClient>,
}

impl CohereBuilder {
    /// Set the API key explicitly.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Override the base URL (e.g. for a local proxy).
    ///
    /// Trailing slashes are stripped to match `withoutTrailingSlash` in ai-sdk.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        let mut s = url.into();
        while s.ends_with('/') {
            s.pop();
        }
        self.base_url = Some(s);
        self
    }

    /// Append or override a header.
    ///
    /// Passing `None` for `value` removes the header.
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: Option<String>) -> Self {
        self.extra_headers.insert(name.into(), value);
        self
    }

    /// Inject a pre-configured HTTP client.
    #[must_use]
    pub fn http_client(mut self, client: HttpClient) -> Self {
        self.http = Some(client);
        self
    }

    /// Finalize the provider.
    ///
    /// # Errors
    ///
    /// - [`ProviderError::load_api_key`] when no explicit key is given and
    ///   `COHERE_API_KEY` is unset / empty.
    /// - [`ProviderError`] from [`HttpClient::new`] if the TLS stack fails
    ///   to initialize (rare).
    pub fn build(self) -> Result<Cohere, ProviderError> {
        let api_key = llmsdk_provider_utils::api_key::load_api_key(
            &llmsdk_provider_utils::api_key::LoadApiKey {
                api_key: self.api_key.as_deref(),
                env_var: API_KEY_ENV_VAR,
                description: "Cohere",
                parameter_name: Some("api_key"),
            },
        )?;

        let base_url = self.base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());

        let mut headers = self.extra_headers;
        headers.insert("authorization".into(), Some(format!("Bearer {api_key}")));

        let http = match self.http {
            Some(client) => client,
            None => HttpClient::new()?,
        };

        Ok(Cohere {
            inner: Arc::new(Inner {
                base_url,
                headers,
                http,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_with_explicit_key_succeeds() {
        let cohere = Cohere::builder()
            .api_key("cohere-test-key")
            .build()
            .expect("ok");
        assert_eq!(cohere.inner.base_url, DEFAULT_BASE_URL);
        assert!(
            cohere
                .inner
                .headers
                .get("authorization")
                .unwrap()
                .as_ref()
                .unwrap()
                .starts_with("Bearer ")
        );
    }

    #[test]
    fn builder_strips_trailing_slash() {
        let cohere = Cohere::builder()
            .api_key("k")
            .base_url("https://proxy.example.com/v2/")
            .build()
            .expect("ok");
        assert_eq!(cohere.inner.base_url, "https://proxy.example.com/v2");
    }

    #[test]
    fn builder_custom_base_url_no_trailing_slash() {
        let cohere = Cohere::builder()
            .api_key("k")
            .base_url("https://proxy.example.com/v2")
            .build()
            .expect("ok");
        assert_eq!(cohere.inner.base_url, "https://proxy.example.com/v2");
    }
}
