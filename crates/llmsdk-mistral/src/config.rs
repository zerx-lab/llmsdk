//! Provider configuration and entry point.
//!
//! Mirrors `@ai-sdk/mistral/src/mistral-provider.ts`.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;

use llmsdk_provider::ProviderError;
use llmsdk_provider_utils::http::HttpClient;

use crate::chat::MistralChatModel;
use crate::embedding::MistralEmbeddingModel;
use crate::{API_KEY_ENV_VAR, DEFAULT_BASE_URL};

/// Mistral provider handle — entry point for model construction.
///
/// Cheap to clone; the underlying HTTP client and headers are shared.
#[derive(Debug, Clone)]
pub struct Mistral {
    inner: Arc<Inner>,
}

/// User-supplied id generator for streaming reasoning blocks.
///
/// Mirrors `MistralProviderSettings.generateId` in upstream
/// `mistral-provider.ts:77`. When set, the chat model invokes the callback
/// each time it needs an identifier for a new reasoning block; otherwise it
/// falls back to a deterministic in-process counter.
pub type GenerateIdFn = dyn Fn() -> String + Send + Sync;

pub(crate) struct Inner {
    pub(crate) base_url: String,
    pub(crate) headers: HashMap<String, Option<String>>,
    pub(crate) http: HttpClient,
    pub(crate) generate_id: Option<Arc<GenerateIdFn>>,
}

impl std::fmt::Debug for Inner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("base_url", &self.base_url)
            .field("headers", &self.headers)
            .field("http", &self.http)
            .field("generate_id", &self.generate_id.is_some())
            .finish()
    }
}

impl Mistral {
    /// Open a [`MistralBuilder`].
    #[must_use]
    pub fn builder() -> MistralBuilder {
        MistralBuilder::default()
    }

    /// Build with defaults: API key from `MISTRAL_API_KEY`, default base URL.
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError::load_api_key`] when the env var is unset
    /// or empty.
    pub fn from_env() -> Result<Self, ProviderError> {
        Self::builder().build()
    }

    /// Construct a Chat Completions model handle.
    ///
    /// `model_id` is the Mistral model name, e.g. `"mistral-large-latest"`.
    #[must_use]
    pub fn chat(&self, model_id: impl Into<String>) -> MistralChatModel {
        MistralChatModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::chat`] — mirrors ai-sdk's `provider.languageModel(id)`.
    #[must_use]
    pub fn language_model(&self, model_id: impl Into<String>) -> MistralChatModel {
        self.chat(model_id)
    }

    /// Construct a text-embedding model handle.
    ///
    /// `model_id` is the Mistral embedding model name, e.g.
    /// `"mistral-embed"` or `"codestral-embed"`.
    #[must_use]
    pub fn embedding(&self, model_id: impl Into<String>) -> MistralEmbeddingModel {
        MistralEmbeddingModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::embedding`] — mirrors ai-sdk's `provider.embeddingModel(id)`.
    #[must_use]
    pub fn embedding_model(&self, model_id: impl Into<String>) -> MistralEmbeddingModel {
        self.embedding(model_id)
    }

    /// Deprecated alias of [`Self::embedding`] retained for ai-sdk parity.
    #[must_use]
    pub fn text_embedding(&self, model_id: impl Into<String>) -> MistralEmbeddingModel {
        self.embedding(model_id)
    }

    /// Deprecated alias of [`Self::embedding_model`] retained for ai-sdk parity.
    #[must_use]
    pub fn text_embedding_model(&self, model_id: impl Into<String>) -> MistralEmbeddingModel {
        self.embedding(model_id)
    }
}

/// Builder for [`Mistral`].
///
/// All setters are optional; `build()` falls back to env / library defaults.
#[derive(Default, Clone)]
pub struct MistralBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    extra_headers: HashMap<String, Option<String>>,
    http: Option<HttpClient>,
    generate_id: Option<Arc<GenerateIdFn>>,
}

impl std::fmt::Debug for MistralBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MistralBuilder")
            .field("api_key", &self.api_key.as_ref().map(|_| "***"))
            .field("base_url", &self.base_url)
            .field("extra_headers", &self.extra_headers)
            .field("http", &self.http.is_some())
            .field("generate_id", &self.generate_id.is_some())
            .finish()
    }
}

impl MistralBuilder {
    /// Set the API key explicitly.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Override the base URL (e.g. for a local proxy).
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
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

    /// Override the id generator used for streaming reasoning blocks.
    ///
    /// Mirrors `config.generateId` on the upstream `MistralChatLanguageModel`
    /// (`mistral-provider.ts:77`). When unset, each new reasoning block
    /// receives a deterministic `reasoning-N` id from an internal counter,
    /// which is fine for tests and offline replay but does not collide-proof
    /// against ids issued by other sessions or downstream consumers.
    #[must_use]
    pub fn generate_id<F>(mut self, f: F) -> Self
    where
        F: Fn() -> String + Send + Sync + 'static,
    {
        self.generate_id = Some(Arc::new(f));
        self
    }

    /// Finalize the provider.
    ///
    /// # Errors
    ///
    /// - [`ProviderError::load_api_key`] when no explicit key is given and
    ///   `MISTRAL_API_KEY` is unset / empty.
    /// - [`ProviderError`] from [`HttpClient::new`] if the TLS stack fails
    ///   to initialize (rare).
    pub fn build(self) -> Result<Mistral, ProviderError> {
        let api_key = llmsdk_provider_utils::api_key::load_api_key(
            &llmsdk_provider_utils::api_key::LoadApiKey {
                api_key: self.api_key.as_deref(),
                env_var: API_KEY_ENV_VAR,
                description: "Mistral",
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

        Ok(Mistral {
            inner: Arc::new(Inner {
                base_url,
                headers,
                http,
                generate_id: self.generate_id,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_with_explicit_key_succeeds() {
        let m = Mistral::builder().api_key("test-key").build().expect("ok");
        assert_eq!(m.inner.base_url, DEFAULT_BASE_URL);
        assert!(
            m.inner
                .headers
                .get("authorization")
                .unwrap()
                .as_ref()
                .unwrap()
                .starts_with("Bearer ")
        );
    }

    #[test]
    fn builder_custom_base_url() {
        let m = Mistral::builder()
            .api_key("k")
            .base_url("https://proxy.example.com/v1")
            .build()
            .expect("ok");
        assert_eq!(m.inner.base_url, "https://proxy.example.com/v1");
    }

    #[test]
    fn builder_generate_id_is_stored() {
        // Mirrors upstream `mistral-provider.ts` accepting `generateId?: () => string`
        // and forwarding it onto the chat model config (lines 77, 108).
        let m = Mistral::builder()
            .api_key("k")
            .generate_id(|| "custom-id".to_owned())
            .build()
            .expect("ok");
        let gen_fn = m.inner.generate_id.as_ref().expect("generate_id stored");
        assert_eq!(gen_fn(), "custom-id");
    }

    #[test]
    fn builder_custom_header() {
        let m = Mistral::builder()
            .api_key("k")
            .header("x-feature", Some("y".into()))
            .build()
            .expect("ok");
        assert_eq!(
            m.inner.headers.get("x-feature").unwrap().as_deref(),
            Some("y")
        );
    }
}
