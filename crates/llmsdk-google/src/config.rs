//! Provider configuration and entry point.
//!
//! Mirrors `@ai-sdk/google/src/google-provider.ts`. Adds an `Arc`-shared
//! [`Inner`] state holding the resolved API key header, base URL, and HTTP
//! client; all model handles are cheap to clone and share this state.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;

use llmsdk_provider::ProviderError;
use llmsdk_provider_utils::http::HttpClient;

use crate::embedding::GoogleEmbeddingModel;
use crate::files::GoogleFiles;
use crate::image::GoogleImageModel;
use crate::interactions::{GoogleInteractionsAgent, GoogleInteractionsLanguageModel};
use crate::language::GoogleLanguageModel;
use crate::video::GoogleVideoModel;
use crate::{API_KEY_ENV_VAR, DEFAULT_BASE_URL, PROVIDER_ID};

/// Google Gemini provider handle.
///
/// Cheap to clone; the underlying HTTP client and headers are shared via
/// [`Arc`].
#[derive(Debug, Clone)]
pub struct Google {
    inner: Arc<Inner>,
}

/// Internal connection / routing state shared across all model handles
/// produced by a single provider instance.
///
/// Public for cross-crate wrapping providers (e.g. Google Vertex) — *not*
/// part of the user-facing surface. Re-exported under [`crate::internal`].
#[derive(Debug)]
pub struct Inner {
    pub(crate) provider: String,
    pub(crate) base_url: String,
    pub(crate) headers: HashMap<String, Option<String>>,
    pub(crate) http: HttpClient,
}

impl Inner {
    /// Open a typed builder for [`Inner`].
    ///
    /// Cross-crate composition entry point: wrapping providers build the
    /// [`Inner`] directly with custom provider name / base URL / headers /
    /// HTTP client and inject it into one of the model `new()` constructors.
    #[must_use]
    pub fn builder() -> InnerBuilder {
        InnerBuilder::default()
    }
}

/// Builder for the cross-crate [`Inner`].
///
/// Used by wrapping providers (Google Vertex) to assemble an [`Inner`]
/// without going through the user-facing [`Google`] builder.
#[derive(Debug, Default, Clone)]
pub struct InnerBuilder {
    provider: Option<String>,
    base_url: Option<String>,
    headers: HashMap<String, Option<String>>,
    http: Option<HttpClient>,
}

impl InnerBuilder {
    /// Override the provider name reported by `*Model::provider()`.
    ///
    /// For Vertex, set this to `"google.vertex.chat"` to enable the Vertex
    /// option-key paths (`googleVertex` / `vertex`) inside the language
    /// model.
    #[must_use]
    pub fn provider(mut self, value: impl Into<String>) -> Self {
        self.provider = Some(value.into());
        self
    }

    /// Override the base URL.
    #[must_use]
    pub fn base_url(mut self, value: impl Into<String>) -> Self {
        self.base_url = Some(value.into());
        self
    }

    /// Append or override a header. `None` removes it.
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: Option<String>) -> Self {
        self.headers.insert(name.into(), value);
        self
    }

    /// Inject a pre-configured HTTP client.
    #[must_use]
    pub fn http_client(mut self, client: HttpClient) -> Self {
        self.http = Some(client);
        self
    }

    /// Finalize the [`Inner`].
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError`] when no HTTP client is provided and the
    /// default client fails to build (rare; misconfigured TLS).
    pub fn build(self) -> Result<Inner, ProviderError> {
        let http = match self.http {
            Some(client) => client,
            None => HttpClient::new()?,
        };
        Ok(Inner {
            provider: self.provider.unwrap_or_else(|| PROVIDER_ID.to_owned()),
            base_url: self.base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_owned()),
            headers: self.headers,
            http,
        })
    }
}

impl Google {
    /// Open a [`GoogleBuilder`].
    #[must_use]
    pub fn builder() -> GoogleBuilder {
        GoogleBuilder::default()
    }

    /// Build with defaults: API key from `GOOGLE_GENERATIVE_AI_API_KEY`,
    /// default base URL.
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError::load_api_key`] when the env var is unset
    /// or empty.
    pub fn from_env() -> Result<Self, ProviderError> {
        Self::builder().build()
    }

    /// Construct a language-model handle (Gemini generateContent /
    /// streamGenerateContent).
    ///
    /// `model_id` is a Gemini model name, e.g. `"gemini-2.5-flash"` or
    /// `"gemini-3-pro-preview"`.
    #[must_use]
    pub fn language_model(&self, model_id: impl Into<String>) -> GoogleLanguageModel {
        GoogleLanguageModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::language_model`] — mirrors ai-sdk's `chat(id)`.
    #[must_use]
    pub fn chat(&self, model_id: impl Into<String>) -> GoogleLanguageModel {
        self.language_model(model_id)
    }

    /// Alias of [`Self::language_model`] — mirrors ai-sdk's deprecated
    /// `generativeAI(id)` factory.
    #[must_use]
    pub fn generative_ai(&self, model_id: impl Into<String>) -> GoogleLanguageModel {
        self.language_model(model_id)
    }

    /// Construct a text-embedding model handle.
    ///
    /// `model_id` is a Gemini embedding model name, e.g.
    /// `"gemini-embedding-001"`.
    #[must_use]
    pub fn embedding(&self, model_id: impl Into<String>) -> GoogleEmbeddingModel {
        GoogleEmbeddingModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::embedding`].
    #[must_use]
    pub fn embedding_model(&self, model_id: impl Into<String>) -> GoogleEmbeddingModel {
        self.embedding(model_id)
    }

    /// Alias of [`Self::embedding`] — mirrors ai-sdk's deprecated
    /// `textEmbedding(id)`.
    #[must_use]
    pub fn text_embedding(&self, model_id: impl Into<String>) -> GoogleEmbeddingModel {
        self.embedding(model_id)
    }

    /// Alias of [`Self::embedding`] — mirrors ai-sdk's deprecated
    /// `textEmbeddingModel(id)`.
    #[must_use]
    pub fn text_embedding_model(&self, model_id: impl Into<String>) -> GoogleEmbeddingModel {
        self.embedding(model_id)
    }

    /// Construct an image-generation model handle. Supports Imagen models
    /// (`imagen-*`) via the `:predict` endpoint and Gemini image-output
    /// models (`gemini-*-image*`) via the language-model path with
    /// `responseModalities: ["IMAGE"]`.
    #[must_use]
    pub fn image(&self, model_id: impl Into<String>) -> GoogleImageModel {
        GoogleImageModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::image`].
    #[must_use]
    pub fn image_model(&self, model_id: impl Into<String>) -> GoogleImageModel {
        self.image(model_id)
    }

    /// Construct a video-generation model handle (Veo `:predictLongRunning`
    /// + async polling).
    #[must_use]
    pub fn video(&self, model_id: impl Into<String>) -> GoogleVideoModel {
        GoogleVideoModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::video`].
    #[must_use]
    pub fn video_model(&self, model_id: impl Into<String>) -> GoogleVideoModel {
        self.video(model_id)
    }

    /// Construct a Files API handle (`POST /upload/v1beta/files`).
    #[must_use]
    pub fn files(&self) -> GoogleFiles {
        GoogleFiles::new(Arc::clone(&self.inner))
    }

    /// Construct an Interactions API handle (`POST /v1beta/interactions`).
    ///
    /// Pass a Gemini model id (`GoogleInteractionsAgent::Model("gemini-2.5-flash")`)
    /// for ad-hoc inference, or an agent / managed-agent resource for the
    /// orchestrated agent runtime.
    #[must_use]
    pub fn interactions(&self, agent: GoogleInteractionsAgent) -> GoogleInteractionsLanguageModel {
        GoogleInteractionsLanguageModel::new(Arc::clone(&self.inner), agent)
    }
}

/// Builder for [`Google`].
///
/// All setters are optional; `build()` falls back to env / library defaults.
#[derive(Debug, Default, Clone)]
pub struct GoogleBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    name: Option<String>,
    extra_headers: HashMap<String, Option<String>>,
    http: Option<HttpClient>,
}

impl GoogleBuilder {
    /// Set the API key explicitly.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Override the base URL (e.g. to use a regional proxy).
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Override the provider name reported by `*Model::provider()`.
    ///
    /// Defaults to `"google"`. Useful for multi-tenant setups that route
    /// telemetry by provider id.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Append or override a header. `None` removes the header.
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
    ///   `GOOGLE_GENERATIVE_AI_API_KEY` is unset / empty.
    /// - [`ProviderError`] from [`HttpClient::new`] when the TLS stack
    ///   fails to initialize.
    pub fn build(self) -> Result<Google, ProviderError> {
        let api_key = llmsdk_provider_utils::api_key::load_api_key(
            &llmsdk_provider_utils::api_key::LoadApiKey {
                api_key: self.api_key.as_deref(),
                env_var: API_KEY_ENV_VAR,
                description: "Google Generative AI",
                parameter_name: Some("api_key"),
            },
        )?;

        let base_url = self
            .base_url
            .map(|s| s.trim_end_matches('/').to_owned())
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());

        let provider = self.name.unwrap_or_else(|| PROVIDER_ID.to_owned());

        let mut headers = self.extra_headers;
        headers.insert("x-goog-api-key".into(), Some(api_key));

        let http = match self.http {
            Some(client) => client,
            None => HttpClient::new()?,
        };

        Ok(Google {
            inner: Arc::new(Inner {
                provider,
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
    fn builder_with_explicit_key() {
        let g = Google::builder().api_key("k").build().expect("ok");
        assert_eq!(g.inner.base_url, DEFAULT_BASE_URL);
        assert_eq!(
            g.inner
                .headers
                .get("x-goog-api-key")
                .and_then(|v| v.as_deref()),
            Some("k")
        );
        assert_eq!(g.inner.provider, "google");
    }

    #[test]
    fn builder_trims_trailing_slash() {
        let g = Google::builder()
            .api_key("k")
            .base_url("https://proxy/v1beta/")
            .build()
            .expect("ok");
        assert_eq!(g.inner.base_url, "https://proxy/v1beta");
    }

    #[test]
    fn builder_custom_name() {
        let g = Google::builder()
            .api_key("k")
            .name("google.tenant1")
            .build()
            .expect("ok");
        assert_eq!(g.inner.provider, "google.tenant1");
    }

    #[test]
    fn builder_missing_key_errors() {
        // Use a different env var key path that we know is unset.
        let result = Google::builder().build();
        // Note: passes when GOOGLE_GENERATIVE_AI_API_KEY happens to be set
        // in the dev environment; we tolerate both branches.
        if std::env::var(API_KEY_ENV_VAR).is_err() {
            assert!(result.is_err());
        }
    }
}
