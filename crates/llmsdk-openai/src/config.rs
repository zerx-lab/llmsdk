//! Provider configuration and entry point.
//!
//! Mirrors `@ai-sdk/openai/src/openai-provider.ts` (the parts we need for
//! Chat Completions only).
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::sync::Arc;

use llmsdk_provider::ProviderError;
use llmsdk_provider_utils::http::HttpClient;

use crate::chat::OpenAiChatModel;
use crate::embedding::OpenAiEmbeddingModel;
use crate::{API_KEY_ENV_VAR, DEFAULT_BASE_URL};

/// `OpenAI` provider handle — entry point for model construction.
///
/// Cheap to clone; the underlying HTTP client and headers are shared.
#[derive(Debug, Clone)]
pub struct OpenAi {
    inner: Arc<Inner>,
}

#[derive(Debug)]
pub(crate) struct Inner {
    pub(crate) base_url: String,
    pub(crate) headers: HashMap<String, Option<String>>,
    pub(crate) http: HttpClient,
}

impl OpenAi {
    /// Open a [`OpenAiBuilder`].
    #[must_use]
    pub fn builder() -> OpenAiBuilder {
        OpenAiBuilder::default()
    }

    /// Build with defaults: API key from `OPENAI_API_KEY`, default base URL.
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
    /// `model_id` is the `OpenAI` model name, e.g. `"gpt-4o-mini"`.
    #[must_use]
    pub fn chat(&self, model_id: impl Into<String>) -> OpenAiChatModel {
        OpenAiChatModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Construct a text Embeddings model handle.
    ///
    /// `model_id` is the `OpenAI` embedding model name, e.g.
    /// `"text-embedding-3-small"`.
    #[must_use]
    pub fn embedding(&self, model_id: impl Into<String>) -> OpenAiEmbeddingModel {
        OpenAiEmbeddingModel::new(Arc::clone(&self.inner), model_id.into())
    }
}

/// Builder for [`OpenAi`].
///
/// All setters are optional; `build()` falls back to env / library defaults.
#[derive(Debug, Default, Clone)]
pub struct OpenAiBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    organization: Option<String>,
    project: Option<String>,
    extra_headers: HashMap<String, Option<String>>,
    http: Option<HttpClient>,
}

impl OpenAiBuilder {
    /// Set the API key explicitly.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Override the base URL (e.g. for Azure or a local proxy).
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the `OpenAI-Organization` header.
    #[must_use]
    pub fn organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Set the `OpenAI-Project` header.
    #[must_use]
    pub fn project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Append or override a header.
    ///
    /// Passing `None` for `value` removes the header (matching ai-sdk's
    /// `string | undefined` convention).
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
    ///   `OPENAI_API_KEY` is unset / empty.
    /// - [`ProviderError`] from [`HttpClient::new`] if the TLS stack fails
    ///   to initialize (rare).
    pub fn build(self) -> Result<OpenAi, ProviderError> {
        let api_key = llmsdk_provider_utils::api_key::load_api_key(
            &llmsdk_provider_utils::api_key::LoadApiKey {
                api_key: self.api_key.as_deref(),
                env_var: API_KEY_ENV_VAR,
                description: "OpenAI",
                parameter_name: Some("api_key"),
            },
        )?;

        let base_url = self.base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());

        let mut headers = self.extra_headers;
        headers.insert("authorization".into(), Some(format!("Bearer {api_key}")));
        if let Some(org) = self.organization {
            headers.insert("openai-organization".into(), Some(org));
        }
        if let Some(project) = self.project {
            headers.insert("openai-project".into(), Some(project));
        }

        let http = match self.http {
            Some(client) => client,
            None => HttpClient::new()?,
        };

        Ok(OpenAi {
            inner: Arc::new(Inner {
                base_url,
                headers,
                http,
            }),
        })
    }
}
