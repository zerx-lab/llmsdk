//! Provider configuration.
//!
//! Mirrors `@ai-sdk/anthropic/src/anthropic-provider.ts` (subset).
//! `Anthropic` uses `x-api-key` rather than `Authorization: Bearer` and
//! requires the `anthropic-version` header on every call.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::sync::Arc;

use llmsdk_provider::ProviderError;
use llmsdk_provider_utils::http::HttpClient;

use crate::messages::AnthropicMessagesModel;
use crate::{API_KEY_ENV_VAR, DEFAULT_ANTHROPIC_VERSION, DEFAULT_BASE_URL};

/// `Anthropic` provider handle.
///
/// Cheap to clone; HTTP client and headers are shared.
#[derive(Debug, Clone)]
pub struct Anthropic {
    inner: Arc<Inner>,
}

#[derive(Debug)]
pub(crate) struct Inner {
    pub(crate) base_url: String,
    pub(crate) headers: HashMap<String, Option<String>>,
    pub(crate) http: HttpClient,
}

impl Anthropic {
    /// Open a builder.
    #[must_use]
    pub fn builder() -> AnthropicBuilder {
        AnthropicBuilder::default()
    }

    /// Build with defaults: API key from `ANTHROPIC_API_KEY`.
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError::load_api_key`] when the env var is unset
    /// or empty.
    pub fn from_env() -> Result<Self, ProviderError> {
        Self::builder().build()
    }

    /// Construct a Messages API model handle.
    #[must_use]
    pub fn messages(&self, model_id: impl Into<String>) -> AnthropicMessagesModel {
        AnthropicMessagesModel::new(Arc::clone(&self.inner), model_id.into())
    }
}

/// Builder for [`Anthropic`].
#[derive(Debug, Default, Clone)]
pub struct AnthropicBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    version: Option<String>,
    extra_headers: HashMap<String, Option<String>>,
    http: Option<HttpClient>,
}

impl AnthropicBuilder {
    /// Set the API key explicitly.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Override the base URL (e.g. for a proxy).
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Override the `anthropic-version` header (defaults to
    /// [`crate::DEFAULT_ANTHROPIC_VERSION`]).
    #[must_use]
    pub fn version(mut self, value: impl Into<String>) -> Self {
        self.version = Some(value.into());
        self
    }

    /// Append or override an extra header. `None` removes it.
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
    ///   `ANTHROPIC_API_KEY` is unset / empty.
    /// - [`ProviderError`] when the HTTP client cannot initialize.
    pub fn build(self) -> Result<Anthropic, ProviderError> {
        let api_key = llmsdk_provider_utils::api_key::load_api_key(
            &llmsdk_provider_utils::api_key::LoadApiKey {
                api_key: self.api_key.as_deref(),
                env_var: API_KEY_ENV_VAR,
                description: "Anthropic",
                parameter_name: Some("api_key"),
            },
        )?;

        let base_url = self.base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());

        let version = self
            .version
            .unwrap_or_else(|| DEFAULT_ANTHROPIC_VERSION.to_owned());

        let mut headers = self.extra_headers;
        headers.insert("x-api-key".into(), Some(api_key));
        headers.insert("anthropic-version".into(), Some(version));

        let http = match self.http {
            Some(client) => client,
            None => HttpClient::new()?,
        };

        Ok(Anthropic {
            inner: Arc::new(Inner {
                base_url,
                headers,
                http,
            }),
        })
    }
}
