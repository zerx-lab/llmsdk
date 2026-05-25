//! Provider configuration and entry point.
//!
//! Mirrors `@ai-sdk/xai/src/xai-provider.ts`.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;

use llmsdk_provider::ProviderError;
use llmsdk_provider_utils::http::HttpClient;

use crate::chat::XaiChatModel;
use crate::files::XaiFiles;
use crate::image::XaiImageModel;
use crate::responses::XaiResponsesLanguageModel;
use crate::video::XaiVideoModel;
use crate::{API_KEY_ENV_VAR, DEFAULT_BASE_URL};

/// xAI provider handle — entry point for model construction.
///
/// Cheap to clone; the underlying HTTP client and headers are shared.
#[derive(Debug, Clone)]
pub struct Xai {
    inner: Arc<Inner>,
}

#[derive(Debug)]
pub(crate) struct Inner {
    pub(crate) base_url: String,
    pub(crate) headers: HashMap<String, Option<String>>,
    pub(crate) http: HttpClient,
}

impl Xai {
    /// Open a [`XaiBuilder`].
    #[must_use]
    pub fn builder() -> XaiBuilder {
        XaiBuilder::default()
    }

    /// Build with defaults: API key from `XAI_API_KEY`, default base URL.
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
    /// `model_id` is the xAI model name, e.g. `"grok-4.3"`.
    #[must_use]
    pub fn chat(&self, model_id: impl Into<String>) -> XaiChatModel {
        XaiChatModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::chat`] — mirrors ai-sdk's `provider.languageModel(id)`.
    #[must_use]
    pub fn language_model(&self, model_id: impl Into<String>) -> XaiChatModel {
        self.chat(model_id)
    }

    /// Construct a Files API handle for `POST /v1/files` uploads.
    ///
    /// Mirrors ai-sdk's `xai.files`. The returned handle is cheap to
    /// clone and shares the parent provider's HTTP client and auth header.
    #[must_use]
    pub fn files(&self) -> XaiFiles {
        XaiFiles::new(Arc::clone(&self.inner))
    }

    /// Construct an image-generation model handle.
    ///
    /// `model_id` is the xAI image model name, e.g. `"grok-imagine-image"`
    /// or `"grok-imagine-image-pro"`.
    #[must_use]
    pub fn image(&self, model_id: impl Into<String>) -> XaiImageModel {
        XaiImageModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::image`] — mirrors ai-sdk's `provider.imageModel(id)`.
    #[must_use]
    pub fn image_model(&self, model_id: impl Into<String>) -> XaiImageModel {
        self.image(model_id)
    }

    /// Construct a video-generation model handle.
    ///
    /// `model_id` is the xAI video model name, e.g. `"grok-imagine-video"`.
    /// Mirrors ai-sdk's `xai.video(id)`.
    #[must_use]
    pub fn video(&self, model_id: impl Into<String>) -> XaiVideoModel {
        XaiVideoModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::video`] — mirrors ai-sdk's `provider.videoModel(id)`.
    #[must_use]
    pub fn video_model(&self, model_id: impl Into<String>) -> XaiVideoModel {
        self.video(model_id)
    }

    /// Construct a Responses API model handle (`POST /v1/responses`).
    ///
    /// `model_id` is the xAI model name, e.g. `"grok-4.3"` or
    /// `"grok-4.20-reasoning"`. Mirrors ai-sdk's `xai.responses(id)`.
    #[must_use]
    pub fn responses(&self, model_id: impl Into<String>) -> XaiResponsesLanguageModel {
        XaiResponsesLanguageModel::new(Arc::clone(&self.inner), model_id.into())
    }
}

/// Builder for [`Xai`].
///
/// All setters are optional; `build()` falls back to env / library defaults.
#[derive(Debug, Default, Clone)]
pub struct XaiBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    extra_headers: HashMap<String, Option<String>>,
    http: Option<HttpClient>,
}

impl XaiBuilder {
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

    /// Finalize the provider.
    ///
    /// # Errors
    ///
    /// - [`ProviderError::load_api_key`] when no explicit key is given and
    ///   `XAI_API_KEY` is unset / empty.
    /// - [`ProviderError`] from [`HttpClient::new`] if the TLS stack fails
    ///   to initialize (rare).
    pub fn build(self) -> Result<Xai, ProviderError> {
        let api_key = llmsdk_provider_utils::api_key::load_api_key(
            &llmsdk_provider_utils::api_key::LoadApiKey {
                api_key: self.api_key.as_deref(),
                env_var: API_KEY_ENV_VAR,
                description: "xAI",
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

        Ok(Xai {
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
        let xai = Xai::builder().api_key("xai-test-key").build().expect("ok");
        assert_eq!(xai.inner.base_url, DEFAULT_BASE_URL);
        assert!(
            xai.inner
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
        let xai = Xai::builder()
            .api_key("k")
            .base_url("https://proxy.example.com/v1")
            .build()
            .expect("ok");
        assert_eq!(xai.inner.base_url, "https://proxy.example.com/v1");
    }
}
