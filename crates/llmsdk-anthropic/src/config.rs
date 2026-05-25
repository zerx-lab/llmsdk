//! Provider configuration.
//!
//! Mirrors `@ai-sdk/anthropic/src/anthropic-provider.ts`. `Anthropic` uses
//! `x-api-key` by default (or `Authorization: Bearer` when `auth_token` is
//! set) and always sends the `anthropic-version` header.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::sync::Arc;

use llmsdk_provider::ProviderError;
use llmsdk_provider_utils::http::HttpClient;

use crate::files::AnthropicFiles;
use crate::messages::AnthropicMessagesModel;
use crate::skills::AnthropicSkills;
use crate::{
    API_KEY_ENV_VAR, AUTH_TOKEN_ENV_VAR, DEFAULT_ANTHROPIC_VERSION, DEFAULT_BASE_URL, PROVIDER_ID,
};

const DEFAULT_PROVIDER_NAME: &str = "anthropic.messages";

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
    pub(crate) provider_name: String,
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
    ///
    /// Mirrors `provider.messages(modelId)` upstream.
    #[must_use]
    pub fn messages(&self, model_id: impl Into<String>) -> AnthropicMessagesModel {
        AnthropicMessagesModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Anthropic::messages`] — mirrors `provider.chat(modelId)`.
    #[must_use]
    pub fn chat(&self, model_id: impl Into<String>) -> AnthropicMessagesModel {
        self.messages(model_id)
    }

    /// Alias of [`Anthropic::messages`] — mirrors `provider.languageModel(modelId)`.
    #[must_use]
    pub fn language_model(&self, model_id: impl Into<String>) -> AnthropicMessagesModel {
        self.messages(model_id)
    }

    /// Files API handle (`POST /v1/files`).
    ///
    /// Mirrors `provider.files()`.
    #[must_use]
    pub fn files(&self) -> AnthropicFiles {
        let provider = format!(
            "{}.files",
            self.inner
                .provider_name
                .strip_suffix(".messages")
                .unwrap_or(&self.inner.provider_name)
        );
        AnthropicFiles::new(Arc::clone(&self.inner), provider)
    }

    /// Skills API handle (`POST /v1/skills`).
    ///
    /// Mirrors `provider.skills()`.
    #[must_use]
    pub fn skills(&self) -> AnthropicSkills {
        let provider = format!(
            "{}.skills",
            self.inner
                .provider_name
                .strip_suffix(".messages")
                .unwrap_or(&self.inner.provider_name)
        );
        AnthropicSkills::new(Arc::clone(&self.inner), provider)
    }

    /// Provider id reported on language-model handles.
    #[must_use]
    pub fn provider_name(&self) -> &str {
        &self.inner.provider_name
    }
}

/// Builder for [`Anthropic`].
#[derive(Debug, Default, Clone)]
pub struct AnthropicBuilder {
    api_key: Option<String>,
    auth_token: Option<String>,
    base_url: Option<String>,
    version: Option<String>,
    provider_name: Option<String>,
    extra_headers: HashMap<String, Option<String>>,
    http: Option<HttpClient>,
}

impl AnthropicBuilder {
    /// Set the API key explicitly. Mutually exclusive with [`Self::auth_token`].
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the bearer auth token explicitly. Mutually exclusive with
    /// [`Self::api_key`]. When set the provider sends
    /// `Authorization: Bearer {token}` instead of `x-api-key`.
    #[must_use]
    pub fn auth_token(mut self, token: impl Into<String>) -> Self {
        self.auth_token = Some(token.into());
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

    /// Override the reported provider name (default `"anthropic.messages"`).
    ///
    /// The `.files` / `.skills` suffix is derived automatically when calling
    /// [`Anthropic::files`] / [`Anthropic::skills`].
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.provider_name = Some(name.into());
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
    /// - [`ProviderError::invalid_argument`] when both `api_key` and
    ///   `auth_token` are explicitly set.
    /// - [`ProviderError::load_api_key`] when neither an explicit key nor
    ///   `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_API_KEY` is available.
    /// - [`ProviderError`] when the HTTP client cannot initialize.
    pub fn build(self) -> Result<Anthropic, ProviderError> {
        if self.api_key.is_some() && self.auth_token.is_some() {
            return Err(ProviderError::invalid_argument(
                "api_key/auth_token",
                "Both apiKey and authToken were provided. Please use only one authentication method.",
            ));
        }

        let resolved_auth_token = self
            .auth_token
            .clone()
            .or_else(|| std::env::var(AUTH_TOKEN_ENV_VAR).ok())
            .filter(|s| !s.is_empty());

        let mut headers = self.extra_headers;
        if let Some(token) = resolved_auth_token {
            headers.insert("Authorization".into(), Some(format!("Bearer {token}")));
        } else {
            let api_key = llmsdk_provider_utils::api_key::load_api_key(
                &llmsdk_provider_utils::api_key::LoadApiKey {
                    api_key: self.api_key.as_deref(),
                    env_var: API_KEY_ENV_VAR,
                    description: "Anthropic",
                    parameter_name: Some("api_key"),
                },
            )?;
            headers.insert("x-api-key".into(), Some(api_key));
        }

        let version = self
            .version
            .unwrap_or_else(|| DEFAULT_ANTHROPIC_VERSION.to_owned());
        headers.insert("anthropic-version".into(), Some(version));

        let base_url = self.base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());

        let provider_name = self
            .provider_name
            .unwrap_or_else(|| DEFAULT_PROVIDER_NAME.to_owned());

        let _ = PROVIDER_ID;

        let http = match self.http {
            Some(client) => client,
            None => HttpClient::new()?,
        };

        Ok(Anthropic {
            inner: Arc::new(Inner {
                base_url,
                headers,
                http,
                provider_name,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::{FilesModel, LanguageModel, SkillsModel};

    fn fixed_key() -> Anthropic {
        Anthropic::builder()
            .api_key("test-key")
            .build()
            .expect("builder should succeed with explicit key")
    }

    #[test]
    fn explicit_api_key_sets_xapikey_header() {
        let a = fixed_key();
        assert_eq!(
            a.inner.headers.get("x-api-key"),
            Some(&Some("test-key".to_owned()))
        );
        assert!(!a.inner.headers.contains_key("Authorization"));
    }

    #[test]
    fn explicit_auth_token_sets_bearer_header() {
        let a = Anthropic::builder()
            .auth_token("oauth-token")
            .build()
            .unwrap();
        assert_eq!(
            a.inner.headers.get("Authorization"),
            Some(&Some("Bearer oauth-token".to_owned()))
        );
        assert!(!a.inner.headers.contains_key("x-api-key"));
    }

    #[test]
    fn both_api_key_and_auth_token_errors() {
        let err = Anthropic::builder()
            .api_key("k")
            .auth_token("t")
            .build()
            .unwrap_err();
        assert!(format!("{err}").contains("Both apiKey and authToken were provided"));
    }

    #[test]
    fn chat_and_language_model_are_messages_aliases() {
        let a = fixed_key();
        let m = a.messages("claude-sonnet-4-6");
        let c = a.chat("claude-sonnet-4-6");
        let l = a.language_model("claude-sonnet-4-6");
        assert_eq!(m.model_id(), c.model_id());
        assert_eq!(m.model_id(), l.model_id());
    }

    #[test]
    fn files_handle_reports_provider_suffix() {
        let a = fixed_key();
        let f = a.files();
        assert_eq!(f.provider(), "anthropic.files");
    }

    #[test]
    fn skills_handle_reports_provider_suffix() {
        let a = fixed_key();
        let s = a.skills();
        assert_eq!(s.provider(), "anthropic.skills");
    }

    #[test]
    fn custom_provider_name_propagates_to_files_skills() {
        let a = Anthropic::builder()
            .api_key("k")
            .name("acme.bedrock.anthropic.messages")
            .build()
            .unwrap();
        assert_eq!(a.files().provider(), "acme.bedrock.anthropic.files");
        assert_eq!(a.skills().provider(), "acme.bedrock.anthropic.skills");
    }

    #[test]
    fn anthropic_version_header_present() {
        let a = fixed_key();
        assert_eq!(
            a.inner.headers.get("anthropic-version"),
            Some(&Some(DEFAULT_ANTHROPIC_VERSION.to_owned()))
        );
    }
}
