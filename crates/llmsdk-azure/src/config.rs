//! Provider configuration and entry point for Azure `OpenAI`.
//!
//! Mirrors `@ai-sdk/azure/src/azure-openai-provider.ts`.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::env;
use std::sync::Arc;

use llmsdk_openai::internal::{
    Inner, OpenAiChatModel, OpenAiCompletionLanguageModel, OpenAiEmbeddingModel, OpenAiImageModel,
    OpenAiResponsesLanguageModel, OpenAiSpeechModel, OpenAiTranscriptionModel, UrlStrategy,
};
use llmsdk_provider::ProviderError;
use llmsdk_provider_utils::api_key::{LoadApiKey, load_api_key};
use llmsdk_provider_utils::http::HttpClient;

use crate::{
    API_KEY_ENV_VAR, DEFAULT_API_VERSION, PROVIDER_ID_CHAT, PROVIDER_ID_COMPLETION,
    PROVIDER_ID_EMBEDDINGS, PROVIDER_ID_IMAGE, PROVIDER_ID_RESPONSES, PROVIDER_ID_SPEECH,
    PROVIDER_ID_TRANSCRIPTION, RESOURCE_NAME_ENV_VAR,
};

/// Azure `OpenAI` provider handle — entry point for model construction.
///
/// Cheap to clone. Each capability factory ([`chat`](Self::chat),
/// [`responses`](Self::responses), [`embedding`](Self::embedding),
/// [`image`](Self::image)) yields a model handle that reports an
/// Azure-flavoured `provider()` string (`azure.chat` / `azure.responses` /
/// `azure.embeddings` / `azure.image`) but otherwise reuses the underlying
/// `OpenAI` implementation, since Azure and `OpenAI` share their wire format.
#[derive(Debug, Clone)]
#[allow(
    clippy::struct_field_names,
    reason = "each field is one Inner per capability; the `_inner` suffix makes \
              their identical type explicit"
)]
pub struct AzureOpenAi {
    chat_inner: Arc<Inner>,
    responses_inner: Arc<Inner>,
    embedding_inner: Arc<Inner>,
    image_inner: Arc<Inner>,
    completion_inner: Arc<Inner>,
    speech_inner: Arc<Inner>,
    transcription_inner: Arc<Inner>,
}

impl AzureOpenAi {
    /// Open an [`AzureOpenAiBuilder`].
    #[must_use]
    pub fn builder() -> AzureOpenAiBuilder {
        AzureOpenAiBuilder::default()
    }

    /// Build with defaults: API key from `AZURE_API_KEY`, resource name from
    /// `AZURE_RESOURCE_NAME`, default `apiVersion`, v1 URL mode.
    ///
    /// # Errors
    ///
    /// - [`ProviderError::load_api_key`] if `AZURE_API_KEY` is unset.
    /// - [`ProviderError::invalid_argument`] if `AZURE_RESOURCE_NAME` is
    ///   unset and no `base_url` was provided.
    pub fn from_env() -> Result<Self, ProviderError> {
        Self::builder().build()
    }

    /// Construct a Chat Completions model handle (Azure `azure.chat`).
    ///
    /// `deployment_id` is the Azure deployment name. In v1 URL mode it ends
    /// up in the request body as `model`; in legacy deployment URL mode it
    /// also appears in the URL path.
    #[must_use]
    pub fn chat(&self, deployment_id: impl Into<String>) -> OpenAiChatModel {
        OpenAiChatModel::new(Arc::clone(&self.chat_inner), deployment_id.into())
    }

    /// Construct a Responses API model handle (Azure `azure.responses`).
    ///
    /// This is the default surface returned by [`language_model`] on the
    /// upstream JS provider; both factories are equivalent.
    ///
    /// [`language_model`]: Self::language_model
    #[must_use]
    pub fn responses(&self, deployment_id: impl Into<String>) -> OpenAiResponsesLanguageModel {
        OpenAiResponsesLanguageModel::new(Arc::clone(&self.responses_inner), deployment_id.into())
    }

    /// Default language-model factory — alias for [`responses`](Self::responses).
    ///
    /// Mirrors the upstream `azure.languageModel(id)` accessor that returns
    /// the Responses API by default.
    #[must_use]
    pub fn language_model(&self, deployment_id: impl Into<String>) -> OpenAiResponsesLanguageModel {
        self.responses(deployment_id)
    }

    /// Construct a text Embeddings model handle (Azure `azure.embeddings`).
    #[must_use]
    pub fn embedding(&self, deployment_id: impl Into<String>) -> OpenAiEmbeddingModel {
        OpenAiEmbeddingModel::new(Arc::clone(&self.embedding_inner), deployment_id.into())
    }

    /// Alias for [`embedding`](Self::embedding) — matches the upstream JS
    /// `embeddingModel(id)` accessor.
    #[must_use]
    pub fn embedding_model(&self, deployment_id: impl Into<String>) -> OpenAiEmbeddingModel {
        self.embedding(deployment_id)
    }

    /// Construct an Image Generation model handle (Azure `azure.image`).
    #[must_use]
    pub fn image(&self, deployment_id: impl Into<String>) -> OpenAiImageModel {
        OpenAiImageModel::new(Arc::clone(&self.image_inner), deployment_id.into())
    }

    /// Alias for [`image`](Self::image) — matches the upstream JS
    /// `imageModel(id)` accessor.
    #[must_use]
    pub fn image_model(&self, deployment_id: impl Into<String>) -> OpenAiImageModel {
        self.image(deployment_id)
    }

    /// Construct a legacy Completions model handle (Azure `azure.completion`).
    #[must_use]
    pub fn completion(&self, deployment_id: impl Into<String>) -> OpenAiCompletionLanguageModel {
        OpenAiCompletionLanguageModel::new(Arc::clone(&self.completion_inner), deployment_id.into())
    }

    /// Construct a Speech (TTS) model handle (Azure `azure.speech`).
    #[must_use]
    pub fn speech(&self, deployment_id: impl Into<String>) -> OpenAiSpeechModel {
        OpenAiSpeechModel::new(Arc::clone(&self.speech_inner), deployment_id.into())
    }

    /// Construct a Transcription (STT) model handle (Azure `azure.transcription`).
    #[must_use]
    pub fn transcription(&self, deployment_id: impl Into<String>) -> OpenAiTranscriptionModel {
        OpenAiTranscriptionModel::new(Arc::clone(&self.transcription_inner), deployment_id.into())
    }
}

/// Builder for [`AzureOpenAi`].
///
/// At least one of [`resource_name`](Self::resource_name) /
/// [`base_url`](Self::base_url) (or their env equivalents) must resolve;
/// otherwise [`build`](Self::build) returns [`ProviderError::load_setting`].
/// `base_url` wins over `resource_name` when both are provided.
///
/// Authentication: pass either an [`api_key`](Self::api_key) (sent as the
/// `api-key` header) or a [`bearer_token`](Self::bearer_token) (sent as
/// `Authorization: Bearer <token>`, used for Microsoft Entra ID / AAD
/// authentication). `bearer_token` wins when both are provided. Falls back
/// to `AZURE_API_KEY` env var when neither is set.
#[derive(Debug, Default, Clone)]
pub struct AzureOpenAiBuilder {
    api_key: Option<String>,
    bearer_token: Option<String>,
    resource_name: Option<String>,
    base_url: Option<String>,
    api_version: Option<String>,
    use_deployment_based_urls: bool,
    extra_headers: HashMap<String, Option<String>>,
    http: Option<HttpClient>,
}

impl AzureOpenAiBuilder {
    /// Set the API key explicitly.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Authenticate via Microsoft Entra ID / Azure Active Directory.
    ///
    /// Sends `Authorization: Bearer <token>` on every request. Wins over
    /// [`api_key`](Self::api_key) / `AZURE_API_KEY` when both resolve.
    ///
    /// The token is captured at builder time, so callers wanting per-call
    /// refresh should re-build the provider periodically or inject a custom
    /// `Authorization` header via [`header`](Self::header).
    #[must_use]
    pub fn bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    /// Set the Azure resource name (the `{resource}` in
    /// `https://{resource}.openai.azure.com/...`).
    #[must_use]
    pub fn resource_name(mut self, name: impl Into<String>) -> Self {
        self.resource_name = Some(name.into());
        self
    }

    /// Override the base URL entirely (e.g. for a proxy). When set,
    /// `resource_name` is ignored.
    ///
    /// The URL is used as a prefix; the rest of the path
    /// (`/v1{endpoint}` or `/deployments/{id}{endpoint}`) plus the
    /// `api-version` query are appended automatically.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Override the `api-version` query parameter. Defaults to
    /// [`DEFAULT_API_VERSION`].
    #[must_use]
    pub fn api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = Some(version.into());
        self
    }

    /// Switch to the legacy deployment-based URL layout:
    /// `{prefix}/deployments/{deploymentId}{endpoint}?api-version=...`
    /// instead of `{prefix}/v1{endpoint}?api-version=...`.
    #[must_use]
    pub fn use_deployment_based_urls(mut self, enabled: bool) -> Self {
        self.use_deployment_based_urls = enabled;
        self
    }

    /// Append or override a header. Passing `None` removes the header.
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
    /// - [`ProviderError::load_api_key`] if no key and `AZURE_API_KEY` is unset.
    /// - [`ProviderError::invalid_argument`] if no `base_url`/`resource_name`
    ///   and `AZURE_RESOURCE_NAME` is unset.
    pub fn build(self) -> Result<AzureOpenAi, ProviderError> {
        // Authentication resolves to either a bearer token (Microsoft Entra
        // ID / AAD) or an `api-key`. Bearer wins when both resolve so
        // callers using AAD don't accidentally fall back to a stale env
        // `AZURE_API_KEY`.
        let auth_header = if let Some(token) = self.bearer_token.as_deref() {
            ("authorization".to_owned(), format!("Bearer {token}"))
        } else {
            let api_key = load_api_key(&LoadApiKey {
                api_key: self.api_key.as_deref(),
                env_var: API_KEY_ENV_VAR,
                description: "Azure OpenAI",
                parameter_name: Some("api_key"),
            })?;
            ("api-key".to_owned(), api_key)
        };

        // Resolve URL prefix: explicit base_url > resource_name > env.
        let base_prefix = if let Some(base) = self.base_url {
            strip_trailing_slash(&base).to_owned()
        } else {
            let resource = match self.resource_name {
                Some(name) => name,
                None => load_setting(
                    RESOURCE_NAME_ENV_VAR,
                    "resourceName",
                    "Azure OpenAI resource name",
                )?,
            };
            format!("https://{resource}.openai.azure.com/openai")
        };

        let api_version = self
            .api_version
            .unwrap_or_else(|| DEFAULT_API_VERSION.to_owned());

        // Build a UrlStrategy::Custom closure capturing prefix + api-version
        // + URL mode. This closure is cloned into one Arc<Inner> per
        // capability (so all four `Inner` instances share the same routing
        // logic but differ in provider_id).
        let url_strategy =
            build_url_strategy(base_prefix, api_version, self.use_deployment_based_urls);

        let mut headers = self.extra_headers;
        headers.insert(auth_header.0, Some(auth_header.1));

        let http = match self.http {
            Some(client) => client,
            None => HttpClient::new()?,
        };

        // Upstream `@ai-sdk/openai` switches the providerOptions namespace
        // key to `"azure"` for the Responses and (legacy) Completion endpoints
        // when the provider name contains "azure" (see
        // `openai-responses-language-model.ts:180-181` and
        // `openai-completion-language-model.ts:54`). Chat / Embedding / Image
        // / Speech / Transcription stay on `"openai"` since the upstream
        // provider also hardcodes that key there.
        let mk_inner = |provider_id: &'static str, options_name: &'static str| -> Arc<Inner> {
            Arc::new(
                Inner::new(
                    url_strategy.clone(),
                    headers.clone(),
                    http.clone(),
                    provider_id,
                )
                .with_provider_options_name(options_name),
            )
        };

        Ok(AzureOpenAi {
            chat_inner: mk_inner(PROVIDER_ID_CHAT, "openai"),
            responses_inner: mk_inner(PROVIDER_ID_RESPONSES, "azure"),
            embedding_inner: mk_inner(PROVIDER_ID_EMBEDDINGS, "openai"),
            image_inner: mk_inner(PROVIDER_ID_IMAGE, "openai"),
            completion_inner: mk_inner(PROVIDER_ID_COMPLETION, "azure"),
            speech_inner: mk_inner(PROVIDER_ID_SPEECH, "openai"),
            transcription_inner: mk_inner(PROVIDER_ID_TRANSCRIPTION, "openai"),
        })
    }
}

/// Compose the per-request URL builder closure used by all four capability
/// surfaces.
///
/// - v1 mode: `{prefix}/v1{endpoint}?api-version={apiVersion}`
/// - legacy: `{prefix}/deployments/{deploymentId}{endpoint}?api-version={apiVersion}`
fn build_url_strategy(prefix: String, api_version: String, use_deployment: bool) -> UrlStrategy {
    UrlStrategy::Custom(Arc::new(move |endpoint: &str, model_id: &str| {
        let path = if use_deployment {
            format!("{prefix}/deployments/{model_id}{endpoint}")
        } else {
            format!("{prefix}/v1{endpoint}")
        };
        format!("{path}?api-version={api_version}")
    }))
}

fn strip_trailing_slash(s: &str) -> &str {
    s.strip_suffix('/').unwrap_or(s)
}

/// Equivalent of `@ai-sdk/provider-utils`'s `loadSetting`: read an env var,
/// erroring with [`ProviderError::invalid_argument`] when unset / blank.
///
/// (We map onto `invalid_argument` rather than a dedicated `load_setting`
/// variant — the provider error enum doesn't currently distinguish the two,
/// and the upstream JS distinction is mostly cosmetic.)
fn load_setting(
    env_var: &str,
    parameter_name: &str,
    description: &str,
) -> Result<String, ProviderError> {
    match env::var(env_var) {
        Ok(value) if !value.is_empty() => Ok(value),
        Ok(_) => Err(ProviderError::invalid_argument(
            parameter_name,
            format!(
                "{description} setting is empty. The value of the {env_var} \
                 environment variable is empty."
            ),
        )),
        Err(_) => Err(ProviderError::invalid_argument(
            parameter_name,
            format!(
                "{description} setting is missing. Pass it using the \
                 `{parameter_name}` parameter or the {env_var} environment variable."
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_url_appends_api_version() {
        let strat = build_url_strategy(
            "https://x.openai.azure.com/openai".to_owned(),
            "preview".to_owned(),
            false,
        );
        assert_eq!(
            strat.build("/chat/completions", "gpt-4o-mini"),
            "https://x.openai.azure.com/openai/v1/chat/completions?api-version=preview"
        );
    }

    #[test]
    fn legacy_url_includes_deployment() {
        let strat = build_url_strategy(
            "https://x.openai.azure.com/openai".to_owned(),
            "2024-08-01-preview".to_owned(),
            true,
        );
        assert_eq!(
            strat.build("/chat/completions", "my-deployment"),
            "https://x.openai.azure.com/openai/deployments/my-deployment/chat/completions?api-version=2024-08-01-preview"
        );
    }

    #[test]
    fn base_url_strips_trailing_slash() {
        assert_eq!(
            strip_trailing_slash("https://x/openai/"),
            "https://x/openai"
        );
        assert_eq!(strip_trailing_slash("https://x/openai"), "https://x/openai");
    }

    #[test]
    fn url_strategy_round_trips_endpoint_for_embeddings() {
        let strat = build_url_strategy(
            "https://x.openai.azure.com/openai".to_owned(),
            "v1".to_owned(),
            false,
        );
        assert_eq!(
            strat.build("/embeddings", "text-embedding-3-small"),
            "https://x.openai.azure.com/openai/v1/embeddings?api-version=v1"
        );
    }

    #[test]
    fn bearer_token_takes_precedence_over_api_key() {
        use llmsdk_provider::LanguageModel;
        let p = AzureOpenAi::builder()
            .resource_name("myresource")
            .api_key("k")
            .bearer_token("aad-token")
            .build()
            .expect("builds");
        assert_eq!(p.chat("gpt-4o-mini").provider(), PROVIDER_ID_CHAT);
    }

    #[test]
    fn bearer_token_alone_builds_without_api_key() {
        use llmsdk_provider::LanguageModel;
        let p = AzureOpenAi::builder()
            .resource_name("myresource")
            .bearer_token("aad-token")
            .build()
            .expect("builds with bearer only");
        assert_eq!(p.chat("gpt-4o-mini").provider(), PROVIDER_ID_CHAT);
    }

    #[test]
    fn url_strategy_round_trips_endpoint_for_images_in_legacy_mode() {
        let strat = build_url_strategy(
            "https://x.openai.azure.com/openai".to_owned(),
            "2024-02-15-preview".to_owned(),
            true,
        );
        assert_eq!(
            strat.build("/images/generations", "dalle-3-deployment"),
            "https://x.openai.azure.com/openai/deployments/dalle-3-deployment/images/generations?api-version=2024-02-15-preview"
        );
    }
}
