//! Provider configuration and entry point.
//!
//! Mirrors `@ai-sdk/openai/src/openai-provider.ts` (the parts we need for
//! Chat Completions only).
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use llmsdk_provider::ProviderError;
use llmsdk_provider_utils::http::HttpClient;

use crate::chat::OpenAiChatModel;
use crate::embedding::OpenAiEmbeddingModel;
use crate::image::OpenAiImageModel;
use crate::responses::OpenAiResponsesLanguageModel;
use crate::{API_KEY_ENV_VAR, DEFAULT_BASE_URL, PROVIDER_ID};

/// `OpenAI` provider handle — entry point for model construction.
///
/// Cheap to clone; the underlying HTTP client and headers are shared.
#[derive(Debug, Clone)]
pub struct OpenAi {
    inner: Arc<Inner>,
}

/// How a model maps an endpoint name (`"/chat/completions"`, `"/embeddings"`, ...)
/// into a fully qualified request URL.
///
/// Type alias for the per-request URL builder closure used by
/// [`UrlStrategy::Custom`]. Called as `builder(endpoint, model_id)`; must
/// return an absolute URL.
pub type CustomUrlFn = dyn Fn(&str, &str) -> String + Send + Sync;

/// Default `OpenAI` behaviour just appends the endpoint to `base_url`.
/// Wrapping providers (Azure `OpenAI` in particular) need to inject a
/// deployment id into the path and / or append an `api-version` query
/// string, which the default concatenation cannot express. They wire in a
/// [`UrlStrategy::Custom`] closure that returns the final URL string per
/// request.
#[derive(Clone)]
pub enum UrlStrategy {
    /// Default behaviour: `format!("{base_url}{endpoint}")`. `endpoint`
    /// always begins with `/` (e.g. `"/chat/completions"`).
    Standard {
        /// Base URL with no trailing slash, e.g. `"https://api.openai.com/v1"`.
        base_url: String,
    },
    /// Custom URL builder. Called as `f(endpoint, model_id)` once per
    /// request; must return an absolute URL.
    Custom(Arc<CustomUrlFn>),
}

impl fmt::Debug for UrlStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard { base_url } => f
                .debug_struct("Standard")
                .field("base_url", base_url)
                .finish(),
            Self::Custom(_) => f.debug_struct("Custom").finish_non_exhaustive(),
        }
    }
}

impl UrlStrategy {
    /// Resolve `endpoint` (e.g. `"/chat/completions"`) for `model_id`.
    #[must_use]
    pub fn build(&self, endpoint: &str, model_id: &str) -> String {
        match self {
            Self::Standard { base_url } => format!("{base_url}{endpoint}"),
            Self::Custom(f) => f(endpoint, model_id),
        }
    }

    /// True for `Custom` strategies that need per-request URL composition.
    #[must_use]
    pub fn is_custom(&self) -> bool {
        matches!(self, Self::Custom(_))
    }
}

/// Internal connection / routing state shared across all model handles
/// produced by a single provider instance.
///
/// Public for cross-crate wrapping providers (e.g. Azure `OpenAI`) — *not* part
/// of the user-facing surface. Re-exported under [`crate::internal`].
#[derive(Debug)]
pub struct Inner {
    pub(crate) url: UrlStrategy,
    pub(crate) headers: HashMap<String, Option<String>>,
    pub(crate) http: HttpClient,
    pub(crate) provider_id: &'static str,
}

impl Inner {
    /// Build a new `Inner` directly. Cross-crate constructor for wrapping
    /// providers; pass a [`UrlStrategy::Custom`] when you need deployment-based
    /// URLs or query-string `api-version` parameters.
    #[must_use]
    pub fn new(
        url: UrlStrategy,
        headers: HashMap<String, Option<String>>,
        http: HttpClient,
        provider_id: &'static str,
    ) -> Self {
        Self {
            url,
            headers,
            http,
            provider_id,
        }
    }

    /// Resolve `endpoint` for `model_id` using the configured strategy.
    #[must_use]
    pub(crate) fn endpoint(&self, endpoint: &str, model_id: &str) -> String {
        self.url.build(endpoint, model_id)
    }

    /// Provider identifier reported via `LanguageModel::provider` / similar.
    #[must_use]
    pub(crate) fn provider_id(&self) -> &'static str {
        self.provider_id
    }
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

    /// Construct an Image Generation model handle.
    ///
    /// `model_id` is the `OpenAI` image model name, e.g. `"dall-e-3"`,
    /// `"gpt-image-1"`, or `"chatgpt-image-latest"`.
    #[must_use]
    pub fn image(&self, model_id: impl Into<String>) -> OpenAiImageModel {
        OpenAiImageModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Construct a Responses API language model handle (`POST /v1/responses`).
    ///
    /// `model_id` is any model that accepts the Responses endpoint
    /// (gpt-4o / gpt-4.1 / gpt-5 / o-series). The same `OpenAi` provider
    /// can mix [`OpenAi::chat`] and [`OpenAi::responses`] handles; they
    /// route to different `OpenAI` endpoints.
    #[must_use]
    pub fn responses(&self, model_id: impl Into<String>) -> OpenAiResponsesLanguageModel {
        OpenAiResponsesLanguageModel::new(Arc::clone(&self.inner), model_id.into())
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
                url: UrlStrategy::Standard { base_url },
                headers,
                http,
                provider_id: PROVIDER_ID,
            }),
        })
    }
}
