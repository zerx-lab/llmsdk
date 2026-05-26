//! Provider configuration and entry point for Google Vertex AI.
//!
//! Mirrors `google-vertex-provider.ts` + `google-vertex-provider-base.ts`.
//! Splits the upstream "Node" and "Edge" entry points into a single Rust
//! builder that selects between the two authentication modes (OAuth vs
//! Express-mode API key) at `build()` time.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;

use llmsdk_provider::ProviderError;
use llmsdk_provider_utils::http::HttpClient;

use crate::anthropic::GoogleVertexAnthropic;
use crate::auth::{AccessTokenProvider, GcpAuthTokenProvider};
use crate::embedding::GoogleVertexEmbeddingModel;
use crate::image::GoogleVertexImageModel;
use crate::language::GoogleVertexLanguageModel;
use crate::maas::GoogleVertexMaas;
use crate::video::GoogleVertexVideoModel;
use crate::xai::GoogleVertexXai;
use crate::{
    API_KEY_ENV_VAR, DEFAULT_LOCATION, EXPRESS_MODE_BASE_URL, LOCATION_ENV_VAR, PROJECT_ENV_VAR,
};

/// Authentication mode selected at builder time.
///
/// Mirrors the upstream branching in `createGoogleVertex`: when an API
/// key is present (explicit or env), the Express base URL is used and
/// the `x-goog-api-key` header is sent; otherwise OAuth tokens are
/// minted via [`GcpAuthTokenProvider`] and sent as
/// `Authorization: Bearer ...`.
#[derive(Debug, Clone)]
pub enum VertexAuthMode {
    /// OAuth mode: per-request bearer token + regionalized URL.
    OAuth {
        /// Resolved GCP project id.
        project: String,
        /// Resolved Vertex location (e.g. `"us-central1"`, `"global"`).
        location: String,
        /// Pluggable access-token source.
        token_provider: Arc<dyn AccessTokenProvider>,
    },
    /// Express mode: API key + global publishers URL.
    Express {
        /// API key sent as `x-goog-api-key`.
        api_key: String,
    },
}

impl VertexAuthMode {
    /// Best-effort label used in provider-id suffixes and tracing.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::OAuth { .. } => "oauth",
            Self::Express { .. } => "express",
        }
    }
}

/// Google Vertex AI provider handle.
///
/// Cheap to clone; the underlying HTTP client + auth state are shared
/// via [`Arc`].
#[derive(Debug, Clone)]
pub struct GoogleVertex {
    pub(crate) inner: Arc<VertexInner>,
}

/// Shared provider state used by every model handle.
#[derive(Debug)]
pub(crate) struct VertexInner {
    pub(crate) auth: VertexAuthMode,
    pub(crate) http: HttpClient,
    pub(crate) extra_headers: HashMap<String, Option<String>>,
    pub(crate) anthropic_base_override: Option<String>,
    pub(crate) sub_base_override: Option<String>,
    pub(crate) language_base_override: Option<String>,
}

impl VertexInner {
    /// Compose the publishers-google base URL (used by language, embedding,
    /// image, video and tools) honouring the active auth mode + any caller
    /// override.
    pub(crate) fn publishers_google_base(&self) -> String {
        if let Some(base) = &self.language_base_override {
            return base.clone();
        }
        match &self.auth {
            VertexAuthMode::Express { .. } => EXPRESS_MODE_BASE_URL.to_owned(),
            VertexAuthMode::OAuth {
                project, location, ..
            } => standard_publishers_url(project, location, "google", GOOGLE_API_VERSION),
        }
    }

    /// Compose the publishers-anthropic base URL (used by Anthropic on
    /// Vertex).
    pub(crate) fn publishers_anthropic_base(&self) -> String {
        if let Some(base) = &self.anthropic_base_override {
            return base.clone();
        }
        match &self.auth {
            VertexAuthMode::Express { .. } => {
                "https://aiplatform.googleapis.com/v1/publishers/anthropic".to_owned()
            }
            VertexAuthMode::OAuth {
                project, location, ..
            } => standard_publishers_url(project, location, "anthropic", ANTHROPIC_API_VERSION),
        }
    }

    /// Compose the `endpoints/openapi` base URL used by OpenAI-compatible
    /// sub-providers (xAI, MaaS).
    pub(crate) fn sub_provider_base(&self) -> String {
        if let Some(base) = &self.sub_base_override {
            return base.clone();
        }
        match &self.auth {
            VertexAuthMode::Express { .. } => {
                "https://aiplatform.googleapis.com/v1/endpoints/openapi".to_owned()
            }
            VertexAuthMode::OAuth {
                project, location, ..
            } => {
                let region = if location == "global" {
                    String::new()
                } else {
                    format!("{location}-")
                };
                format!(
                    "https://{region}aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi"
                )
            }
        }
    }
}

/// Vertex Gemini publishers path uses the `v1beta1` Google API surface
/// (mirrors ai-sdk `google-vertex-provider-base.ts:171`). Required for
/// preview-tier features like `cachedContent`, `thinkingConfig`, and
/// `responseModalities`; `v1` rejects these fields.
const GOOGLE_API_VERSION: &str = "v1beta1";

/// Anthropic-on-Vertex publishers path stays on the stable `v1` API
/// surface (mirrors ai-sdk `anthropic/google-vertex-anthropic-provider.ts:192`).
const ANTHROPIC_API_VERSION: &str = "v1";

fn standard_publishers_url(
    project: &str,
    location: &str,
    publisher: &str,
    api_version: &str,
) -> String {
    let region = if location == "global" {
        String::new()
    } else {
        format!("{location}-")
    };
    format!(
        "https://{region}aiplatform.googleapis.com/{api_version}/projects/{project}/locations/{location}/publishers/{publisher}"
    )
}

impl GoogleVertex {
    /// Open a [`GoogleVertexBuilder`].
    #[must_use]
    pub fn builder() -> GoogleVertexBuilder {
        GoogleVertexBuilder::default()
    }

    /// Build with defaults: env-driven project/location/api-key.
    ///
    /// Calls [`GoogleVertexBuilder::build`] internally.
    ///
    /// # Errors
    ///
    /// See [`GoogleVertexBuilder::build`].
    pub async fn from_env() -> Result<Self, ProviderError> {
        Self::builder().build().await
    }

    /// Borrow the active authentication mode (telemetry helper).
    #[must_use]
    pub fn auth_mode(&self) -> &VertexAuthMode {
        &self.inner.auth
    }

    /// Borrow extra headers configured at builder time.
    #[must_use]
    pub fn extra_headers(&self) -> &HashMap<String, Option<String>> {
        &self.inner.extra_headers
    }

    /// Construct a Gemini language-model handle.
    #[must_use]
    pub fn language_model(&self, model_id: impl Into<String>) -> GoogleVertexLanguageModel {
        GoogleVertexLanguageModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::language_model`].
    #[must_use]
    pub fn chat(&self, model_id: impl Into<String>) -> GoogleVertexLanguageModel {
        self.language_model(model_id)
    }

    /// Construct a Vertex embedding-model handle (Vertex-native
    /// `instances[]` + `parameters` wire).
    #[must_use]
    pub fn embedding(&self, model_id: impl Into<String>) -> GoogleVertexEmbeddingModel {
        GoogleVertexEmbeddingModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::embedding`].
    #[must_use]
    pub fn embedding_model(&self, model_id: impl Into<String>) -> GoogleVertexEmbeddingModel {
        self.embedding(model_id)
    }

    /// Alias of [`Self::embedding`] — mirrors ai-sdk's deprecated
    /// `textEmbeddingModel(id)`.
    #[must_use]
    pub fn text_embedding_model(&self, model_id: impl Into<String>) -> GoogleVertexEmbeddingModel {
        self.embedding(model_id)
    }

    /// Construct an image-generation model handle (Imagen via `:predict`;
    /// Gemini image models delegate to the language model with
    /// `responseModalities=IMAGE`).
    #[must_use]
    pub fn image(&self, model_id: impl Into<String>) -> GoogleVertexImageModel {
        GoogleVertexImageModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::image`].
    #[must_use]
    pub fn image_model(&self, model_id: impl Into<String>) -> GoogleVertexImageModel {
        self.image(model_id)
    }

    /// Construct a video-generation model handle (Veo on Vertex,
    /// `:predictLongRunning` + `:fetchPredictOperation` polling).
    #[must_use]
    pub fn video(&self, model_id: impl Into<String>) -> GoogleVertexVideoModel {
        GoogleVertexVideoModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::video`].
    #[must_use]
    pub fn video_model(&self, model_id: impl Into<String>) -> GoogleVertexVideoModel {
        self.video(model_id)
    }

    /// Construct an Anthropic-on-Vertex sub-provider handle.
    #[must_use]
    pub fn anthropic(&self) -> GoogleVertexAnthropic {
        GoogleVertexAnthropic::new(Arc::clone(&self.inner))
    }

    /// Construct an xAI-on-Vertex sub-provider handle.
    #[must_use]
    pub fn xai(&self) -> GoogleVertexXai {
        GoogleVertexXai::new(Arc::clone(&self.inner))
    }

    /// Construct a MaaS-on-Vertex sub-provider handle (OpenAI-compatible
    /// partner / open models like DeepSeek / Llama / Mistral).
    #[must_use]
    pub fn maas(&self) -> GoogleVertexMaas {
        GoogleVertexMaas::new(Arc::clone(&self.inner))
    }
}

/// Builder for [`GoogleVertex`].
///
/// At least one of [`api_key`](Self::api_key) (which selects Express
/// mode) **or** [`project`](Self::project) (which selects OAuth mode)
/// must resolve at build time; otherwise [`build`](Self::build) returns
/// [`ProviderError::invalid_argument`].
#[derive(Default)]
pub struct GoogleVertexBuilder {
    project: Option<String>,
    location: Option<String>,
    api_key: Option<String>,
    extra_headers: HashMap<String, Option<String>>,
    http: Option<HttpClient>,
    token_provider: Option<Arc<dyn AccessTokenProvider>>,
    language_base_override: Option<String>,
    anthropic_base_override: Option<String>,
    sub_base_override: Option<String>,
}

impl std::fmt::Debug for GoogleVertexBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoogleVertexBuilder")
            .field("project", &self.project)
            .field("location", &self.location)
            .field("api_key", &self.api_key.as_ref().map(|_| "<set>"))
            .field("extra_headers", &self.extra_headers)
            .field("http", &self.http.as_ref().map(|_| "<set>"))
            .field(
                "token_provider",
                &self.token_provider.as_ref().map(|_| "<set>"),
            )
            .field("language_base_override", &self.language_base_override)
            .field("anthropic_base_override", &self.anthropic_base_override)
            .field("sub_base_override", &self.sub_base_override)
            .finish()
    }
}

impl GoogleVertexBuilder {
    /// Set the GCP project id explicitly (OAuth mode).
    ///
    /// Falls back to [`PROJECT_ENV_VAR`] when not given.
    #[must_use]
    pub fn project(mut self, value: impl Into<String>) -> Self {
        self.project = Some(value.into());
        self
    }

    /// Set the Vertex location / region explicitly (OAuth mode).
    ///
    /// Falls back to [`LOCATION_ENV_VAR`] (or [`DEFAULT_LOCATION`] when
    /// that is also unset). `"global"` is accepted and triggers the
    /// region-less host `aiplatform.googleapis.com`.
    #[must_use]
    pub fn location(mut self, value: impl Into<String>) -> Self {
        self.location = Some(value.into());
        self
    }

    /// Activate Express mode by passing an explicit API key.
    ///
    /// Falls back to [`API_KEY_ENV_VAR`] when not given.
    #[must_use]
    pub fn api_key(mut self, value: impl Into<String>) -> Self {
        self.api_key = Some(value.into());
        self
    }

    /// Plug in a custom [`AccessTokenProvider`]. Defaults to
    /// [`GcpAuthTokenProvider`].
    #[must_use]
    pub fn token_provider(mut self, provider: Arc<dyn AccessTokenProvider>) -> Self {
        self.token_provider = Some(provider);
        self
    }

    /// Use a pre-minted OAuth access token instead of binding `gcp_auth`.
    ///
    /// Mirrors the upstream Edge Runtime entry point: the caller obtains
    /// the bearer token out-of-band (e.g. via a fronted IAM service,
    /// a custom JWT signer, or a workflow that already holds the token)
    /// and passes it straight through. No `google-auth-library` /
    /// `gcp_auth` invocation is performed.
    ///
    /// Project + location must still be provided (explicit or env vars).
    #[must_use]
    pub fn access_token(self, token: impl Into<String>) -> Self {
        self.token_provider(Arc::new(crate::auth::StaticAccessTokenProvider::new(token)))
    }

    /// Append or override a per-request header. `None` removes it.
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

    /// Override the language / embedding / image / video publishers-google
    /// base URL. Primarily used by integration tests pointing at a
    /// `wiremock` server.
    #[must_use]
    pub fn language_base_url(mut self, url: impl Into<String>) -> Self {
        self.language_base_override = Some(url.into().trim_end_matches('/').to_owned());
        self
    }

    /// Override the Anthropic publishers base URL (testing escape hatch).
    #[must_use]
    pub fn anthropic_base_url(mut self, url: impl Into<String>) -> Self {
        self.anthropic_base_override = Some(url.into().trim_end_matches('/').to_owned());
        self
    }

    /// Override the `endpoints/openapi` base URL used by xAI and MaaS
    /// (testing escape hatch).
    #[must_use]
    pub fn sub_provider_base_url(mut self, url: impl Into<String>) -> Self {
        self.sub_base_override = Some(url.into().trim_end_matches('/').to_owned());
        self
    }

    /// Finalize the provider.
    ///
    /// Auth-mode selection:
    /// - Explicit [`api_key`](Self::api_key) **or** [`API_KEY_ENV_VAR`]
    ///   non-empty → Express mode.
    /// - Otherwise → OAuth mode; project must be resolvable from explicit
    ///   value or [`PROJECT_ENV_VAR`].
    ///
    /// `async` for future-proofing — current implementation is
    /// synchronous because the OAuth token provider defers token minting
    /// until the first per-call request, but consumers that pre-fetch a
    /// token to validate credentials at build time will need to await.
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError::invalid_argument`] when OAuth mode is
    /// requested without a resolvable project.
    #[allow(
        clippy::unused_async,
        reason = "kept async for forward compatibility with credentials-prefetch builders"
    )]
    pub async fn build(self) -> Result<GoogleVertex, ProviderError> {
        let resolved_key = self
            .api_key
            .clone()
            .or_else(|| std::env::var(API_KEY_ENV_VAR).ok())
            .filter(|s| !s.is_empty());

        let auth = if let Some(key) = resolved_key {
            VertexAuthMode::Express { api_key: key }
        } else {
            let project = self
                .project
                .clone()
                .or_else(|| std::env::var(PROJECT_ENV_VAR).ok())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    ProviderError::invalid_argument(
                        "project",
                        format!(
                            "Google Vertex project is missing. Pass it via `GoogleVertexBuilder::project` \
                             or the {PROJECT_ENV_VAR} environment variable, or supply an `api_key` to \
                             use Express mode."
                        ),
                    )
                })?;
            let location = self
                .location
                .clone()
                .or_else(|| std::env::var(LOCATION_ENV_VAR).ok())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| DEFAULT_LOCATION.to_owned());
            let token_provider: Arc<dyn AccessTokenProvider> = self
                .token_provider
                .unwrap_or_else(|| Arc::new(GcpAuthTokenProvider::new()));
            VertexAuthMode::OAuth {
                project,
                location,
                token_provider,
            }
        };

        let http = match self.http {
            Some(client) => client,
            None => HttpClient::new()?,
        };

        Ok(GoogleVertex {
            inner: Arc::new(VertexInner {
                auth,
                http,
                extra_headers: self.extra_headers,
                language_base_override: self.language_base_override,
                anthropic_base_override: self.anthropic_base_override,
                sub_base_override: self.sub_base_override,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn express_mode_uses_express_publishers_url() {
        let p = GoogleVertex::builder()
            .api_key("ek")
            .build()
            .await
            .expect("ok");
        assert_eq!(p.auth_mode().label(), "express");
        assert_eq!(
            p.inner.publishers_google_base(),
            "https://aiplatform.googleapis.com/v1/publishers/google"
        );
    }

    #[tokio::test]
    async fn oauth_mode_regionalizes_url() {
        let p = GoogleVertex::builder()
            .project("acme")
            .location("us-central1")
            .build()
            .await
            .expect("ok");
        assert_eq!(p.auth_mode().label(), "oauth");
        assert_eq!(
            p.inner.publishers_google_base(),
            "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/acme/locations/us-central1/publishers/google"
        );
    }

    #[tokio::test]
    async fn oauth_global_drops_region_prefix() {
        let p = GoogleVertex::builder()
            .project("acme")
            .location("global")
            .build()
            .await
            .expect("ok");
        assert_eq!(
            p.inner.publishers_google_base(),
            "https://aiplatform.googleapis.com/v1beta1/projects/acme/locations/global/publishers/google"
        );
    }

    #[tokio::test]
    async fn oauth_anthropic_base_routes_to_anthropic_publisher() {
        let p = GoogleVertex::builder()
            .project("acme")
            .location("europe-west1")
            .build()
            .await
            .expect("ok");
        assert_eq!(
            p.inner.publishers_anthropic_base(),
            "https://europe-west1-aiplatform.googleapis.com/v1/projects/acme/locations/europe-west1/publishers/anthropic"
        );
    }

    #[tokio::test]
    async fn oauth_sub_provider_base_uses_endpoints_openapi() {
        let p = GoogleVertex::builder()
            .project("acme")
            .location("us-central1")
            .build()
            .await
            .expect("ok");
        assert_eq!(
            p.inner.sub_provider_base(),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/acme/locations/us-central1/endpoints/openapi"
        );
    }

    #[tokio::test]
    async fn missing_project_and_key_is_invalid() {
        let result = GoogleVertex::builder().build().await;
        // Tolerate the case where GOOGLE_VERTEX_PROJECT happens to be set
        // in the dev environment.
        if std::env::var(PROJECT_ENV_VAR).is_err() && std::env::var(API_KEY_ENV_VAR).is_err() {
            assert!(result.is_err());
        }
    }

    #[tokio::test]
    async fn language_base_override_wins() {
        let p = GoogleVertex::builder()
            .api_key("k")
            .language_base_url("http://localhost:1234/v1/publishers/google")
            .build()
            .await
            .expect("ok");
        assert_eq!(
            p.inner.publishers_google_base(),
            "http://localhost:1234/v1/publishers/google"
        );
    }
}
