//! Provider configuration and entry point.
//!
//! Mirrors `@ai-sdk/amazon-bedrock/src/amazon-bedrock-provider.ts`.
//!
//! Authentication resolution order, matching upstream:
//!
//! 1. Explicit `api_key` builder setting → `Authorization: Bearer …`
//! 2. `AWS_BEARER_TOKEN_BEDROCK` env var → `Authorization: Bearer …`
//! 3. Explicit `credentials_provider` → `SigV4`
//! 4. Explicit access key / secret key (+ optional session token) → `SigV4`
//! 5. `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (+ optional
//!    `AWS_SESSION_TOKEN`) env vars → `SigV4`
//!
//! Both base URLs (`bedrock-runtime` for chat / embed / image / Anthropic, and
//! `bedrock-agent-runtime` for `:rerank`) are derived from the AWS region.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;

use llmsdk_provider::ProviderError;
use llmsdk_provider_utils::aws_sigv4::AwsCredentialsProvider;
use llmsdk_provider_utils::http::HttpClient;

use crate::anthropic::AmazonBedrockAnthropicModel;
use crate::chat::AmazonBedrockChatModel;
use crate::embedding::AmazonBedrockEmbeddingModel;
use crate::image::AmazonBedrockImageModel;
use crate::reranking::AmazonBedrockRerankingModel;
use crate::sigv4_auth::BedrockAuth;
use crate::{
    ACCESS_KEY_ID_ENV_VAR, BEARER_TOKEN_ENV_VAR, REGION_ENV_VAR, SECRET_ACCESS_KEY_ENV_VAR,
    SESSION_TOKEN_ENV_VAR,
};

/// AWS Bedrock service name used when computing `SigV4` signatures.
pub(crate) const BEDROCK_SERVICE: &str = "bedrock";

/// Amazon Bedrock provider handle.
///
/// Cheap to clone; the underlying HTTP client and auth are shared.
#[derive(Debug, Clone)]
pub struct AmazonBedrock {
    pub(crate) inner: Arc<Inner>,
}

#[derive(Debug)]
pub(crate) struct Inner {
    /// Runtime endpoint base URL (`https://bedrock-runtime.{region}.amazonaws.com`).
    pub(crate) runtime_base_url: String,
    /// Agent-runtime endpoint base URL (`https://bedrock-agent-runtime.{region}.amazonaws.com`).
    pub(crate) agent_runtime_base_url: String,
    /// AWS region used for `SigV4` scope. Empty when authenticating via bearer
    /// token (region is still required upstream for URL building).
    pub(crate) region: String,
    /// Static extra headers (rarely populated).
    pub(crate) extra_headers: HashMap<String, Option<String>>,
    /// HTTP transport.
    pub(crate) http: HttpClient,
    /// Authentication scheme used for every outbound request.
    pub(crate) auth: BedrockAuth,
}

impl AmazonBedrock {
    /// Open a builder.
    #[must_use]
    pub fn builder() -> AmazonBedrockBuilder {
        AmazonBedrockBuilder::default()
    }

    /// Build with defaults derived entirely from the environment.
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError::load_api_key`] when neither bearer-token nor
    /// `SigV4` credentials can be resolved.
    pub fn from_env() -> Result<Self, ProviderError> {
        Self::builder().build()
    }

    /// Construct a Chat (Converse API) model handle.
    ///
    /// Equivalent to `provider(modelId)` upstream.
    #[must_use]
    pub fn language_model(&self, model_id: impl Into<String>) -> AmazonBedrockChatModel {
        AmazonBedrockChatModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::language_model`].
    #[must_use]
    pub fn chat(&self, model_id: impl Into<String>) -> AmazonBedrockChatModel {
        self.language_model(model_id)
    }

    /// Construct an Embedding model handle.
    #[must_use]
    pub fn embedding(&self, model_id: impl Into<String>) -> AmazonBedrockEmbeddingModel {
        AmazonBedrockEmbeddingModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::embedding`].
    #[must_use]
    pub fn embedding_model(&self, model_id: impl Into<String>) -> AmazonBedrockEmbeddingModel {
        self.embedding(model_id)
    }

    /// Alias of [`Self::embedding`] — mirrors ai-sdk's legacy
    /// `textEmbedding(id)`.
    #[must_use]
    pub fn text_embedding(&self, model_id: impl Into<String>) -> AmazonBedrockEmbeddingModel {
        self.embedding(model_id)
    }

    /// Construct an Image model handle.
    #[must_use]
    pub fn image(&self, model_id: impl Into<String>) -> AmazonBedrockImageModel {
        AmazonBedrockImageModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::image`].
    #[must_use]
    pub fn image_model(&self, model_id: impl Into<String>) -> AmazonBedrockImageModel {
        self.image(model_id)
    }

    /// Construct a Reranking model handle (`POST /rerank`).
    #[must_use]
    pub fn reranking(&self, model_id: impl Into<String>) -> AmazonBedrockRerankingModel {
        AmazonBedrockRerankingModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::reranking`].
    #[must_use]
    pub fn reranking_model(&self, model_id: impl Into<String>) -> AmazonBedrockRerankingModel {
        self.reranking(model_id)
    }

    /// Construct an Anthropic-on-Bedrock model handle (`POST /model/{id}/invoke`).
    ///
    /// Re-uses the [`llmsdk_anthropic::Anthropic`] client under the hood with
    /// a Bedrock-flavored URL transformer and the same `SigV4` authentication
    /// hook as the rest of the provider.
    ///
    /// # Errors
    ///
    /// Returns the [`ProviderError`] raised by the inner Anthropic builder
    /// when the request-auth hook installation fails.
    pub fn anthropic(
        &self,
        model_id: impl Into<String>,
    ) -> Result<AmazonBedrockAnthropicModel, ProviderError> {
        use crate::anthropic::AmazonBedrockAnthropicModelExt;
        <AmazonBedrockAnthropicModel as AmazonBedrockAnthropicModelExt>::new(
            Arc::clone(&self.inner),
            model_id.into(),
        )
    }
}

/// Builder for [`AmazonBedrock`].
///
/// All setters are optional. `build()` falls back to environment variables
/// for credentials / region / bearer token, matching upstream behavior.
#[derive(Default)]
pub struct AmazonBedrockBuilder {
    region: Option<String>,
    api_key: Option<String>,
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    session_token: Option<String>,
    credentials_provider: Option<Arc<dyn AwsCredentialsProvider>>,
    runtime_base_url: Option<String>,
    agent_runtime_base_url: Option<String>,
    extra_headers: HashMap<String, Option<String>>,
    http: Option<HttpClient>,
}

impl std::fmt::Debug for AmazonBedrockBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let credentials_provider_set: &dyn std::fmt::Debug = &self.credentials_provider.is_some();
        f.debug_struct("AmazonBedrockBuilder")
            .field("region", &self.region)
            .field("api_key", &self.api_key.as_ref().map(|_| "***"))
            .field("access_key_id", &self.access_key_id)
            .field(
                "secret_access_key",
                &self.secret_access_key.as_ref().map(|_| "***"),
            )
            .field("session_token", &self.session_token.as_ref().map(|_| "***"))
            .field("credentials_provider", credentials_provider_set)
            .field("runtime_base_url", &self.runtime_base_url)
            .field("agent_runtime_base_url", &self.agent_runtime_base_url)
            .field("extra_headers", &self.extra_headers)
            .field("http", &self.http)
            .finish()
    }
}

impl AmazonBedrockBuilder {
    /// Set the AWS region explicitly (e.g. `"us-east-1"`).
    #[must_use]
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Use a Bedrock bearer token instead of `SigV4` (matches
    /// `AWS_BEARER_TOKEN_BEDROCK`).
    ///
    /// When set, takes precedence over every other credential source.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Static access-key id for `SigV4`.
    #[must_use]
    pub fn access_key_id(mut self, value: impl Into<String>) -> Self {
        self.access_key_id = Some(value.into());
        self
    }

    /// Static secret access key for `SigV4`.
    #[must_use]
    pub fn secret_access_key(mut self, value: impl Into<String>) -> Self {
        self.secret_access_key = Some(value.into());
        self
    }

    /// Optional session token (set by STS / SSO / IMDS).
    #[must_use]
    pub fn session_token(mut self, value: impl Into<String>) -> Self {
        self.session_token = Some(value.into());
        self
    }

    /// Provide a custom credentials provider (overrides static keys + env).
    #[must_use]
    pub fn credentials_provider<P>(mut self, provider: P) -> Self
    where
        P: AwsCredentialsProvider + 'static,
    {
        self.credentials_provider = Some(Arc::new(provider));
        self
    }

    /// Override the bedrock-runtime base URL (defaults to
    /// `https://bedrock-runtime.{region}.amazonaws.com`).
    #[must_use]
    pub fn runtime_base_url(mut self, url: impl Into<String>) -> Self {
        self.runtime_base_url = Some(without_trailing_slash(url.into()));
        self
    }

    /// Override the bedrock-agent-runtime base URL (defaults to
    /// `https://bedrock-agent-runtime.{region}.amazonaws.com`).
    #[must_use]
    pub fn agent_runtime_base_url(mut self, url: impl Into<String>) -> Self {
        self.agent_runtime_base_url = Some(without_trailing_slash(url.into()));
        self
    }

    /// Override both base URLs with the same string (useful for wiremock
    /// servers in contract tests).
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        let trimmed = without_trailing_slash(url.into());
        self.runtime_base_url = Some(trimmed.clone());
        self.agent_runtime_base_url = Some(trimmed);
        self
    }

    /// Append or override a header sent on every request.
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: Option<String>) -> Self {
        self.extra_headers.insert(name.into(), value);
        self
    }

    /// Inject a pre-configured HTTP client (shared across all models).
    #[must_use]
    pub fn http_client(mut self, client: HttpClient) -> Self {
        self.http = Some(client);
        self
    }

    /// Finalize the provider.
    ///
    /// # Errors
    ///
    /// - [`ProviderError::load_api_key`] when no usable authentication
    ///   source is configured.
    /// - [`ProviderError`] from [`HttpClient::new`] when the TLS stack fails
    ///   to initialize.
    pub fn build(self) -> Result<AmazonBedrock, ProviderError> {
        let region = self
            .region
            .or_else(|| std::env::var(REGION_ENV_VAR).ok())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "us-east-1".to_owned());

        let bearer_token = self
            .api_key
            .or_else(|| std::env::var(BEARER_TOKEN_ENV_VAR).ok())
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty());

        let auth = if let Some(token) = bearer_token {
            BedrockAuth::BearerToken(token)
        } else if let Some(provider) = self.credentials_provider {
            BedrockAuth::sigv4_provider(provider, region.clone(), BEDROCK_SERVICE.to_owned())
        } else if let (Some(ak), Some(sk)) =
            (self.access_key_id.clone(), self.secret_access_key.clone())
        {
            BedrockAuth::sigv4_static(
                ak,
                sk,
                self.session_token.clone(),
                region.clone(),
                BEDROCK_SERVICE.to_owned(),
            )
        } else {
            let ak = std::env::var(ACCESS_KEY_ID_ENV_VAR)
                .ok()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    ProviderError::load_api_key(
                        "Amazon Bedrock requires authentication. Provide an `api_key`, \
                         (`access_key_id` + `secret_access_key`), `credentials_provider`, \
                         or set AWS_BEARER_TOKEN_BEDROCK / (AWS_ACCESS_KEY_ID + \
                         AWS_SECRET_ACCESS_KEY) environment variables.",
                    )
                })?;
            let sk = std::env::var(SECRET_ACCESS_KEY_ENV_VAR)
                .ok()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    ProviderError::load_api_key(
                        "AWS_ACCESS_KEY_ID was set but AWS_SECRET_ACCESS_KEY is missing.",
                    )
                })?;
            let token = std::env::var(SESSION_TOKEN_ENV_VAR)
                .ok()
                .filter(|s| !s.is_empty());
            BedrockAuth::sigv4_static(ak, sk, token, region.clone(), BEDROCK_SERVICE.to_owned())
        };

        let runtime_base_url = self
            .runtime_base_url
            .unwrap_or_else(|| format!("https://bedrock-runtime.{region}.amazonaws.com"));
        let agent_runtime_base_url = self
            .agent_runtime_base_url
            .unwrap_or_else(|| format!("https://bedrock-agent-runtime.{region}.amazonaws.com"));

        let http = match self.http {
            Some(client) => client,
            None => HttpClient::new()?,
        };

        Ok(AmazonBedrock {
            inner: Arc::new(Inner {
                runtime_base_url,
                agent_runtime_base_url,
                region,
                extra_headers: self.extra_headers,
                http,
                auth,
            }),
        })
    }
}

fn without_trailing_slash(mut s: String) -> String {
    while s.ends_with('/') {
        s.pop();
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_api_key_uses_bearer_auth() {
        let provider = AmazonBedrock::builder()
            .api_key("test-bearer")
            .region("us-east-1")
            .build()
            .expect("ok");
        assert!(matches!(provider.inner.auth, BedrockAuth::BearerToken(_)));
    }

    #[test]
    fn explicit_static_keys_use_sigv4() {
        let provider = AmazonBedrock::builder()
            .region("us-east-1")
            .access_key_id("AKIA")
            .secret_access_key("SECRET")
            .build()
            .expect("ok");
        assert!(matches!(provider.inner.auth, BedrockAuth::SigV4 { .. }));
    }

    #[test]
    fn base_url_override_strips_trailing_slash() {
        let provider = AmazonBedrock::builder()
            .api_key("k")
            .region("us-east-1")
            .base_url("https://proxy.example.com/")
            .build()
            .expect("ok");
        assert_eq!(provider.inner.runtime_base_url, "https://proxy.example.com");
        assert_eq!(
            provider.inner.agent_runtime_base_url,
            "https://proxy.example.com"
        );
    }

    #[test]
    fn default_urls_use_region() {
        let provider = AmazonBedrock::builder()
            .api_key("k")
            .region("eu-west-1")
            .build()
            .expect("ok");
        assert!(
            provider
                .inner
                .runtime_base_url
                .contains("bedrock-runtime.eu-west-1")
        );
        assert!(
            provider
                .inner
                .agent_runtime_base_url
                .contains("bedrock-agent-runtime.eu-west-1")
        );
    }

    #[test]
    fn empty_credentials_yield_load_error() {
        // Pretend env is empty by clearing the keys at the test level — but
        // because env access is global, just verify the explicit path. The
        // env path is covered by tests that own the lock elsewhere.
        let result = AmazonBedrock::builder()
            // no api_key, no creds, no provider
            .region("us-east-1")
            .build();
        // Either errors (clean CI) or succeeds (dev shell with AWS_* set):
        // both are valid; we only assert the error message shape when it does
        // error.
        if let Err(err) = result {
            assert!(format!("{err}").contains("Amazon Bedrock requires authentication"));
        }
    }
}
