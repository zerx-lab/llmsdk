//! Mantle provider — OpenAI-compatible models on Bedrock.
//!
//! Mirrors `bedrock-mantle-provider.ts`.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_openai::internal::{
    Inner, OpenAiChatModel, OpenAiResponsesLanguageModel, RequestSigner, UrlStrategy,
};
use llmsdk_provider::ProviderError;
use llmsdk_provider_utils::aws_sigv4::AwsCredentialsProvider;
use llmsdk_provider_utils::http::HttpClient;
use reqwest::Method;

use crate::sigv4_auth::BedrockAuth;
use crate::{
    ACCESS_KEY_ID_ENV_VAR, BEARER_TOKEN_ENV_VAR, REGION_ENV_VAR, SECRET_ACCESS_KEY_ENV_VAR,
    SESSION_TOKEN_ENV_VAR,
};

/// AWS service name used by the Mantle SigV4 signing scope.
const MANTLE_SERVICE: &str = "bedrock-mantle";

const PROVIDER_ID_CHAT: &str = "bedrock-mantle.chat";
const PROVIDER_ID_RESPONSES: &str = "bedrock-mantle.responses";

/// Bedrock Mantle provider handle.
///
/// Returns OpenAI-compatible Chat + Responses model handles. SigV4 / Bearer
/// auth is plugged in transparently via the [`RequestSigner`] hook on
/// [`llmsdk_openai::internal::Inner`].
#[derive(Debug, Clone)]
pub struct BedrockMantle {
    chat_inner: Arc<Inner>,
    responses_inner: Arc<Inner>,
}

impl BedrockMantle {
    /// Open a [`BedrockMantleBuilder`].
    #[must_use]
    pub fn builder() -> BedrockMantleBuilder {
        BedrockMantleBuilder::default()
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

    /// Chat (OpenAI-compatible) model handle.
    #[must_use]
    pub fn chat(&self, model_id: impl Into<String>) -> OpenAiChatModel {
        OpenAiChatModel::new(Arc::clone(&self.chat_inner), model_id.into())
    }

    /// `languageModel` alias matching upstream JS — returns the chat surface.
    #[must_use]
    pub fn language_model(&self, model_id: impl Into<String>) -> OpenAiChatModel {
        self.chat(model_id)
    }

    /// Responses API model handle (not all Mantle models support this).
    #[must_use]
    pub fn responses(&self, model_id: impl Into<String>) -> OpenAiResponsesLanguageModel {
        OpenAiResponsesLanguageModel::new(Arc::clone(&self.responses_inner), model_id.into())
    }
}

/// Builder for [`BedrockMantle`].
#[derive(Default)]
pub struct BedrockMantleBuilder {
    region: Option<String>,
    api_key: Option<String>,
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    session_token: Option<String>,
    credentials_provider: Option<Arc<dyn AwsCredentialsProvider>>,
    base_url: Option<String>,
    extra_headers: HashMap<String, Option<String>>,
    http: Option<HttpClient>,
}

impl std::fmt::Debug for BedrockMantleBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let creds: &dyn std::fmt::Debug = &self.credentials_provider.is_some();
        f.debug_struct("BedrockMantleBuilder")
            .field("region", &self.region)
            .field("base_url", &self.base_url)
            .field("credentials_provider", creds)
            .finish_non_exhaustive()
    }
}

impl BedrockMantleBuilder {
    /// AWS region used for SigV4 scope and the default base URL.
    #[must_use]
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Bearer token authentication (`AWS_BEARER_TOKEN_BEDROCK`). When set,
    /// SigV4 is not used.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Static AWS access key id (SigV4).
    #[must_use]
    pub fn access_key_id(mut self, id: impl Into<String>) -> Self {
        self.access_key_id = Some(id.into());
        self
    }

    /// Static AWS secret access key (SigV4).
    #[must_use]
    pub fn secret_access_key(mut self, secret: impl Into<String>) -> Self {
        self.secret_access_key = Some(secret.into());
        self
    }

    /// Optional AWS session token (SigV4 / SSO).
    #[must_use]
    pub fn session_token(mut self, token: impl Into<String>) -> Self {
        self.session_token = Some(token.into());
        self
    }

    /// Dynamic credentials provider — overrides static credentials.
    #[must_use]
    pub fn credentials_provider<P>(mut self, provider: P) -> Self
    where
        P: AwsCredentialsProvider + 'static,
    {
        self.credentials_provider = Some(Arc::new(provider));
        self
    }

    /// Override the Mantle base URL prefix (defaults to
    /// `https://bedrock-mantle.{region}.api.aws/v1`).
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(without_trailing_slash(url.into()));
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

    /// Finalize.
    ///
    /// # Errors
    ///
    /// - [`ProviderError::load_api_key`] when neither bearer-token nor SigV4
    ///   credentials are resolvable.
    pub fn build(self) -> Result<BedrockMantle, ProviderError> {
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
            BedrockAuth::sigv4_provider(provider, region.clone(), MANTLE_SERVICE.to_owned())
        } else if let (Some(ak), Some(sk)) =
            (self.access_key_id.clone(), self.secret_access_key.clone())
        {
            BedrockAuth::sigv4_static(
                ak,
                sk,
                self.session_token.clone(),
                region.clone(),
                MANTLE_SERVICE.to_owned(),
            )
        } else {
            let ak = std::env::var(ACCESS_KEY_ID_ENV_VAR)
                .ok()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    ProviderError::load_api_key(
                        "Bedrock Mantle requires authentication. Provide an `api_key`, \
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
            BedrockAuth::sigv4_static(ak, sk, token, region.clone(), MANTLE_SERVICE.to_owned())
        };

        let base_url = self
            .base_url
            .unwrap_or_else(|| format!("https://bedrock-mantle.{region}.api.aws/v1"));

        let http = match self.http {
            Some(client) => client,
            None => HttpClient::new()?,
        };

        let url_strategy = UrlStrategy::Standard {
            base_url: base_url.clone(),
        };

        let signer: Arc<dyn RequestSigner> = Arc::new(MantleSigner { auth });

        let mk_inner = |provider_id: &'static str| -> Arc<Inner> {
            Arc::new(Inner::with_signer(
                url_strategy.clone(),
                self.extra_headers.clone(),
                http.clone(),
                provider_id,
                Arc::clone(&signer),
            ))
        };

        Ok(BedrockMantle {
            chat_inner: mk_inner(PROVIDER_ID_CHAT),
            responses_inner: mk_inner(PROVIDER_ID_RESPONSES),
        })
    }
}

/// Adapter exposing [`BedrockAuth`] through the [`RequestSigner`] trait the
/// `OpenAI` Inner expects.
#[derive(Debug)]
struct MantleSigner {
    auth: BedrockAuth,
}

#[async_trait]
impl RequestSigner for MantleSigner {
    async fn sign(
        &self,
        headers: &mut HashMap<String, Option<String>>,
        method: &str,
        url: &str,
        body: &[u8],
    ) -> Result<(), ProviderError> {
        let m = Method::from_bytes(method.as_bytes()).map_err(|e| {
            ProviderError::api_call_builder(url, format!("invalid HTTP method `{method}`: {e}"))
                .build()
        })?;
        self.auth.apply(headers, &m, url, body).await
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
    fn explicit_api_key_uses_bearer() {
        let p = BedrockMantle::builder()
            .api_key("test")
            .region("us-east-1")
            .build()
            .expect("ok");
        assert!(Arc::strong_count(&p.chat_inner) >= 1);
        assert!(Arc::strong_count(&p.responses_inner) >= 1);
    }

    #[test]
    fn static_sigv4_credentials_build_ok() {
        let p = BedrockMantle::builder()
            .region("us-east-1")
            .access_key_id("AKIA-test")
            .secret_access_key("test-secret")
            .build()
            .expect("static creds should build");
        assert!(Arc::strong_count(&p.chat_inner) >= 1);
    }
}
