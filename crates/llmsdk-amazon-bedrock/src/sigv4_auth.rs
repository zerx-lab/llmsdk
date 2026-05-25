//! Per-request `SigV4` signer that plugs into shared infrastructure.
//!
//! Mirrors `amazon-bedrock-sigv4-fetch.ts`'s `createSigV4FetchFunction`. The
//! sign step is delegated to [`llmsdk_provider_utils::aws_sigv4::sign_request`];
//! this module wraps it in the small set of shapes the rest of the crate
//! needs (`RequestAuth` for the Anthropic-on-Bedrock path; a free function
//! for the Converse / Embedding / Image / Rerank paths).
// Rust guideline compliant 2026-05-25

use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_anthropic::{RequestAuth, SignedHeaders, SigningContext};
use llmsdk_provider::ProviderError;
use llmsdk_provider_utils::aws_sigv4::{
    AwsCredentialsProvider, SignRequest, StaticCredentialsProvider, sign_request,
};
use reqwest::Method;

/// Authentication scheme used by the Bedrock provider.
#[derive(Clone)]
pub(crate) enum BedrockAuth {
    /// Sign every request with `AWS_BEARER_TOKEN_BEDROCK`. Used when an API
    /// key is supplied (matches upstream's `createApiKeyFetchFunction`).
    BearerToken(String),
    /// Compute `SigV4` signatures per request.
    SigV4 {
        /// Provider responsible for resolving fresh credentials.
        credentials: Arc<dyn AwsCredentialsProvider>,
        /// AWS region used as part of the signing scope.
        region: String,
        /// AWS service name (`"bedrock"` for runtime calls).
        service: String,
    },
}

impl std::fmt::Debug for BedrockAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BearerToken(_) => f.debug_struct("BearerToken").finish(),
            Self::SigV4 {
                region, service, ..
            } => f
                .debug_struct("SigV4")
                .field("region", region)
                .field("service", service)
                .field("credentials", &"<dyn AwsCredentialsProvider>")
                .finish(),
        }
    }
}

impl BedrockAuth {
    /// Build a `SigV4` auth from a static access key pair.
    pub(crate) fn sigv4_static(
        access_key_id: String,
        secret_access_key: String,
        session_token: Option<String>,
        region: String,
        service: String,
    ) -> Self {
        let mut creds =
            llmsdk_provider_utils::aws_sigv4::AwsCredentials::new(access_key_id, secret_access_key);
        if let Some(token) = session_token {
            creds = creds.with_session_token(token);
        }
        Self::SigV4 {
            credentials: Arc::new(StaticCredentialsProvider::new(creds)),
            region,
            service,
        }
    }

    /// Build a `SigV4` auth from a caller-supplied credentials provider.
    pub(crate) fn sigv4_provider(
        provider: Arc<dyn AwsCredentialsProvider>,
        region: String,
        service: String,
    ) -> Self {
        Self::SigV4 {
            credentials: provider,
            region,
            service,
        }
    }

    /// Append authentication headers to the given map.
    ///
    /// `body` may be empty for bodyless requests. Returns once headers have
    /// been mutated; on failure the original map is left untouched.
    pub(crate) async fn apply(
        &self,
        headers: &mut std::collections::HashMap<String, Option<String>>,
        method: &Method,
        url: &str,
        body: &[u8],
    ) -> Result<(), ProviderError> {
        match self {
            Self::BearerToken(token) => {
                headers.insert("Authorization".into(), Some(format!("Bearer {token}")));
                Ok(())
            }
            Self::SigV4 {
                credentials,
                region,
                service,
            } => {
                let creds = credentials.get_credentials().await?;
                let parsed = reqwest::Url::parse(url).map_err(|e| {
                    ProviderError::api_call_builder(url, format!("invalid URL: {e}")).build()
                })?;
                let host = parsed
                    .host_str()
                    .ok_or_else(|| {
                        ProviderError::api_call_builder(url, "URL has no host component").build()
                    })?
                    .to_owned();
                let pre_signed: Vec<(&str, &str)> = vec![
                    ("host", host.as_str()),
                    ("content-type", "application/json"),
                ];
                let signed = sign_request(&SignRequest {
                    method,
                    url,
                    headers: &pre_signed,
                    body,
                    credentials: &creds,
                    region,
                    service,
                    signing_time: None,
                })?;
                for (name, value) in signed {
                    let v = value
                        .to_str()
                        .map_err(|e| {
                            ProviderError::api_call_builder(
                                url,
                                format!("signed header value is not ASCII: {e}"),
                            )
                            .build()
                        })?
                        .to_owned();
                    headers.insert(name.as_str().to_owned(), Some(v));
                }
                Ok(())
            }
        }
    }
}

/// Adapter exposing a [`BedrockAuth`] through the [`RequestAuth`] trait the
/// Anthropic provider expects. Used by the Anthropic-on-Bedrock path.
#[derive(Debug)]
pub(crate) struct AnthropicAuthAdapter {
    pub(crate) auth: BedrockAuth,
}

#[async_trait]
impl RequestAuth for AnthropicAuthAdapter {
    async fn sign(&self, context: &SigningContext<'_>) -> Result<SignedHeaders, ProviderError> {
        let method = Method::from_bytes(context.method.as_bytes()).map_err(|e| {
            ProviderError::api_call_builder(
                context.url,
                format!("invalid HTTP method `{}`: {e}", context.method),
            )
            .build()
        })?;
        let mut headers: std::collections::HashMap<String, Option<String>> =
            std::collections::HashMap::new();
        self.auth
            .apply(&mut headers, &method, context.url, context.body)
            .await?;
        let signed: SignedHeaders = headers
            .into_iter()
            .filter_map(|(k, v)| v.map(|val| (k, val)))
            .collect();
        Ok(signed)
    }
}
