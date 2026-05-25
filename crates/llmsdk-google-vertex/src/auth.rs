//! GCP OAuth token providers for Vertex.
//!
//! Mirrors `google-vertex-auth-google-auth-library.ts`. Defines an
//! [`AccessTokenProvider`] trait so callers can plug in alternative
//! implementations (mocked tokens for tests, custom token caches, ...)
//! and a default [`GcpAuthTokenProvider`] backed by the [`gcp_auth`]
//! crate (service-account JSON, GCE metadata server, gcloud CLI ADC).
// Rust guideline compliant 2026-05-25

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use tokio::sync::OnceCell;

use crate::CLOUD_PLATFORM_SCOPE;

/// Pluggable access-token source for Vertex AI requests.
///
/// Implement this trait if you need a custom token cache, want to mock
/// tokens during tests, or speak to a different identity provider than
/// GCP. The default implementation is [`GcpAuthTokenProvider`].
#[async_trait]
pub trait AccessTokenProvider: Send + Sync + fmt::Debug {
    /// Mint or fetch a cached OAuth access token for the given `scope`.
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError`] when the underlying identity provider
    /// fails (missing ADC, expired credentials, network failure).
    async fn token(&self, scope: &str) -> Result<String, ProviderError>;
}

/// Default [`AccessTokenProvider`] backed by [`gcp_auth`].
///
/// Lazily resolves a `gcp_auth::TokenProvider` on first use; subsequent
/// calls reuse the cached provider (which performs its own access-token
/// caching internally).
#[derive(Clone)]
pub struct GcpAuthTokenProvider {
    inner: Arc<OnceCell<Arc<dyn gcp_auth::TokenProvider>>>,
}

impl Default for GcpAuthTokenProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl GcpAuthTokenProvider {
    /// Build a fresh provider. The `gcp_auth` machinery only kicks in on
    /// the first [`token`](Self::token) call.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(OnceCell::new()),
        }
    }
}

impl fmt::Debug for GcpAuthTokenProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GcpAuthTokenProvider").finish()
    }
}

#[async_trait]
impl AccessTokenProvider for GcpAuthTokenProvider {
    async fn token(&self, scope: &str) -> Result<String, ProviderError> {
        let provider = self
            .inner
            .get_or_try_init(|| async {
                gcp_auth::provider().await.map_err(|e| {
                    ProviderError::load_api_key(format!(
                        "failed to initialize gcp_auth provider: {e}"
                    ))
                })
            })
            .await?;
        let token = provider.token(&[scope]).await.map_err(|e| {
            ProviderError::load_api_key(format!("gcp_auth failed to mint token: {e}"))
        })?;
        Ok(token.as_str().to_owned())
    }
}

/// Convenience: mint a token using the default scope
/// ([`CLOUD_PLATFORM_SCOPE`]).
///
/// # Errors
///
/// Forwards [`AccessTokenProvider::token`] failures.
pub async fn cloud_platform_token(
    provider: &dyn AccessTokenProvider,
) -> Result<String, ProviderError> {
    provider.token(CLOUD_PLATFORM_SCOPE).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default, Clone)]
    struct StaticToken {
        token: String,
    }

    #[async_trait]
    impl AccessTokenProvider for StaticToken {
        async fn token(&self, _scope: &str) -> Result<String, ProviderError> {
            Ok(self.token.clone())
        }
    }

    #[tokio::test]
    async fn static_token_returns_value() {
        let p = StaticToken {
            token: "tok".into(),
        };
        let v = cloud_platform_token(&p).await.expect("ok");
        assert_eq!(v, "tok");
    }
}
