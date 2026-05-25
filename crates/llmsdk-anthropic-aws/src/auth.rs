//! [`RequestAuth`] implementations for Claude Platform on AWS.
//!
//! Mirrors `anthropic-aws-fetch.ts` (`createSigV4FetchFunction` +
//! `createApiKeyFetchFunction`). Both implementations append a single set
//! of headers per request and otherwise let the upstream
//! [`llmsdk_anthropic`] pipeline (headers, multipart, SSE) run unmodified.
// Rust guideline compliant 2026-02-21

use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use llmsdk_anthropic::{RequestAuth, SignedHeaders, SigningContext};
use llmsdk_provider::ProviderError;
use llmsdk_provider_utils::aws_sigv4::{AwsCredentialsProvider, sign_post};

use crate::SIGV4_SERVICE;

/// Authenticate every outbound request with an AWS-provisioned `x-api-key`.
///
/// The header value is cloned per-request and overrides any pre-existing
/// `x-api-key` header at the [`llmsdk_anthropic::AnthropicBuilder`] layer.
///
/// Mirrors `createApiKeyFetchFunction` from `anthropic-aws-fetch.ts`.
#[derive(Debug, Clone)]
pub struct ApiKeyAuth {
    api_key: String,
}

impl ApiKeyAuth {
    /// Construct from a key string.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
        }
    }
}

#[async_trait]
impl RequestAuth for ApiKeyAuth {
    async fn sign(&self, _context: &SigningContext<'_>) -> Result<SignedHeaders, ProviderError> {
        Ok(vec![("x-api-key".to_owned(), self.api_key.clone())])
    }
}

/// Authenticate every outbound POST with AWS Signature Version 4.
///
/// `GET` and bodyless requests bypass signing (matching `aws4fetch`'s
/// `createSigV4FetchFunction` behavior in `anthropic-aws-fetch.ts`): the AWS
/// gateway only requires `SigV4` on mutating operations and the upstream
/// `llmsdk-anthropic` Skills handle issues a `GET` for `versions/{v}`
/// metadata that the upstream JS implementation also lets pass through.
///
/// Credentials are resolved on every call via the supplied
/// [`AwsCredentialsProvider`] so callers can plug in IMDS / STS / refreshing
/// providers without rebuilding the [`crate::AnthropicAws`] instance.
pub struct SigV4Auth {
    credentials_provider: Arc<dyn AwsCredentialsProvider>,
    region: String,
}

impl std::fmt::Debug for SigV4Auth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let credentials_provider: &dyn std::fmt::Debug = &"<dyn AwsCredentialsProvider>";
        f.debug_struct("SigV4Auth")
            .field("credentials_provider", credentials_provider)
            .field("region", &self.region)
            .finish()
    }
}

impl SigV4Auth {
    /// Build from a region and a credentials provider.
    ///
    /// `region` is also the signing region; it must match the region used in
    /// the request URL (the upstream gateway rejects cross-region signatures).
    #[must_use]
    pub fn new(
        region: impl Into<String>,
        credentials_provider: Arc<dyn AwsCredentialsProvider>,
    ) -> Self {
        Self {
            credentials_provider,
            region: region.into(),
        }
    }

    /// Region this signer uses (e.g. `"us-west-2"`).
    #[must_use]
    pub fn region(&self) -> &str {
        &self.region
    }
}

#[async_trait]
impl RequestAuth for SigV4Auth {
    async fn sign(&self, context: &SigningContext<'_>) -> Result<SignedHeaders, ProviderError> {
        // Mirrors the early-return in `anthropic-aws-fetch.ts`: only sign
        // POST + body present. GET / bodyless POST flow through unchanged
        // because the AWS gateway accepts them unsigned (Files metadata
        // GET is the only such path llmsdk-anthropic emits today).
        if !context.method.eq_ignore_ascii_case("POST") || context.body.is_empty() {
            return Ok(Vec::new());
        }

        let credentials = self.credentials_provider.get_credentials().await?;
        // We don't know the on-wire content-type from inside the hook, so
        // detect it from the body bytes: a leading `{` or `[` indicates
        // JSON; otherwise the body is a `multipart/form-data` payload.
        // The value must match what `reqwest` actually sends (provider-utils
        // post_json sets `application/json`; post_raw forwards the caller's
        // string). llmsdk-anthropic only emits these two shapes today.
        let content_type = sniff_content_type(context.body);
        sign_post(
            context.url,
            context.body,
            &[("content-type", content_type)],
            &credentials,
            &self.region,
            SIGV4_SERVICE,
        )
    }
}

// SystemTime stays referenced for parity with the upstream JS, which
// passes "now" to AwsV4Signer. `sign_post` does the same internally.
const _: fn() = || {
    let _ = SystemTime::now;
};

/// Best-effort MIME sniffer: JSON if it parses as JSON, otherwise multipart.
///
/// llmsdk-anthropic emits exactly two content types on POST:
/// `application/json` (Messages) and `multipart/form-data` (Files /
/// Skills). For `SigV4` we only need the value that participates in the
/// canonical request — the receiving Claude Platform on AWS gateway is
/// flexible about which `content-type` participates as long as the same
/// value is on the wire, and reqwest sets the wire value from the
/// `RawRequest::content_type` / JSON serializer, both of which we forward
/// via `SignedHeaders` so the gateway accepts the signature.
fn sniff_content_type(body: &[u8]) -> &'static str {
    if body.first() == Some(&b'{') || body.first() == Some(&b'[') {
        "application/json"
    } else {
        "multipart/form-data"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider_utils::aws_sigv4::{AwsCredentials, StaticCredentialsProvider};

    #[tokio::test]
    async fn api_key_auth_emits_xapikey() {
        let auth = ApiKeyAuth::new("sk-test");
        let headers = auth
            .sign(&SigningContext {
                method: "POST",
                url: "https://example.com/messages",
                body: b"{}",
            })
            .await
            .unwrap();
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].0, "x-api-key");
        assert_eq!(headers[0].1, "sk-test");
    }

    #[tokio::test]
    async fn sigv4_auth_skips_get_and_empty_post() {
        let auth = SigV4Auth::new(
            "us-west-2",
            Arc::new(StaticCredentialsProvider::new(AwsCredentials::new(
                "ak", "sk",
            ))),
        );
        let get_headers = auth
            .sign(&SigningContext {
                method: "GET",
                url: "https://example.com/skills/abc/versions/1",
                body: &[],
            })
            .await
            .unwrap();
        assert!(get_headers.is_empty());

        let empty_post = auth
            .sign(&SigningContext {
                method: "POST",
                url: "https://example.com/messages",
                body: &[],
            })
            .await
            .unwrap();
        assert!(empty_post.is_empty());
    }

    #[tokio::test]
    async fn sigv4_auth_signs_post_with_body() {
        let auth = SigV4Auth::new(
            "us-east-1",
            Arc::new(StaticCredentialsProvider::new(AwsCredentials::new(
                "AKIDEXAMPLE",
                "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            ))),
        );
        let headers = auth
            .sign(&SigningContext {
                method: "POST",
                url: "https://aws-external-anthropic.us-east-1.api.aws/v1/messages",
                body: br#"{"hello":"world"}"#,
            })
            .await
            .unwrap();
        let names: Vec<_> = headers.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"authorization"), "got: {names:?}");
        assert!(names.contains(&"x-amz-date"), "got: {names:?}");
        let auth_value = headers
            .iter()
            .find(|(n, _)| n == "authorization")
            .map(|(_, v)| v.as_str())
            .unwrap();
        assert!(
            auth_value.contains("aws-external-anthropic/aws4_request"),
            "got: {auth_value}"
        );
    }

    #[tokio::test]
    async fn sigv4_auth_includes_session_token_when_present() {
        let auth = SigV4Auth::new(
            "us-west-2",
            Arc::new(StaticCredentialsProvider::new(
                AwsCredentials::new("AKID", "SECRET").with_session_token("sess-tok"),
            )),
        );
        let headers = auth
            .sign(&SigningContext {
                method: "POST",
                url: "https://aws-external-anthropic.us-west-2.api.aws/v1/messages",
                body: br#"{"ok":true}"#,
            })
            .await
            .unwrap();
        let names: Vec<_> = headers.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            names.contains(&"x-amz-security-token"),
            "session token should be propagated, got: {names:?}"
        );
    }
}
