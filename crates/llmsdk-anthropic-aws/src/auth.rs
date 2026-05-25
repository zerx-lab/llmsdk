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
        // Prefer the authoritative `content-type` the request builder already
        // chose (`application/json` for Messages, `multipart/form-data; ...`
        // for Files/Skills). Fall back to body-byte sniffing only when the
        // framework can't supply one (legacy or third-party RequestAuth
        // construction sites).
        let content_type = context
            .content_type
            .map_or_else(|| sniff_content_type(context.body).to_owned(), str::to_owned);
        sign_post(
            context.url,
            context.body,
            &[("content-type", content_type.as_str())],
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

/// Best-effort MIME sniffer used as a fallback when [`SigningContext::content_type`]
/// is `None`.
///
/// Skips ASCII whitespace before looking at the first significant byte so a
/// pretty-printed JSON document with a leading newline still classifies as
/// JSON. Multipart bodies always start with `--` (the boundary marker per
/// RFC 7578), so we explicitly check that prefix before falling back to
/// `application/octet-stream` for opaque payloads.
///
/// llmsdk-anthropic emits exactly two POST content types today
/// (`application/json` and `multipart/form-data`), but the fallback is kept
/// permissive so third-party `RequestAuth` impls that don't yet pass an
/// explicit `content_type` keep working.
fn sniff_content_type(body: &[u8]) -> &'static str {
    let trimmed = body
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .map_or(body, |idx| &body[idx..]);
    if trimmed.starts_with(b"--") {
        return "multipart/form-data";
    }
    if matches!(trimmed.first(), Some(&(b'{' | b'[' | b'"'))) {
        return "application/json";
    }
    "application/octet-stream"
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider_utils::aws_sigv4::{AwsCredentials, StaticCredentialsProvider};

    #[test]
    fn sniff_skips_leading_whitespace_for_json() {
        assert_eq!(sniff_content_type(b"\n  {\"k\":1}"), "application/json");
        assert_eq!(sniff_content_type(b"\t[1,2]"), "application/json");
        assert_eq!(sniff_content_type(b"\"hi\""), "application/json");
    }

    #[test]
    fn sniff_detects_multipart_boundary_prefix() {
        assert_eq!(
            sniff_content_type(b"--boundary\r\nContent-Disposition"),
            "multipart/form-data"
        );
    }

    #[test]
    fn sniff_unknown_binary_falls_back_to_octet_stream() {
        assert_eq!(
            sniff_content_type(&[0xFF, 0xD8, 0xFF, 0xE0]),
            "application/octet-stream"
        );
    }

    #[tokio::test]
    async fn api_key_auth_emits_xapikey() {
        let auth = ApiKeyAuth::new("sk-test");
        let headers = auth
            .sign(&SigningContext {
                method: "POST",
                url: "https://example.com/messages",
                body: b"{}",
                content_type: Some("application/json"),
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
                content_type: None,
            })
            .await
            .unwrap();
        assert!(get_headers.is_empty());

        let empty_post = auth
            .sign(&SigningContext {
                method: "POST",
                url: "https://example.com/messages",
                body: &[],
                content_type: Some("application/json"),
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
                content_type: Some("application/json"),
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
                content_type: Some("application/json"),
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
