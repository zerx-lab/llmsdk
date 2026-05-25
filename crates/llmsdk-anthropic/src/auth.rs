//! Pluggable per-request authentication hook.
//!
//! Mirrors the [`FetchFunction`] indirection used by `@ai-sdk/anthropic-aws`'s
//! `createSigV4FetchFunction` / `createApiKeyFetchFunction`. The default
//! [`Anthropic`](crate::Anthropic) provider authenticates via static headers
//! (`x-api-key` / `Authorization: Bearer ...`) set at builder time; downstream
//! providers (Anthropic on AWS, Amazon Bedrock, Google Vertex's Anthropic
//! passthrough, ...) need to compute auth headers **per request** because the
//! signature depends on method, URL, body bytes, and a fresh credential.
//!
//! This module exposes a minimal async trait such providers can implement
//! without forking the entire Messages / Files / Skills code paths.
// Rust guideline compliant 2026-02-21

use std::fmt::Debug;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;

/// Inputs the auth hook needs to produce a signature.
///
/// Owned strings + a borrowed body slice keeps the trait object-safe and
/// avoids forcing implementors to clone large request payloads.
#[derive(Debug)]
pub struct SigningContext<'a> {
    /// HTTP method (`"POST"`, `"GET"`, ...). Uppercase ASCII.
    pub method: &'a str,
    /// Absolute request URL, exactly as it will be sent on the wire.
    pub url: &'a str,
    /// Raw request body bytes. Empty slice for `GET` / bodyless requests.
    pub body: &'a [u8],
    /// On-wire `content-type` already chosen by the request builder, when
    /// known (`application/json` for Messages, `multipart/form-data; ...`
    /// for Files/Skills). `None` for `GET` / bodyless requests.
    ///
    /// Hooks that need this for canonicalization (e.g. AWS `SigV4`) should
    /// prefer this value over body-byte sniffing — it's authoritative.
    pub content_type: Option<&'a str>,
}

/// Headers produced by [`RequestAuth::sign`], merged into the final request.
///
/// Use `(name, value)` ordering — values may legitimately differ per request
/// (e.g. `x-amz-date`). Names follow [`reqwest::header::HeaderName`] rules.
pub type SignedHeaders = Vec<(String, String)>;

/// Apply a [`RequestAuth`] hook to a header map.
///
/// `body` may be empty for `GET` requests. The returned headers from the
/// hook are merged in last; on name collisions the hook value wins.
pub(crate) async fn apply_request_auth(
    auth: Option<&std::sync::Arc<dyn RequestAuth>>,
    headers: &mut std::collections::HashMap<String, Option<String>>,
    method: &str,
    url: &str,
    body: &[u8],
    content_type: Option<&str>,
) -> Result<(), ProviderError> {
    let Some(hook) = auth else {
        return Ok(());
    };
    let signed = hook
        .sign(&SigningContext {
            method,
            url,
            body,
            content_type,
        })
        .await?;
    for (name, value) in signed {
        headers.insert(name, Some(value));
    }
    Ok(())
}

/// Per-request authentication hook.
///
/// Implementations resolve credentials and compute any signature-derived
/// headers the upstream API requires (e.g. AWS `SigV4`'s `Authorization` +
/// `x-amz-date` + `x-amz-security-token`). The returned pairs are appended
/// to the request headers **after** the provider's built-in headers; later
/// values win, so an implementation may override any provider-level header
/// by emitting the same name.
///
/// Hooks are invoked on every outbound request (Messages, Files, Skills).
///
/// # Errors
///
/// Implementations should surface credential-resolution errors as
/// [`ProviderError::load_api_key`] and signing failures via
/// [`ProviderError::api_call_builder`].
#[async_trait]
pub trait RequestAuth: Send + Sync + Debug {
    /// Compute the headers to append for one outgoing request.
    async fn sign(&self, context: &SigningContext<'_>) -> Result<SignedHeaders, ProviderError>;
}
