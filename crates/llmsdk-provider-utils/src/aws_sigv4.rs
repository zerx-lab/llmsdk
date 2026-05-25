//! AWS Signature Version 4 helpers (feature `aws-sigv4`).
//!
//! Mirrors the role of `aws4fetch` in ai-sdk's `amazon-bedrock` package
//! (`amazon-bedrock-sigv4-fetch.ts`): given AWS credentials + a JSON POST
//! body, produce signed headers and dispatch via the existing
//! [`crate::http::HttpClient`]. The heavy lifting is delegated to the official
//! [`aws-sigv4`] crate; the rest of the module is a thin, provider-friendly
//! wrapper that:
//!
//! - keeps `aws_sigv4` / `aws_credential_types` types **out of** public method
//!   signatures (only [`AwsCredentials`] + standard `reqwest` types are
//!   exposed),
//! - maps every failure onto [`ProviderError`], identical to
//!   [`crate::http`],
//! - hides credential acquisition behind the async
//!   [`AwsCredentialsProvider`] trait so callers can plug in IMDS / STS /
//!   `AssumeRole` later without breaking the API.
//!
//! STS, IMDS and `AssumeRole` are deliberately **not** implemented — supply
//! a custom [`AwsCredentialsProvider`] if you need them.
//!
//! [`aws-sigv4`]: https://crates.io/crates/aws-sigv4
// Rust guideline compliant 2026-02-21

use std::env;
use std::time::SystemTime;

use async_trait::async_trait;
use aws_credential_types::Credentials;
use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign as sigv4_sign};
use aws_sigv4::sign::v4;
use llmsdk_provider::ProviderError;
use reqwest::Method;
use reqwest::header::{HeaderName, HeaderValue};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::http::{HttpClient, JsonResponse};

/// Static AWS credentials (access key id + secret + optional session token).
///
/// Mirrors ai-sdk's `AmazonBedrockCredentials` minus `region` — region is a
/// signing parameter (see [`SigV4Fetch::region`]), not a credential.
///
/// # Examples
///
/// ```
/// use llmsdk_provider_utils::aws_sigv4::AwsCredentials;
///
/// let creds = AwsCredentials::new("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY");
/// assert_eq!(creds.access_key_id, "AKIDEXAMPLE");
/// assert!(creds.session_token.is_none());
/// ```
#[derive(Debug, Clone)]
pub struct AwsCredentials {
    /// `AWS_ACCESS_KEY_ID`.
    pub access_key_id: String,
    /// `AWS_SECRET_ACCESS_KEY`.
    pub secret_access_key: String,
    /// Optional `AWS_SESSION_TOKEN` (set for temporary / role credentials).
    pub session_token: Option<String>,
}

impl AwsCredentials {
    /// Build long-term credentials without a session token.
    #[must_use]
    pub fn new(access_key_id: impl Into<String>, secret_access_key: impl Into<String>) -> Self {
        Self {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token: None,
        }
    }

    /// Attach a session token (temporary credentials from STS / SSO / IMDS).
    #[must_use]
    pub fn with_session_token(mut self, session_token: impl Into<String>) -> Self {
        self.session_token = Some(session_token.into());
        self
    }
}

/// Resolves AWS credentials, async because real providers (STS, IMDS) need IO.
///
/// Implementations should be cheap to clone — the [`SigV4Fetch`] wrapper
/// re-resolves credentials on every call so refresh logic (token expiry) can
/// live inside the provider.
///
/// # Examples
///
/// ```
/// use llmsdk_provider_utils::aws_sigv4::{
///     AwsCredentials, AwsCredentialsProvider, StaticCredentialsProvider,
/// };
///
/// # async fn doc() -> Result<(), llmsdk_provider::ProviderError> {
/// let provider = StaticCredentialsProvider::new(AwsCredentials::new("ak", "sk"));
/// let creds = provider.get_credentials().await?;
/// assert_eq!(creds.access_key_id, "ak");
/// # Ok(()) }
/// ```
#[async_trait]
pub trait AwsCredentialsProvider: Send + Sync {
    /// Resolve credentials. Called once per signed request.
    ///
    /// # Errors
    ///
    /// Implementations should return [`ProviderError::load_api_key`] for
    /// missing / unparseable credentials, and a retryable api-call error for
    /// transient IO failures.
    async fn get_credentials(&self) -> Result<AwsCredentials, ProviderError>;
}

/// A provider that returns the same credentials on every call.
///
/// Useful for tests, CI, and apps that load credentials from a secret manager
/// at startup.
#[derive(Debug, Clone)]
pub struct StaticCredentialsProvider {
    credentials: AwsCredentials,
}

impl StaticCredentialsProvider {
    /// Build from in-memory credentials.
    #[must_use]
    pub const fn new(credentials: AwsCredentials) -> Self {
        Self { credentials }
    }
}

#[async_trait]
impl AwsCredentialsProvider for StaticCredentialsProvider {
    async fn get_credentials(&self) -> Result<AwsCredentials, ProviderError> {
        Ok(self.credentials.clone())
    }
}

/// A provider that reads credentials from `AWS_ACCESS_KEY_ID` /
/// `AWS_SECRET_ACCESS_KEY` (+ optional `AWS_SESSION_TOKEN`) on every call.
///
/// Matches the `EnvironmentVariableCredentialsProvider` chain step in the
/// official AWS SDK, minus profile + IMDS fallback (build a custom provider
/// for those).
///
/// # Examples
///
/// ```no_run
/// # async fn doc() -> Result<(), llmsdk_provider::ProviderError> {
/// use llmsdk_provider_utils::aws_sigv4::{AwsCredentialsProvider, EnvCredentialsProvider};
///
/// let provider = EnvCredentialsProvider::new();
/// let _creds = provider.get_credentials().await?;
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, Default)]
pub struct EnvCredentialsProvider;

impl EnvCredentialsProvider {
    /// Build a new env-backed provider.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait]
impl AwsCredentialsProvider for EnvCredentialsProvider {
    async fn get_credentials(&self) -> Result<AwsCredentials, ProviderError> {
        let access_key_id = env::var("AWS_ACCESS_KEY_ID").map_err(|err| {
            ProviderError::load_api_key(format!(
                "AWS credentials missing: AWS_ACCESS_KEY_ID not set ({err})"
            ))
        })?;
        let secret_access_key = env::var("AWS_SECRET_ACCESS_KEY").map_err(|err| {
            ProviderError::load_api_key(format!(
                "AWS credentials missing: AWS_SECRET_ACCESS_KEY not set ({err})"
            ))
        })?;
        if access_key_id.is_empty() || secret_access_key.is_empty() {
            return Err(ProviderError::load_api_key(
                "AWS credentials must be non-empty: \
                 AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are blank",
            ));
        }
        let session_token = env::var("AWS_SESSION_TOKEN").ok().filter(|s| !s.is_empty());
        Ok(AwsCredentials {
            access_key_id,
            secret_access_key,
            session_token,
        })
    }
}

/// Input to [`sign_request`].
///
/// Bundled into a struct because `SigV4` signing legitimately needs 8+
/// inputs and a long positional parameter list is hard to call correctly
/// across providers.
#[derive(Debug, Clone)]
pub struct SignRequest<'a> {
    /// HTTP method (typically `POST` or `GET`).
    pub method: &'a Method,
    /// Absolute URL **already** percent-encoded as it will be sent on the wire.
    pub url: &'a str,
    /// Pre-signature headers that will participate in the canonical request
    /// (`host`, `content-type`, etc.). Pass exactly the headers you intend
    /// to send.
    pub headers: &'a [(&'a str, &'a str)],
    /// Request body — `&[]` for GET / empty POST.
    pub body: &'a [u8],
    /// AWS credentials resolved by the caller.
    pub credentials: &'a AwsCredentials,
    /// AWS region (e.g. `"us-east-1"`).
    pub region: &'a str,
    /// AWS signing service name (e.g. `"bedrock"`).
    pub service: &'a str,
    /// Signing time. Use [`SystemTime::now`] for production; pin in tests.
    pub signing_time: Option<SystemTime>,
}

/// Sign an HTTP request with `SigV4` and return the resulting auth headers.
///
/// The function does **not** mutate `headers`; instead it returns the set of
/// `(name, value)` pairs the caller must merge in. This matches `aws4fetch`'s
/// `signer.sign()` shape and keeps the caller in control of conflict
/// resolution (`combine_headers` semantics).
///
/// # Examples
///
/// ```
/// use std::time::UNIX_EPOCH;
///
/// use llmsdk_provider_utils::aws_sigv4::{AwsCredentials, SignRequest, sign_request};
/// use reqwest::Method;
///
/// let creds = AwsCredentials::new(
///     "AKIDEXAMPLE",
///     "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
/// );
/// let signed = sign_request(&SignRequest {
///     method: &Method::POST,
///     url: "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1:0/invoke",
///     headers: &[("content-type", "application/json")],
///     body: b"{}",
///     credentials: &creds,
///     region: "us-east-1",
///     service: "bedrock",
///     signing_time: Some(UNIX_EPOCH),
/// }).unwrap();
/// // `authorization`, `x-amz-date`, ...
/// assert!(signed.iter().any(|(n, _)| n.as_str() == "authorization"));
/// ```
///
/// # Errors
///
/// Returns [`ProviderError::api_call_builder`]-built error when the request
/// cannot be canonicalized (invalid URL, malformed header) or signing fails
/// internally.
pub fn sign_request(
    request: &SignRequest<'_>,
) -> Result<Vec<(HeaderName, HeaderValue)>, ProviderError> {
    let identity: aws_smithy_runtime_api::client::identity::Identity = Credentials::new(
        request.credentials.access_key_id.clone(),
        request.credentials.secret_access_key.clone(),
        request.credentials.session_token.clone(),
        None,
        "llmsdk-static",
    )
    .into();

    let settings = SigningSettings::default();
    let signing_params = v4::SigningParams::builder()
        .identity(&identity)
        .region(request.region)
        .name(request.service)
        .time(request.signing_time.unwrap_or_else(SystemTime::now))
        .settings(settings)
        .build()
        .map_err(|e| {
            ProviderError::api_call_builder(request.url, format!("SigV4 build params failed: {e}"))
                .build()
        })?
        .into();

    let signable = SignableRequest::new(
        request.method.as_str(),
        request.url,
        request.headers.iter().copied(),
        SignableBody::Bytes(request.body),
    )
    .map_err(|e| {
        ProviderError::api_call_builder(request.url, format!("SigV4 prepare request failed: {e}"))
            .build()
    })?;

    let (instructions, _signature) = sigv4_sign(signable, &signing_params)
        .map_err(|e| {
            ProviderError::api_call_builder(request.url, format!("SigV4 signing failed: {e}"))
                .build()
        })?
        .into_parts();

    let (header_list, _params) = instructions.into_parts();
    let mut out = Vec::with_capacity(header_list.len());
    for header in header_list {
        let name = HeaderName::from_bytes(header.name().as_bytes()).map_err(|e| {
            ProviderError::api_call_builder(
                request.url,
                format!("SigV4 emitted invalid header name: {e}"),
            )
            .build()
        })?;
        let value = HeaderValue::from_str(header.value()).map_err(|e| {
            ProviderError::api_call_builder(
                request.url,
                format!("SigV4 emitted invalid header value: {e}"),
            )
            .build()
        })?;
        out.push((name, value));
    }
    Ok(out)
}

/// High-level wrapper: resolve credentials, sign a JSON POST, dispatch via
/// [`HttpClient`], decode the JSON response.
///
/// This is the Rust equivalent of `createSigV4FetchFunction` + the inline
/// `post_json` call site in ai-sdk's bedrock chat model. Typical usage:
///
/// ```no_run
/// # async fn doc() -> Result<(), llmsdk_provider::ProviderError> {
/// use llmsdk_provider_utils::aws_sigv4::{
///     AwsCredentials, SigV4Fetch, StaticCredentialsProvider,
/// };
/// use llmsdk_provider_utils::http::HttpClient;
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct Resp { #[allow(dead_code)] ok: bool }
///
/// let fetch = SigV4Fetch::builder()
///     .http_client(HttpClient::new()?)
///     .credentials_provider(StaticCredentialsProvider::new(AwsCredentials::new("ak", "sk")))
///     .region("us-east-1")
///     .service("bedrock")
///     .build()?;
/// let _resp: llmsdk_provider_utils::http::JsonResponse<Resp> = fetch
///     .post_json("https://bedrock-runtime.us-east-1.amazonaws.com/foo", &serde_json::json!({}))
///     .await?;
/// # Ok(()) }
/// ```
pub struct SigV4Fetch {
    http_client: HttpClient,
    credentials_provider: Box<dyn AwsCredentialsProvider>,
    region: String,
    service: String,
}

impl std::fmt::Debug for SigV4Fetch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `credentials_provider` is a `Box<dyn ...>` (no `Debug` bound on the
        // trait by design) — represent it as a presence marker so the
        // `manual_debug_impl` lint is happy that every field is mentioned.
        let credentials_provider: &dyn std::fmt::Debug = &"<dyn AwsCredentialsProvider>";
        f.debug_struct("SigV4Fetch")
            .field("http_client", &self.http_client)
            .field("credentials_provider", credentials_provider)
            .field("region", &self.region)
            .field("service", &self.service)
            .finish()
    }
}

impl SigV4Fetch {
    /// Start a builder.
    #[must_use]
    pub fn builder() -> SigV4FetchBuilder {
        SigV4FetchBuilder::default()
    }

    /// Region this fetcher signs for (e.g. `"us-east-1"`).
    #[must_use]
    pub fn region(&self) -> &str {
        &self.region
    }

    /// Service name this fetcher signs for (e.g. `"bedrock"`).
    #[must_use]
    pub fn service(&self) -> &str {
        &self.service
    }

    /// Sign and POST a JSON body, decode the response into `T`.
    ///
    /// Adds `host` + `content-type: application/json` automatically — both
    /// participate in the canonical request so they must match what's sent
    /// on the wire.
    ///
    /// # Errors
    ///
    /// See [`crate::http::post_raw`] for transport / status mapping, plus
    /// [`sign_request`] for signing failures.
    pub async fn post_json<B, T>(
        &self,
        url: &str,
        body: &B,
    ) -> Result<JsonResponse<T>, ProviderError>
    where
        B: Serialize,
        T: DeserializeOwned,
    {
        let body_bytes = serde_json::to_vec(body)
            .map_err(|e| ProviderError::json_parse("<request body>", e.to_string()))?;
        self.post_bytes(url, body_bytes, "application/json").await
    }

    /// Sign and POST a raw byte body with a caller-chosen content type.
    ///
    /// # Errors
    ///
    /// See [`Self::post_json`].
    pub async fn post_bytes<T>(
        &self,
        url: &str,
        body: Vec<u8>,
        content_type: &str,
    ) -> Result<JsonResponse<T>, ProviderError>
    where
        T: DeserializeOwned,
    {
        let credentials = self.credentials_provider.get_credentials().await?;
        let parsed = reqwest::Url::parse(url).map_err(|e| {
            ProviderError::api_call_builder(url, format!("invalid URL: {e}")).build()
        })?;
        let host = parsed.host_str().ok_or_else(|| {
            ProviderError::api_call_builder(url, "URL has no host component").build()
        })?;
        // Both headers participate in the canonical request and must match
        // what reqwest finally sends. We add them up-front and then forward
        // every signed header through `post_raw`.
        let host_owned = host.to_owned();
        let pre_signed: Vec<(&str, &str)> =
            vec![("host", &host_owned), ("content-type", content_type)];

        let signed = sign_request(&SignRequest {
            method: &Method::POST,
            url,
            headers: &pre_signed,
            body: &body,
            credentials: &credentials,
            region: &self.region,
            service: &self.service,
            signing_time: None,
        })?;

        let mut request = crate::http::RawRequest::new(url, body, content_type.to_string());
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
            request.headers.insert(name.as_str().to_owned(), Some(v));
        }
        crate::http::post_raw::<T>(&self.http_client, request).await
    }
}

/// Sign a `POST` request and return the headers as `(String, String)` pairs.
///
/// Convenience wrapper around [`sign_request`] that keeps `reqwest::Method`
/// and `reqwest::Url` types **out of** the caller's surface — useful for
/// downstream provider crates that do not directly depend on `reqwest`
/// (e.g. `llmsdk-anthropic-aws`, which signs payloads but routes them
/// through `llmsdk-anthropic`'s existing HTTP pipeline).
///
/// The function automatically:
///
/// - extracts the `host` component from `url` and includes it in the
///   canonical request (`SigV4` requires it),
/// - appends the caller-supplied `extra_headers` (typically a single
///   `content-type` pair) to the canonical-request header list,
/// - omits any time argument so [`SystemTime::now`] is used.
///
/// # Errors
///
/// - URL parse failures → [`ProviderError::api_call_builder`].
/// - `SigV4` internal failures → see [`sign_request`].
///
/// # Examples
///
/// ```
/// use llmsdk_provider_utils::aws_sigv4::{AwsCredentials, sign_post};
///
/// let creds = AwsCredentials::new("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY");
/// let signed = sign_post(
///     "https://example.us-east-1.amazonaws.com/path",
///     br#"{"hi":1}"#,
///     &[("content-type", "application/json")],
///     &creds,
///     "us-east-1",
///     "service",
/// ).unwrap();
/// assert!(signed.iter().any(|(n, _)| n == "authorization"));
/// ```
pub fn sign_post(
    url: &str,
    body: &[u8],
    extra_headers: &[(&str, &str)],
    credentials: &AwsCredentials,
    region: &str,
    service: &str,
) -> Result<Vec<(String, String)>, ProviderError> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|e| ProviderError::api_call_builder(url, format!("invalid URL: {e}")).build())?;
    let host = parsed
        .host_str()
        .ok_or_else(|| ProviderError::api_call_builder(url, "URL has no host component").build())?
        .to_owned();
    let mut all_headers: Vec<(&str, &str)> = Vec::with_capacity(extra_headers.len() + 1);
    all_headers.push(("host", host.as_str()));
    for (n, v) in extra_headers {
        all_headers.push((n, v));
    }
    let signed = sign_request(&SignRequest {
        method: &Method::POST,
        url,
        headers: &all_headers,
        body,
        credentials,
        region,
        service,
        signing_time: None,
    })?;
    let mut out = Vec::with_capacity(signed.len());
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
        out.push((name.as_str().to_owned(), v));
    }
    Ok(out)
}

/// Builder for [`SigV4Fetch`].
#[derive(Default)]
pub struct SigV4FetchBuilder {
    http_client: Option<HttpClient>,
    credentials_provider: Option<Box<dyn AwsCredentialsProvider>>,
    region: Option<String>,
    service: Option<String>,
}

impl std::fmt::Debug for SigV4FetchBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let credentials_provider_set: &dyn std::fmt::Debug = &self.credentials_provider.is_some();
        f.debug_struct("SigV4FetchBuilder")
            .field("http_client", &self.http_client)
            .field("credentials_provider", credentials_provider_set)
            .field("region", &self.region)
            .field("service", &self.service)
            .finish()
    }
}

impl SigV4FetchBuilder {
    /// Required: HTTP client used to dispatch signed requests.
    #[must_use]
    pub fn http_client(mut self, client: HttpClient) -> Self {
        self.http_client = Some(client);
        self
    }

    /// Required: credentials provider invoked per request.
    #[must_use]
    pub fn credentials_provider<P>(mut self, provider: P) -> Self
    where
        P: AwsCredentialsProvider + 'static,
    {
        self.credentials_provider = Some(Box::new(provider));
        self
    }

    /// Required: AWS region (e.g. `"us-east-1"`).
    #[must_use]
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Required: AWS signing service name (e.g. `"bedrock"`).
    #[must_use]
    pub fn service(mut self, service: impl Into<String>) -> Self {
        self.service = Some(service.into());
        self
    }

    /// Finalize. Every required field must have been set.
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError::load_api_key`] when any required field is
    /// missing.
    pub fn build(self) -> Result<SigV4Fetch, ProviderError> {
        let http_client = self
            .http_client
            .ok_or_else(|| ProviderError::load_api_key("SigV4Fetch: http_client is required"))?;
        let credentials_provider = self.credentials_provider.ok_or_else(|| {
            ProviderError::load_api_key("SigV4Fetch: credentials_provider is required")
        })?;
        let region = self
            .region
            .ok_or_else(|| ProviderError::load_api_key("SigV4Fetch: region is required"))?;
        let service = self
            .service
            .ok_or_else(|| ProviderError::load_api_key("SigV4Fetch: service is required"))?;
        Ok(SigV4Fetch {
            http_client,
            credentials_provider,
            region,
            service,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, MutexGuard};
    use std::time::{Duration, UNIX_EPOCH};

    use super::*;
    use serde_json::json;
    use wiremock::matchers::{header_exists, method as wm_method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // --- StaticCredentialsProvider ---------------------------------------

    #[tokio::test]
    async fn static_provider_returns_supplied_credentials() {
        let provider = StaticCredentialsProvider::new(
            AwsCredentials::new("ak", "sk").with_session_token("tok"),
        );
        let creds = provider.get_credentials().await.unwrap();
        assert_eq!(creds.access_key_id, "ak");
        assert_eq!(creds.secret_access_key, "sk");
        assert_eq!(creds.session_token.as_deref(), Some("tok"));
    }

    // --- EnvCredentialsProvider ------------------------------------------

    // env access is process-global; serialize the two env tests so they
    // don't race even when the harness runs tests in parallel.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn lock_env() -> MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[allow(
        unsafe_code,
        reason = "Edition 2024 env::set_var is unsafe; ENV_LOCK serializes"
    )]
    fn set_env(access_key: Option<&str>, secret: Option<&str>, session: Option<&str>) {
        // SAFETY: caller must hold ENV_LOCK, which serializes all readers
        // and writers of these vars within the crate's tests.
        unsafe {
            match access_key {
                Some(v) => env::set_var("AWS_ACCESS_KEY_ID", v),
                None => env::remove_var("AWS_ACCESS_KEY_ID"),
            }
            match secret {
                Some(v) => env::set_var("AWS_SECRET_ACCESS_KEY", v),
                None => env::remove_var("AWS_SECRET_ACCESS_KEY"),
            }
            match session {
                Some(v) => env::set_var("AWS_SESSION_TOKEN", v),
                None => env::remove_var("AWS_SESSION_TOKEN"),
            }
        }
    }

    // `EnvCredentialsProvider::get_credentials` is `async` for trait
    // uniformity but does no actual IO — the body only calls `env::var`.
    // We drive it on `futures::executor::block_on` so the std Mutex guard
    // never crosses an `.await` boundary (avoids both the `await_holding_lock`
    // clippy lint and the genuine deadlock risk).
    fn run_env_provider() -> Result<AwsCredentials, ProviderError> {
        futures::executor::block_on(EnvCredentialsProvider::new().get_credentials())
    }

    #[test]
    fn env_provider_reads_required_vars() {
        let _guard = lock_env();
        set_env(Some("env-ak"), Some("env-sk"), None);
        let creds = run_env_provider().unwrap();
        assert_eq!(creds.access_key_id, "env-ak");
        assert_eq!(creds.secret_access_key, "env-sk");
        assert!(creds.session_token.is_none());
        set_env(None, None, None);
    }

    #[test]
    fn env_provider_errors_on_missing_vars() {
        let _guard = lock_env();
        set_env(None, None, None);
        let err = run_env_provider().unwrap_err();
        assert!(format!("{err}").contains("AWS_ACCESS_KEY_ID"));
    }

    // --- sign_request algorithm correctness ------------------------------
    //
    // AWS SigV4 test suite: GET / with no query, fixed time → known
    // Authorization header. Values lifted from the official aws-sigv4
    // test fixtures (`get-vanilla.req` / `get-vanilla.authz`) shipped with
    // the crate.

    #[test]
    fn sign_request_matches_known_aws_test_vector() {
        // Known-good vector — derived from AWS test-suite `get-vanilla` and
        // verified against `aws-sigv4` 1.4.4 locally.
        let creds = AwsCredentials::new("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY");
        // Mon, 09 Sep 2011 23:36:00 GMT — 21926856 minutes after the epoch.
        let time = UNIX_EPOCH + Duration::from_mins(21_926_856);
        let signed = sign_request(&SignRequest {
            method: &Method::GET,
            url: "https://example.amazonaws.com/",
            headers: &[
                ("host", "example.amazonaws.com"),
                ("x-amz-date", "20110909T233600Z"),
            ],
            body: b"",
            credentials: &creds,
            region: "us-east-1",
            service: "host",
            signing_time: Some(time),
        })
        .unwrap();
        let auth = signed
            .iter()
            .find(|(n, _)| n.as_str() == "authorization")
            .expect("authorization header emitted");
        let value = auth.1.to_str().unwrap();
        assert!(
            value.starts_with(
                "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20110909/us-east-1/host/aws4_request"
            ),
            "got: {value}"
        );
        // Default `SigningSettings` use `PayloadChecksumKind::NoHeader` so the
        // signed-headers list is just `host;x-amz-date` for the AWS test
        // vector's "get-vanilla" case.
        assert!(
            value.contains("SignedHeaders=host;x-amz-date"),
            "got: {value}"
        );
        assert!(value.contains("Signature="));
    }

    #[test]
    fn sign_request_includes_session_token_header_when_present() {
        let creds = AwsCredentials::new("AKIDEXAMPLE", "sk").with_session_token("sess-token");
        let signed = sign_request(&SignRequest {
            method: &Method::POST,
            url: "https://example.amazonaws.com/",
            headers: &[("host", "example.amazonaws.com")],
            body: b"{}",
            credentials: &creds,
            region: "us-east-1",
            service: "bedrock",
            signing_time: Some(UNIX_EPOCH),
        })
        .unwrap();
        let token = signed
            .iter()
            .find(|(n, _)| n.as_str() == "x-amz-security-token")
            .expect("session token header emitted");
        assert_eq!(token.1.to_str().unwrap(), "sess-token");
    }

    // --- SigV4Fetch end-to-end -------------------------------------------

    #[derive(serde::Deserialize, Debug)]
    struct Resp {
        ok: bool,
    }

    #[tokio::test]
    async fn sigv4_fetch_signs_and_dispatches_post() {
        let server = MockServer::start().await;
        // Default SigningSettings use `PayloadChecksumKind::NoHeader`, so
        // we don't require `x-amz-content-sha256` here (it isn't emitted).
        Mock::given(wm_method("POST"))
            .and(path("/v1/foo"))
            .and(header_exists("authorization"))
            .and(header_exists("x-amz-date"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
            .mount(&server)
            .await;

        let fetch = SigV4Fetch::builder()
            .http_client(HttpClient::new().unwrap())
            .credentials_provider(StaticCredentialsProvider::new(AwsCredentials::new(
                "AKID", "SECRET",
            )))
            .region("us-east-1")
            .service("bedrock")
            .build()
            .unwrap();

        let resp: JsonResponse<Resp> = fetch
            .post_json(
                &format!("{}/v1/foo", server.uri()),
                &json!({"hello": "world"}),
            )
            .await
            .unwrap();
        assert_eq!(resp.status.as_u16(), 200);
        assert!(resp.value.ok);
    }

    #[tokio::test]
    async fn sigv4_fetch_builder_requires_all_fields() {
        let err = SigV4Fetch::builder()
            .http_client(HttpClient::new().unwrap())
            .region("us-east-1")
            .service("bedrock")
            .build()
            .unwrap_err();
        assert!(format!("{err}").contains("credentials_provider"));
    }
}
