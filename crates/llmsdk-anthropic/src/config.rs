//! Provider configuration.
//!
//! Mirrors `@ai-sdk/anthropic/src/anthropic-provider.ts`. `Anthropic` uses
//! `x-api-key` by default (or `Authorization: Bearer` when `auth_token` is
//! set) and always sends the `anthropic-version` header.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use llmsdk_provider::ProviderError;
use llmsdk_provider_utils::http::HttpClient;
use serde_json::Value;

use crate::auth::RequestAuth;
use crate::files::AnthropicFiles;
use crate::messages::AnthropicMessagesModel;
use crate::skills::AnthropicSkills;
use crate::{
    API_KEY_ENV_VAR, AUTH_TOKEN_ENV_VAR, DEFAULT_ANTHROPIC_VERSION, DEFAULT_BASE_URL, PROVIDER_ID,
};

const DEFAULT_PROVIDER_NAME: &str = "anthropic.messages";

/// Per-request URL builder used by [`Inner::endpoint_override`].
///
/// Called as `f(base_url, model_id, is_streaming)`. Returns the absolute
/// URL the messages endpoint should target. Wrapping providers (Google
/// Vertex Anthropic, Amazon Bedrock Anthropic) use this to inject
/// `{model_id}:rawPredict` style paths instead of the default
/// `{base_url}/messages`.
pub type EndpointFn = dyn Fn(&str, &str, bool) -> String + Send + Sync;

/// Per-request body transformer used by [`Inner::body_transformer`].
///
/// Called on the JSON wire body just before serialization, with the full
/// set of collected `anthropic-beta` tokens. Wrapping providers use this
/// to strip `model` (Bedrock / Vertex put it in the URL), inject
/// `anthropic_version`, and — for Bedrock — fold all collected `betas`
/// into the body's `anthropic_beta` field (Bedrock's Anthropic surface
/// reads the beta list from the body, not from headers).
pub type BodyTransformFn = dyn Fn(&mut Value, &std::collections::BTreeSet<String>) + Send + Sync;

/// Callback used to generate ids for `Source` parts produced from
/// `citations_delta` blocks. Mirrors `config.generateId` in the upstream
/// `AnthropicLanguageModel`.
pub type GenerateIdFn = dyn Fn() -> String + Send + Sync;

/// `Anthropic` provider handle.
///
/// Cheap to clone; HTTP client and headers are shared.
#[derive(Debug, Clone)]
pub struct Anthropic {
    inner: Arc<Inner>,
}

/// Internal connection / routing state shared across all model handles
/// produced by a single provider instance.
///
/// Public for cross-crate wrapping providers (Google Vertex Anthropic,
/// Amazon Bedrock Anthropic) — *not* part of the user-facing surface.
/// Re-exported under [`crate::internal`].
pub struct Inner {
    pub(crate) base_url: String,
    pub(crate) headers: HashMap<String, Option<String>>,
    pub(crate) http: HttpClient,
    pub(crate) provider_name: String,
    /// Optional per-request signer. Default `Anthropic` providers leave this
    /// `None` and rely entirely on the static headers above.
    pub(crate) request_auth: Option<Arc<dyn RequestAuth>>,
    /// Optional custom URL builder. When `None`, the default
    /// `{base_url}/messages` is used.
    pub(crate) endpoint_override: Option<Arc<EndpointFn>>,
    /// Optional request body transformer. When `None`, the JSON body is
    /// sent verbatim.
    pub(crate) body_transformer: Option<Arc<BodyTransformFn>>,
    /// Optional generator for citation source ids. When `None`, an
    /// in-stream counter is used (`anthropic-cite-{n}`).
    pub(crate) generate_id: Option<Arc<GenerateIdFn>>,
    /// Wrapping-provider override for `supportsNativeStructuredOutput`
    /// (default `true`). Bedrock pins this to `false` for `claude-opus-4-7`
    /// because the AWS gateway rejects `output_config.format` for that
    /// model. Combined with `model_capabilities().supports_structured_output`
    /// to drive the jsonResponseTool fallback path. Mirrors upstream
    /// `anthropic-language-model.ts:332` reading `config.supportsNativeStructuredOutput`.
    pub(crate) supports_native_structured_output: bool,
}

impl fmt::Debug for Inner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Inner")
            .field("base_url", &self.base_url)
            .field("headers", &self.headers)
            .field("http", &self.http)
            .field("provider_name", &self.provider_name)
            .field("request_auth", &self.request_auth.is_some())
            .field("endpoint_override", &self.endpoint_override.is_some())
            .field("body_transformer", &self.body_transformer.is_some())
            .field("generate_id", &self.generate_id.is_some())
            .field(
                "supports_native_structured_output",
                &self.supports_native_structured_output,
            )
            .finish()
    }
}

impl Inner {
    /// Open a typed builder for [`Inner`].
    ///
    /// Cross-crate composition entry point: wrapping providers build the
    /// [`Inner`] directly with custom provider name / base URL / headers /
    /// HTTP client / URL hook / body transform and inject it into
    /// [`AnthropicMessagesModel::new`].
    #[must_use]
    pub fn builder() -> InnerBuilder {
        InnerBuilder::default()
    }

    /// Resolve the messages endpoint for `model_id`.
    ///
    /// Default: `{base_url}/messages`. Wrapping providers override via
    /// [`InnerBuilder::endpoint`].
    #[must_use]
    pub fn endpoint_url(&self, model_id: &str, is_streaming: bool) -> String {
        match &self.endpoint_override {
            Some(f) => f(&self.base_url, model_id, is_streaming),
            None => format!("{}/messages", self.base_url),
        }
    }

    /// Apply the registered body transformer (no-op when none set).
    ///
    /// `betas` carries every `anthropic-beta` token the language-model
    /// collected for this call so wrapping backends (Bedrock) can copy them
    /// into the request body.
    pub fn transform_body(&self, body: &mut Value, betas: &std::collections::BTreeSet<String>) {
        if let Some(f) = &self.body_transformer {
            f(body, betas);
        }
    }

    /// Whether the configured backend honors `output_config.format`
    /// natively. Wrapping providers (Bedrock for `claude-opus-4-7`) flip
    /// this off via [`InnerBuilder::supports_native_structured_output`]
    /// so the language model can fall back to the jsonResponseTool path.
    #[must_use]
    pub fn supports_native_structured_output(&self) -> bool {
        self.supports_native_structured_output
    }
}

/// Builder for the cross-crate [`Inner`].
///
/// Used by wrapping providers (Google Vertex Anthropic, Amazon Bedrock
/// Anthropic) to assemble an [`Inner`] without going through the
/// user-facing [`Anthropic`] builder.
#[derive(Clone)]
pub struct InnerBuilder {
    base_url: Option<String>,
    headers: HashMap<String, Option<String>>,
    http: Option<HttpClient>,
    provider_name: Option<String>,
    request_auth: Option<Arc<dyn RequestAuth>>,
    endpoint_override: Option<Arc<EndpointFn>>,
    body_transformer: Option<Arc<BodyTransformFn>>,
    generate_id: Option<Arc<GenerateIdFn>>,
    /// Default `true`. Cleared by wrapping providers whose backend
    /// rejects `output_config.format` (e.g. Bedrock + claude-opus-4-7),
    /// forcing the request through the jsonResponseTool fallback.
    supports_native_structured_output: bool,
}

impl Default for InnerBuilder {
    fn default() -> Self {
        Self {
            base_url: None,
            headers: HashMap::new(),
            http: None,
            provider_name: None,
            request_auth: None,
            endpoint_override: None,
            body_transformer: None,
            generate_id: None,
            // Match upstream `supportsNativeStructuredOutput ?? true`
            // (anthropic-language-model.ts:332). Wrapping providers
            // override per-model.
            supports_native_structured_output: true,
        }
    }
}

impl fmt::Debug for InnerBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InnerBuilder")
            .field("base_url", &self.base_url)
            .field("headers", &self.headers)
            .field("http", &self.http)
            .field("provider_name", &self.provider_name)
            .field("request_auth", &self.request_auth.is_some())
            .field("endpoint_override", &self.endpoint_override.is_some())
            .field("body_transformer", &self.body_transformer.is_some())
            .field("generate_id", &self.generate_id.is_some())
            .field(
                "supports_native_structured_output",
                &self.supports_native_structured_output,
            )
            .finish()
    }
}

impl InnerBuilder {
    /// Override the base URL. Defaults to [`crate::DEFAULT_BASE_URL`].
    #[must_use]
    pub fn base_url(mut self, value: impl Into<String>) -> Self {
        self.base_url = Some(value.into());
        self
    }

    /// Append or override a header. `None` removes it.
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: Option<String>) -> Self {
        self.headers.insert(name.into(), value);
        self
    }

    /// Inject a pre-configured HTTP client.
    #[must_use]
    pub fn http_client(mut self, client: HttpClient) -> Self {
        self.http = Some(client);
        self
    }

    /// Override the reported provider name.
    ///
    /// Defaults to `"anthropic.messages"`. Cross-crate composition uses
    /// values like `"google.vertex.anthropic.messages"` or
    /// `"bedrock.anthropic.messages"`.
    #[must_use]
    pub fn provider_name(mut self, value: impl Into<String>) -> Self {
        self.provider_name = Some(value.into());
        self
    }

    /// Install a per-request authentication hook (e.g. AWS `SigV4`).
    #[must_use]
    pub fn request_auth(mut self, auth: Arc<dyn RequestAuth>) -> Self {
        self.request_auth = Some(auth);
        self
    }

    /// Install a custom URL builder.
    ///
    /// Wrapping providers use this to route to publisher-prefixed
    /// `{model_id}:rawPredict` / `:streamRawPredict` paths instead of
    /// the default `{base_url}/messages`.
    #[must_use]
    pub fn endpoint<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &str, bool) -> String + Send + Sync + 'static,
    {
        self.endpoint_override = Some(Arc::new(f));
        self
    }

    /// Install a request-body transformer.
    ///
    /// Wrapping providers use this to strip `model` from the body (already
    /// in the URL) and inject `anthropic_version`.
    #[must_use]
    pub fn body_transform<F>(mut self, f: F) -> Self
    where
        F: Fn(&mut Value, &std::collections::BTreeSet<String>) + Send + Sync + 'static,
    {
        self.body_transformer = Some(Arc::new(f));
        self
    }

    /// Install a citation source id generator.
    ///
    /// Mirrors `config.generateId` on the upstream `AnthropicLanguageModel`.
    /// When `None`, an in-stream counter is used.
    #[must_use]
    pub fn generate_id<F>(mut self, f: F) -> Self
    where
        F: Fn() -> String + Send + Sync + 'static,
    {
        self.generate_id = Some(Arc::new(f));
        self
    }

    /// Set the `supportsNativeStructuredOutput` capability flag.
    ///
    /// Defaults to `true` (mirrors upstream `?? true`). Wrapping providers
    /// flip this to `false` to drive the model through the jsonResponseTool
    /// fallback — see Amazon Bedrock's `claude-opus-4-7` route, which
    /// rejects `output_config.format`.
    #[must_use]
    pub fn supports_native_structured_output(mut self, value: bool) -> Self {
        self.supports_native_structured_output = value;
        self
    }

    /// Finalize the [`Inner`].
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError`] when the default HTTP client fails to
    /// build (rare; misconfigured TLS).
    pub fn build(self) -> Result<Inner, ProviderError> {
        let http = match self.http {
            Some(client) => client,
            None => HttpClient::new()?,
        };
        Ok(Inner {
            base_url: self.base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_owned()),
            headers: self.headers,
            http,
            provider_name: self
                .provider_name
                .unwrap_or_else(|| DEFAULT_PROVIDER_NAME.to_owned()),
            request_auth: self.request_auth,
            endpoint_override: self.endpoint_override,
            body_transformer: self.body_transformer,
            generate_id: self.generate_id,
            supports_native_structured_output: self.supports_native_structured_output,
        })
    }
}

impl Anthropic {
    /// Open a builder.
    #[must_use]
    pub fn builder() -> AnthropicBuilder {
        AnthropicBuilder::default()
    }

    /// Build with defaults: API key from `ANTHROPIC_API_KEY`.
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError::load_api_key`] when the env var is unset
    /// or empty.
    pub fn from_env() -> Result<Self, ProviderError> {
        Self::builder().build()
    }

    /// Construct a Messages API model handle.
    ///
    /// Mirrors `provider.messages(modelId)` upstream.
    #[must_use]
    pub fn messages(&self, model_id: impl Into<String>) -> AnthropicMessagesModel {
        AnthropicMessagesModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Anthropic::messages`] — mirrors `provider.chat(modelId)`.
    #[must_use]
    pub fn chat(&self, model_id: impl Into<String>) -> AnthropicMessagesModel {
        self.messages(model_id)
    }

    /// Alias of [`Anthropic::messages`] — mirrors `provider.languageModel(modelId)`.
    #[must_use]
    pub fn language_model(&self, model_id: impl Into<String>) -> AnthropicMessagesModel {
        self.messages(model_id)
    }

    /// Files API handle (`POST /v1/files`).
    ///
    /// Mirrors `provider.files()`. The handle's `provider()` reports the
    /// same string as the Messages model handle — upstream sets
    /// `provider: providerName` for Files (no `.files` suffix), see
    /// `@ai-sdk/anthropic/src/anthropic-provider.ts:190` and the
    /// `expect(files.provider).toBe('anthropic-aws.messages')` test in
    /// `anthropic-aws-provider.test.ts:454`.
    #[must_use]
    pub fn files(&self) -> AnthropicFiles {
        AnthropicFiles::new(Arc::clone(&self.inner), self.inner.provider_name.clone())
    }

    /// Skills API handle (`POST /v1/skills`).
    ///
    /// Mirrors `provider.skills()`.
    #[must_use]
    pub fn skills(&self) -> AnthropicSkills {
        let provider = format!(
            "{}.skills",
            self.inner
                .provider_name
                .strip_suffix(".messages")
                .unwrap_or(&self.inner.provider_name)
        );
        AnthropicSkills::new(Arc::clone(&self.inner), provider)
    }

    /// Provider id reported on language-model handles.
    #[must_use]
    pub fn provider_name(&self) -> &str {
        &self.inner.provider_name
    }
}

/// Builder for [`Anthropic`].
#[derive(Default, Clone)]
pub struct AnthropicBuilder {
    api_key: Option<String>,
    auth_token: Option<String>,
    base_url: Option<String>,
    version: Option<String>,
    provider_name: Option<String>,
    extra_headers: HashMap<String, Option<String>>,
    http: Option<HttpClient>,
    request_auth: Option<Arc<dyn RequestAuth>>,
    skip_default_auth_headers: bool,
    generate_id: Option<Arc<GenerateIdFn>>,
}

impl fmt::Debug for AnthropicBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicBuilder")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field(
                "auth_token",
                &self.auth_token.as_ref().map(|_| "<redacted>"),
            )
            .field("base_url", &self.base_url)
            .field("version", &self.version)
            .field("provider_name", &self.provider_name)
            .field("extra_headers", &self.extra_headers)
            .field("http", &self.http)
            .field("request_auth", &self.request_auth.is_some())
            .field("skip_default_auth_headers", &self.skip_default_auth_headers)
            .field("generate_id", &self.generate_id.is_some())
            .finish()
    }
}

impl AnthropicBuilder {
    /// Set the API key explicitly. Mutually exclusive with [`Self::auth_token`].
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the bearer auth token explicitly. Mutually exclusive with
    /// [`Self::api_key`]. When set the provider sends
    /// `Authorization: Bearer {token}` instead of `x-api-key`.
    #[must_use]
    pub fn auth_token(mut self, token: impl Into<String>) -> Self {
        self.auth_token = Some(token.into());
        self
    }

    /// Override the base URL (e.g. for a proxy).
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Override the `anthropic-version` header (defaults to
    /// [`crate::DEFAULT_ANTHROPIC_VERSION`]).
    #[must_use]
    pub fn version(mut self, value: impl Into<String>) -> Self {
        self.version = Some(value.into());
        self
    }

    /// Override the reported provider name (default `"anthropic.messages"`).
    ///
    /// The `.files` / `.skills` suffix is derived automatically when calling
    /// [`Anthropic::files`] / [`Anthropic::skills`].
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.provider_name = Some(name.into());
        self
    }

    /// Append or override an extra header. `None` removes it.
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

    /// Install a per-request authentication hook.
    ///
    /// When set, the hook is invoked before every Messages / Files / Skills
    /// request and its returned headers are merged in on top of the
    /// builder-time headers. This enables AWS `SigV4` (see
    /// `llmsdk-anthropic-aws`) and similar dynamic auth schemes without
    /// forking the request pipeline.
    #[must_use]
    pub fn request_auth(mut self, auth: Arc<dyn RequestAuth>) -> Self {
        self.request_auth = Some(auth);
        self
    }

    /// Skip resolving the default `x-api-key` / `Authorization: Bearer`
    /// headers from env / explicit options.
    ///
    /// Use this when the caller has installed a [`Self::request_auth`] hook
    /// that supplies an alternative auth scheme (e.g. AWS `SigV4`, IAM key
    /// header) and the static API-key env vars are intentionally absent.
    /// Has no effect when `api_key` or `auth_token` is set explicitly
    /// (those continue to populate the corresponding header).
    #[must_use]
    pub fn skip_default_auth_headers(mut self, skip: bool) -> Self {
        self.skip_default_auth_headers = skip;
        self
    }

    /// Install a citation source id generator.
    ///
    /// Mirrors `config.generateId` on the upstream `AnthropicLanguageModel`.
    /// When unset, citation sources use an in-stream counter (
    /// `anthropic-cite-{n}`).
    #[must_use]
    pub fn generate_id<F>(mut self, f: F) -> Self
    where
        F: Fn() -> String + Send + Sync + 'static,
    {
        self.generate_id = Some(Arc::new(f));
        self
    }

    /// Finalize the provider.
    ///
    /// # Errors
    ///
    /// - [`ProviderError::invalid_argument`] when both `api_key` and
    ///   `auth_token` are explicitly set.
    /// - [`ProviderError::load_api_key`] when neither an explicit key nor
    ///   `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_API_KEY` is available.
    /// - [`ProviderError`] when the HTTP client cannot initialize.
    pub fn build(self) -> Result<Anthropic, ProviderError> {
        if self.api_key.is_some() && self.auth_token.is_some() {
            return Err(ProviderError::invalid_argument(
                "api_key/auth_token",
                "Both apiKey and authToken were provided. Please use only one authentication method.",
            ));
        }

        let explicit_token = self.auth_token.clone();
        let resolved_auth_token = explicit_token
            .clone()
            .or_else(|| std::env::var(AUTH_TOKEN_ENV_VAR).ok())
            .filter(|s| !s.is_empty());

        let mut headers = self.extra_headers;
        if let Some(token) = resolved_auth_token {
            headers.insert("Authorization".into(), Some(format!("Bearer {token}")));
        } else if self.api_key.is_some() || !self.skip_default_auth_headers {
            // Either an explicit api_key was provided, or we have to fall
            // back to env-resolution because no alternative auth hook was
            // installed via skip_default_auth_headers + request_auth.
            let api_key = llmsdk_provider_utils::api_key::load_api_key(
                &llmsdk_provider_utils::api_key::LoadApiKey {
                    api_key: self.api_key.as_deref(),
                    env_var: API_KEY_ENV_VAR,
                    description: "Anthropic",
                    parameter_name: Some("api_key"),
                },
            )?;
            headers.insert("x-api-key".into(), Some(api_key));
        }

        let version = self
            .version
            .unwrap_or_else(|| DEFAULT_ANTHROPIC_VERSION.to_owned());
        headers.insert("anthropic-version".into(), Some(version));

        let base_url = self.base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());

        let provider_name = self
            .provider_name
            .unwrap_or_else(|| DEFAULT_PROVIDER_NAME.to_owned());

        let _ = PROVIDER_ID;

        let http = match self.http {
            Some(client) => client,
            None => HttpClient::new()?,
        };

        Ok(Anthropic {
            inner: Arc::new(Inner {
                base_url,
                headers,
                http,
                provider_name,
                request_auth: self.request_auth,
                endpoint_override: None,
                body_transformer: None,
                generate_id: self.generate_id,
                // Native Anthropic backend honors output_config.format.
                supports_native_structured_output: true,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::{FilesModel, LanguageModel, SkillsModel};

    fn fixed_key() -> Anthropic {
        Anthropic::builder()
            .api_key("test-key")
            .build()
            .expect("builder should succeed with explicit key")
    }

    #[test]
    fn explicit_api_key_sets_xapikey_header() {
        let a = fixed_key();
        assert_eq!(
            a.inner.headers.get("x-api-key"),
            Some(&Some("test-key".to_owned()))
        );
        assert!(!a.inner.headers.contains_key("Authorization"));
    }

    #[test]
    fn explicit_auth_token_sets_bearer_header() {
        let a = Anthropic::builder()
            .auth_token("oauth-token")
            .build()
            .unwrap();
        assert_eq!(
            a.inner.headers.get("Authorization"),
            Some(&Some("Bearer oauth-token".to_owned()))
        );
        assert!(!a.inner.headers.contains_key("x-api-key"));
    }

    #[test]
    fn both_api_key_and_auth_token_errors() {
        let err = Anthropic::builder()
            .api_key("k")
            .auth_token("t")
            .build()
            .unwrap_err();
        assert!(format!("{err}").contains("Both apiKey and authToken were provided"));
    }

    #[test]
    fn chat_and_language_model_are_messages_aliases() {
        let a = fixed_key();
        let m = a.messages("claude-sonnet-4-6");
        let c = a.chat("claude-sonnet-4-6");
        let l = a.language_model("claude-sonnet-4-6");
        assert_eq!(m.model_id(), c.model_id());
        assert_eq!(m.model_id(), l.model_id());
    }

    #[test]
    fn files_handle_reuses_messages_provider_name() {
        // Mirrors upstream `anthropic-provider.ts:190` where Files takes
        // `provider: providerName` directly (no `.files` suffix).
        let a = fixed_key();
        let f = a.files();
        assert_eq!(f.provider(), "anthropic.messages");
    }

    #[test]
    fn skills_handle_reports_provider_suffix() {
        let a = fixed_key();
        let s = a.skills();
        assert_eq!(s.provider(), "anthropic.skills");
    }

    #[test]
    fn custom_provider_name_propagates_to_files_skills() {
        let a = Anthropic::builder()
            .api_key("k")
            .name("acme.bedrock.anthropic.messages")
            .build()
            .unwrap();
        // Files reuses the full messages name; Skills swaps the suffix.
        assert_eq!(a.files().provider(), "acme.bedrock.anthropic.messages");
        assert_eq!(a.skills().provider(), "acme.bedrock.anthropic.skills");
    }

    #[test]
    fn anthropic_version_header_present() {
        let a = fixed_key();
        assert_eq!(
            a.inner.headers.get("anthropic-version"),
            Some(&Some(DEFAULT_ANTHROPIC_VERSION.to_owned()))
        );
    }
}
