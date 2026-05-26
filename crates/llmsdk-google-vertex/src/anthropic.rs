//! Anthropic on Vertex (`/publishers/anthropic/models/{id}:rawPredict`).
//!
//! Mirrors `anthropic/google-vertex-anthropic-provider.ts`. Builds on
//! `llmsdk-anthropic`'s internal API surface: we construct an
//! `AnthropicMessagesModel` with a custom URL hook (`{model}:rawPredict`
//! / `:streamRawPredict`) and body transform (strip `model`, inject
//! `anthropic_version: "vertex-2023-10-16"`).
//!
//! Per the upstream tool-allowlist, only a subset of Anthropic typed
//! tools are recognized by Vertex — see [`googleVertexAnthropicTools`]
//! re-exports below.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_anthropic::internal::{
    AnthropicMessagesModel, Inner as AnthropicInner, InnerBuilder as AnthropicInnerBuilder,
};
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    CallOptions, GenerateResult, LanguageModel, StreamResult, SupportedUrls,
};
use serde_json::Value;

use crate::PROVIDER_ID_ANTHROPIC;
use crate::auth::cloud_platform_token;
use crate::config::{VertexAuthMode, VertexInner};

const ANTHROPIC_VERTEX_VERSION: &str = "vertex-2023-10-16";

/// Vertex Anthropic sub-provider handle.
///
/// Mirrors the upstream `GoogleVertexAnthropicProvider`. Use
/// [`Self::language_model`] / [`Self::messages`] / [`Self::chat`] to
/// construct a Claude-on-Vertex chat model.
#[derive(Debug, Clone)]
pub struct GoogleVertexAnthropic {
    inner: Arc<VertexInner>,
}

impl GoogleVertexAnthropic {
    pub(crate) fn new(inner: Arc<VertexInner>) -> Self {
        Self { inner }
    }

    /// Construct a Claude-on-Vertex language model handle.
    #[must_use]
    pub fn language_model(
        &self,
        model_id: impl Into<String>,
    ) -> GoogleVertexAnthropicLanguageModel {
        GoogleVertexAnthropicLanguageModel::new(Arc::clone(&self.inner), model_id.into())
    }

    /// Alias of [`Self::language_model`] — matches upstream's `chat()`.
    #[must_use]
    pub fn chat(&self, model_id: impl Into<String>) -> GoogleVertexAnthropicLanguageModel {
        self.language_model(model_id)
    }

    /// Alias of [`Self::language_model`] — matches upstream's `messages()`.
    #[must_use]
    pub fn messages(&self, model_id: impl Into<String>) -> GoogleVertexAnthropicLanguageModel {
        self.language_model(model_id)
    }
}

/// Vertex Anthropic language-model handle.
///
/// Implements [`LanguageModel`] by delegating to a per-call
/// `AnthropicMessagesModel` so OAuth tokens can be minted just-in-time
/// without mutating shared state.
#[derive(Debug, Clone)]
pub struct GoogleVertexAnthropicLanguageModel {
    inner: Arc<VertexInner>,
    model_id: String,
}

impl GoogleVertexAnthropicLanguageModel {
    pub(crate) fn new(inner: Arc<VertexInner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    async fn delegate(&self) -> Result<AnthropicMessagesModel, ProviderError> {
        let inner = build_anthropic_inner(&self.inner).await?;
        Ok(AnthropicMessagesModel::new(
            Arc::new(inner),
            self.model_id.clone(),
        ))
    }
}

#[async_trait]
impl LanguageModel for GoogleVertexAnthropicLanguageModel {
    fn provider(&self) -> &str {
        PROVIDER_ID_ANTHROPIC
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn supported_urls(&self) -> SupportedUrls {
        // Vertex Anthropic does not support URL-based image sources; the
        // upstream returns `{}` so the caller is forced to download +
        // base64-encode any external image data.
        SupportedUrls::default()
    }

    async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult, ProviderError> {
        self.delegate().await?.do_generate(options).await
    }

    async fn do_stream(&self, options: CallOptions) -> Result<StreamResult, ProviderError> {
        self.delegate().await?.do_stream(options).await
    }
}

async fn build_anthropic_inner(state: &VertexInner) -> Result<AnthropicInner, ProviderError> {
    let base_url = state.publishers_anthropic_base();
    let mut builder: AnthropicInnerBuilder = AnthropicInner::builder()
        .base_url(base_url)
        .http_client(state.http.clone())
        .provider_name(PROVIDER_ID_ANTHROPIC)
        .endpoint(|base, model_id, is_streaming| {
            let suffix = if is_streaming {
                "streamRawPredict"
            } else {
                "rawPredict"
            };
            format!("{base}/models/{model_id}:{suffix}")
        })
        .body_transform(|body, _betas| {
            // Vertex puts model in the URL and fixes the API version; betas
            // ride along on the `anthropic-beta` header (default path), so
            // this transformer leaves them alone.
            if let Value::Object(map) = body {
                map.remove("model");
                map.insert(
                    "anthropic_version".into(),
                    Value::String(ANTHROPIC_VERTEX_VERSION.into()),
                );
            }
        });

    let mut headers: HashMap<String, Option<String>> = state.extra_headers.clone();
    match &state.auth {
        VertexAuthMode::Express { api_key } => {
            headers.insert("x-goog-api-key".into(), Some(api_key.clone()));
        }
        VertexAuthMode::OAuth { token_provider, .. } => {
            let token = cloud_platform_token(token_provider.as_ref()).await?;
            headers.insert("Authorization".into(), Some(format!("Bearer {token}")));
        }
    }
    // Vertex Anthropic does not accept the `anthropic-version` header —
    // we communicate the version via the `anthropic_version` body field
    // instead (injected by the body transform above).
    headers.insert("anthropic-version".into(), None);
    for (name, value) in headers {
        builder = builder.header(name, value);
    }
    builder.build()
}

/// Re-export the subset of Anthropic typed tools recognized by Vertex.
///
/// Mirrors `googleVertexAnthropicTools` in
/// `anthropic/google-vertex-anthropic-provider.ts`. Tools not listed
/// here are silently ignored by the upstream API; for forward
/// compatibility we still let the caller pass any
/// `llmsdk_anthropic::tools::*` tool through.
#[allow(
    deprecated,
    unused_imports,
    reason = "vertex retains text_editor_20250429 in its allowlist for parity; \
              re-exported factories are intentionally not called from this crate"
)]
pub mod tools {
    pub use llmsdk_anthropic::tools::{
        bash_20241022, bash_20250124, computer_20241022, text_editor_20241022,
        text_editor_20250124, text_editor_20250429, text_editor_20250728,
        tool_search_bm25_20251119, tool_search_regex_20251119, web_search_20250305,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GoogleVertex;

    #[tokio::test]
    async fn anthropic_handle_reports_vertex_provider() {
        let p = GoogleVertex::builder().api_key("k").build().await.unwrap();
        let a = p.anthropic().language_model("claude-opus-4-7");
        assert_eq!(a.provider(), PROVIDER_ID_ANTHROPIC);
        assert_eq!(a.model_id(), "claude-opus-4-7");
    }

    #[tokio::test]
    async fn anthropic_inner_url_routes_to_raw_predict() {
        // Express mode avoids the OAuth path so the URL composition
        // test does not need network / TLS access.
        let p = GoogleVertex::builder().api_key("k").build().await.unwrap();
        let inner = build_anthropic_inner(p.inner.as_ref()).await.unwrap();
        let url = inner.endpoint_url("claude-sonnet-4-5", false);
        assert!(url.contains(":rawPredict"));
        let stream_url = inner.endpoint_url("claude-sonnet-4-5", true);
        assert!(stream_url.contains(":streamRawPredict"));
    }

    #[tokio::test]
    async fn anthropic_body_transform_strips_model_and_injects_version() {
        let p = GoogleVertex::builder().api_key("k").build().await.unwrap();
        let inner = build_anthropic_inner(p.inner.as_ref()).await.unwrap();
        let mut body = serde_json::json!({"model": "claude-sonnet-4-5", "messages": []});
        let betas = std::collections::BTreeSet::new();
        inner.transform_body(&mut body, &betas);
        let obj = body.as_object().unwrap();
        assert!(obj.get("model").is_none());
        assert_eq!(
            obj.get("anthropic_version").and_then(|v| v.as_str()),
            Some(ANTHROPIC_VERTEX_VERSION)
        );
    }
}
