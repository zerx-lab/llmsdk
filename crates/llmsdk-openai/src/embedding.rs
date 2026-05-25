//! `OpenAI` text embedding model.
//!
//! Mirrors `@ai-sdk/openai/src/embedding/*`. The Embeddings API is much
//! simpler than Chat Completions: one POST, one JSON response, no
//! streaming.
// Rust guideline compliant 2026-02-21

use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::embedding_model::{
    EmbedOptions, EmbedResult, Embedding, EmbeddingModel, EmbeddingUsage,
};
use llmsdk_provider::shared::{RequestInfo, ResponseInfo};
use llmsdk_provider_utils::http::{JsonRequest, post_json};
use serde::{Deserialize, Serialize};

use crate::PROVIDER_ID;
use crate::config::Inner;

/// Default `maxEmbeddingsPerCall` reported by `OpenAI` (matches ai-sdk).
const MAX_PER_CALL: u32 = 2048;

/// `OpenAI` Embeddings model handle.
///
/// Cheap to clone — shares the provider's HTTP client and auth state.
#[derive(Debug, Clone)]
pub struct OpenAiEmbeddingModel {
    inner: Arc<Inner>,
    model_id: String,
}

impl OpenAiEmbeddingModel {
    /// Construct from a fully assembled [`Inner`].
    ///
    /// Public for cross-crate composition (Azure `OpenAI`). End-users should
    /// prefer [`crate::OpenAi::embedding`].
    #[must_use]
    pub fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn endpoint(&self) -> String {
        self.inner.endpoint("/embeddings", &self.model_id)
    }
}

#[async_trait]
impl EmbeddingModel for OpenAiEmbeddingModel {
    fn provider(&self) -> &str {
        self.inner.provider_id()
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_embeddings_per_call(&self) -> Option<u32> {
        Some(MAX_PER_CALL)
    }

    async fn supports_parallel_calls(&self) -> bool {
        true
    }

    async fn do_embed(&self, options: EmbedOptions) -> Result<EmbedResult, ProviderError> {
        let total = options.values.len();
        if u32::try_from(total).is_ok_and(|n| n > MAX_PER_CALL) {
            return Err(ProviderError::too_many_embedding_values(
                MAX_PER_CALL as usize,
                total,
            ));
        }

        let dimensions = options
            .provider_options
            .as_ref()
            .and_then(|p| p.get(PROVIDER_ID))
            .and_then(|m| m.get("dimensions"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| u32::try_from(n).ok());

        let user = options
            .provider_options
            .as_ref()
            .and_then(|p| p.get(PROVIDER_ID))
            .and_then(|m| m.get("user"))
            .and_then(serde_json::Value::as_str)
            .map(std::string::ToString::to_string);

        let request = EmbeddingRequest {
            model: self.model_id.clone(),
            input: options.values,
            encoding_format: "float",
            dimensions,
            user,
        };

        let request_body_value = serde_json::to_value(&request).ok();

        let mut request_headers = self.inner.headers.clone();
        if let Some(headers) = options.headers {
            for (name, value) in headers {
                request_headers.insert(name, value);
            }
        }

        let mut http_request = JsonRequest::new(self.endpoint(), request);
        http_request.headers = request_headers;

        let response = match post_json::<_, EmbeddingResponse>(&self.inner.http, http_request).await
        {
            Ok(r) => r,
            Err(err) => return Err(crate::error::rewrite_openai_error(err)),
        };

        let embeddings: Vec<Embedding> = response
            .value
            .data
            .into_iter()
            .map(|d| d.embedding)
            .collect();
        let usage = response.value.usage.map(|u| EmbeddingUsage {
            tokens: Some(u.prompt_tokens),
        });

        Ok(EmbedResult {
            embeddings,
            usage,
            provider_metadata: None,
            request: Some(RequestInfo {
                body: request_body_value,
            }),
            response: Some(ResponseInfo {
                headers: Some(
                    response
                        .headers
                        .into_iter()
                        .map(|(k, v)| (k, Some(v)))
                        .collect(),
                ),
                ..ResponseInfo::default()
            }),
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
    encoding_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    #[serde(default)]
    usage: Option<EmbeddingResponseUsage>,
}

#[derive(Debug, Clone, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
struct EmbeddingResponseUsage {
    prompt_tokens: u64,
}
