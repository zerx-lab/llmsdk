//! [`EmbeddingModel`] implementation for Mistral.
//!
//! Mirrors `mistral-embedding-model.ts`. Entry: [`MistralEmbeddingModel::new`]
//! via [`crate::Mistral::embedding`].
// Rust guideline compliant 2026-05-25

use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::embedding_model::{
    EmbedOptions, EmbedResult, Embedding, EmbeddingModel, EmbeddingUsage,
};
use llmsdk_provider::shared::{RequestInfo, ResponseInfo};
use llmsdk_provider_utils::http::{JsonRequest, post_json};

use crate::PROVIDER_ID;
use crate::config::Inner;

use super::options::parse as parse_options;
use super::wire::{EmbeddingRequest, EmbeddingResponse};

/// `maxEmbeddingsPerCall` reported by Mistral (matches ai-sdk).
const MAX_PER_CALL: u32 = 32;

/// Mistral Embeddings model handle.
///
/// Cheap to clone — shares the provider's HTTP client and auth state.
#[derive(Debug, Clone)]
pub struct MistralEmbeddingModel {
    inner: Arc<Inner>,
    model_id: String,
}

impl MistralEmbeddingModel {
    pub(crate) fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn endpoint(&self) -> String {
        format!("{}/embeddings", self.inner.base_url)
    }
}

#[async_trait]
impl EmbeddingModel for MistralEmbeddingModel {
    fn provider(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_embeddings_per_call(&self) -> Option<u32> {
        Some(MAX_PER_CALL)
    }

    async fn supports_parallel_calls(&self) -> bool {
        false
    }

    async fn do_embed(&self, options: EmbedOptions) -> Result<EmbedResult, ProviderError> {
        let total = options.values.len();
        if u32::try_from(total).is_ok_and(|n| n > MAX_PER_CALL) {
            return Err(ProviderError::too_many_embedding_values(
                MAX_PER_CALL as usize,
                total,
            ));
        }

        let mistral_opts = parse_options(options.provider_options.as_ref());

        let request = EmbeddingRequest {
            model: self.model_id.clone(),
            input: options.values,
            encoding_format: "float",
            output_dimension: mistral_opts.output_dimension,
            output_dtype: mistral_opts.output_dtype,
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

        let response = post_json::<_, EmbeddingResponse>(&self.inner.http, http_request).await?;

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
