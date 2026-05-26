//! [`EmbeddingModel`] implementation for Cohere v2 embeddings.
//!
//! Mirrors `cohere-embedding-model.ts`.
// Rust guideline compliant 2026-05-25

use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::embedding_model::{
    EmbedOptions, EmbedResult, Embedding, EmbeddingModel, EmbeddingUsage,
};
use llmsdk_provider::shared::{ProviderMetadata, RequestInfo, ResponseInfo};
use llmsdk_provider_utils::http::{JsonRequest, post_json};

use crate::PROVIDER_ID;
use crate::config::Inner;

use super::options::parse as parse_options;
use super::wire::{EmbedRequest, EmbedResponse};

/// Maximum inputs per call documented by Cohere — matches ai-sdk's constant.
const MAX_PER_CALL: u32 = 96;

/// Cohere v2 Embeddings model handle.
///
/// Cheap to clone; shares the parent provider's HTTP client.
#[derive(Debug, Clone)]
pub struct CohereEmbeddingModel {
    inner: Arc<Inner>,
    model_id: String,
}

impl CohereEmbeddingModel {
    pub(crate) fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn endpoint(&self) -> String {
        format!("{}/embed", self.inner.base_url)
    }
}

#[async_trait]
impl EmbeddingModel for CohereEmbeddingModel {
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

        let parsed = parse_options(options.provider_options.as_ref());

        // ai-sdk hard-codes `embedding_types: ['float']` because non-float
        // payloads (int8 / uint8 / binary / ubinary) aren't surfaced through
        // EmbedResult — see the matching comment in
        // cohere-embedding-model.ts. We mirror that and ignore any
        // `provider_options.cohere.embeddingTypes` override silently.
        let _ = parsed.embedding_types.as_ref();
        let embedding_types = vec!["float".to_owned()];

        let request = EmbedRequest {
            model: self.model_id.clone(),
            texts: options.values,
            embedding_types,
            input_type: parsed
                .input_type
                .clone()
                .unwrap_or_else(|| "search_query".to_owned()),
            truncate: parsed.truncate.clone(),
            output_dimension: parsed.output_dimension,
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

        let response = post_json::<_, EmbedResponse>(&self.inner.http, http_request).await?;
        let value = response.value;

        let float_vectors = value.embeddings.float.clone().unwrap_or_default();
        let embeddings: Vec<Embedding> = float_vectors;

        let usage = value
            .meta
            .as_ref()
            .and_then(|m| m.billed_units.as_ref())
            .and_then(|b| b.input_tokens)
            .map(|tokens| EmbeddingUsage {
                tokens: Some(tokens),
            });

        let provider_metadata = build_provider_metadata(&value);

        Ok(EmbedResult {
            embeddings,
            usage,
            provider_metadata,
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

/// Surface non-float embedding flavors via `provider_metadata.cohere.embeddings`.
fn build_provider_metadata(response: &EmbedResponse) -> Option<ProviderMetadata> {
    let mut entries = serde_json::Map::new();
    if let Some(v) = &response.embeddings.int8 {
        entries.insert("int8".into(), serde_json::to_value(v).ok()?);
    }
    if let Some(v) = &response.embeddings.uint8 {
        entries.insert("uint8".into(), serde_json::to_value(v).ok()?);
    }
    if let Some(v) = &response.embeddings.binary {
        entries.insert("binary".into(), serde_json::to_value(v).ok()?);
    }
    if let Some(v) = &response.embeddings.ubinary {
        entries.insert("ubinary".into(), serde_json::to_value(v).ok()?);
    }
    if entries.is_empty() {
        return None;
    }
    let mut cohere = serde_json::Map::new();
    cohere.insert("embeddings".into(), serde_json::Value::Object(entries));
    let mut metadata = ProviderMetadata::new();
    metadata.insert("cohere".into(), cohere);
    Some(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn rejects_too_many_inputs() {
        let inner = Arc::new(Inner {
            base_url: "http://localhost".into(),
            headers: std::collections::HashMap::new(),
            http: llmsdk_provider_utils::http::HttpClient::new().expect("http"),
        });
        let model = CohereEmbeddingModel::new(inner, "embed-english-v3.0".into());

        let too_many: Vec<String> = (0..97).map(|i| format!("doc-{i}")).collect();
        let err = model
            .do_embed(EmbedOptions {
                values: too_many,
                headers: None,
                provider_options: None,
            })
            .await
            .expect_err("must error");
        assert!(format!("{err}").contains("96"));
    }

    #[test]
    fn provider_metadata_surfaces_int8_branch() {
        let response = EmbedResponse {
            embeddings: super::super::wire::EmbedResponseVectors {
                float: None,
                int8: Some(vec![vec![1, 2, 3]]),
                uint8: None,
                binary: None,
                ubinary: None,
            },
            meta: None,
        };
        let pm = build_provider_metadata(&response).unwrap();
        let cohere = pm.get("cohere").unwrap();
        assert_eq!(cohere["embeddings"]["int8"], json!([[1, 2, 3]]));
    }
}
