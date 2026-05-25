//! Vertex AI embeddings (`:predict` with `instances[]` + `parameters`).
//!
//! Mirrors `google-vertex-embedding-model.ts`. Vertex's embedding wire is
//! **not** the same as the public Gemini Generative Language API
//! (`batchEmbedContents`); it uses the Vertex prediction protocol.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::embedding_model::{EmbedOptions, EmbedResult, EmbeddingModel, EmbeddingUsage};
use llmsdk_provider::shared::{ProviderOptions, ResponseInfo};
use llmsdk_provider_utils::http::{JsonRequest, post_json};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::PROVIDER_ID_EMBEDDING;
use crate::auth::cloud_platform_token;
use crate::config::{VertexAuthMode, VertexInner};

/// Maximum inputs per Vertex embedding call (matches upstream).
pub const MAX_EMBEDDINGS_PER_CALL: u32 = 2048;

/// Vertex embedding-model handle.
#[derive(Debug, Clone)]
pub struct GoogleVertexEmbeddingModel {
    inner: Arc<VertexInner>,
    model_id: String,
}

impl GoogleVertexEmbeddingModel {
    pub(crate) fn new(inner: Arc<VertexInner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    async fn merged_headers(
        &self,
        per_call: Option<&llmsdk_provider::shared::Headers>,
    ) -> Result<HashMap<String, Option<String>>, ProviderError> {
        let mut headers = self.inner.extra_headers.clone();
        match &self.inner.auth {
            VertexAuthMode::Express { api_key } => {
                headers.insert("x-goog-api-key".into(), Some(api_key.clone()));
            }
            VertexAuthMode::OAuth { token_provider, .. } => {
                let token = cloud_platform_token(token_provider.as_ref()).await?;
                headers.insert("Authorization".into(), Some(format!("Bearer {token}")));
            }
        }
        if let Some(h) = per_call {
            for (k, v) in h {
                headers.insert(k.clone(), v.clone());
            }
        }
        Ok(headers)
    }
}

#[async_trait]
impl EmbeddingModel for GoogleVertexEmbeddingModel {
    fn provider(&self) -> &str {
        PROVIDER_ID_EMBEDDING
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_embeddings_per_call(&self) -> Option<u32> {
        Some(MAX_EMBEDDINGS_PER_CALL)
    }

    async fn supports_parallel_calls(&self) -> bool {
        true
    }

    async fn do_embed(&self, options: EmbedOptions) -> Result<EmbedResult, ProviderError> {
        if options.values.len() > MAX_EMBEDDINGS_PER_CALL as usize {
            return Err(ProviderError::too_many_embedding_values(
                MAX_EMBEDDINGS_PER_CALL as usize,
                options.values.len(),
            ));
        }
        let vertex_options =
            parse_embedding_options(options.provider_options.as_ref())?.unwrap_or_default();

        let mut instances: Vec<Value> = Vec::with_capacity(options.values.len());
        for value in &options.values {
            let mut instance = Map::new();
            instance.insert("content".into(), Value::String(value.clone()));
            if let Some(t) = &vertex_options.task_type {
                instance.insert("task_type".into(), Value::String(t.clone()));
            }
            if let Some(t) = &vertex_options.title {
                instance.insert("title".into(), Value::String(t.clone()));
            }
            instances.push(Value::Object(instance));
        }

        let mut parameters = Map::new();
        if let Some(d) = vertex_options.output_dimensionality {
            parameters.insert("outputDimensionality".into(), Value::from(d));
        }
        if let Some(at) = vertex_options.auto_truncate {
            parameters.insert("autoTruncate".into(), Value::Bool(at));
        }

        let mut body = Map::new();
        body.insert("instances".into(), Value::Array(instances));
        if !parameters.is_empty() {
            body.insert("parameters".into(), Value::Object(parameters));
        }

        let url = format!(
            "{}/models/{}:predict",
            self.inner.publishers_google_base(),
            self.model_id
        );
        let headers = self.merged_headers(options.headers.as_ref()).await?;
        let mut req = JsonRequest::new(url, Value::Object(body));
        req.headers = headers;

        let envelope = post_json::<_, VertexEmbedResponse>(&self.inner.http, req).await?;

        let mut total_tokens: u64 = 0;
        let mut embeddings = Vec::with_capacity(envelope.value.predictions.len());
        for pred in envelope.value.predictions {
            total_tokens =
                total_tokens.saturating_add(pred.embeddings.statistics.token_count.unwrap_or(0));
            embeddings.push(pred.embeddings.values);
        }

        Ok(EmbedResult {
            embeddings,
            usage: Some(EmbeddingUsage {
                tokens: Some(total_tokens),
            }),
            provider_metadata: None,
            request: None,
            response: Some(ResponseInfo {
                headers: Some(
                    envelope
                        .headers
                        .into_iter()
                        .map(|(k, v)| (k, Some(v)))
                        .collect(),
                ),
                ..Default::default()
            }),
        })
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct VertexEmbeddingOptions {
    #[serde(default, rename = "outputDimensionality")]
    output_dimensionality: Option<u32>,
    #[serde(default, rename = "taskType")]
    task_type: Option<String>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default, rename = "autoTruncate")]
    auto_truncate: Option<bool>,
}

/// Parse `provider_options` under any of the recognized Vertex keys
/// (`googleVertex`, `vertex`, `google`). Mirrors upstream's three-key
/// fallback for backward compatibility.
fn parse_embedding_options(
    provider_options: Option<&ProviderOptions>,
) -> Result<Option<VertexEmbeddingOptions>, ProviderError> {
    let Some(opts) = provider_options else {
        return Ok(None);
    };
    for key in ["googleVertex", "vertex", "google"] {
        if let Some(payload) = opts.get(key) {
            let value = Value::Object(payload.clone());
            let parsed: VertexEmbeddingOptions =
                serde_json::from_value(value.clone()).map_err(|e| {
                    ProviderError::type_validation(
                        format!("provider_options.{key}"),
                        value,
                        e.to_string(),
                    )
                })?;
            return Ok(Some(parsed));
        }
    }
    Ok(None)
}

#[derive(Debug, Clone, Deserialize)]
struct VertexEmbedResponse {
    predictions: Vec<VertexEmbeddingPrediction>,
}

#[derive(Debug, Clone, Deserialize)]
struct VertexEmbeddingPrediction {
    embeddings: VertexEmbeddingPayload,
}

#[derive(Debug, Clone, Deserialize)]
struct VertexEmbeddingPayload {
    values: Vec<f32>,
    #[serde(default)]
    statistics: VertexEmbeddingStatistics,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct VertexEmbeddingStatistics {
    #[serde(default, rename = "token_count")]
    token_count: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_options_googlevertex_key() {
        let mut opts = ProviderOptions::new();
        opts.insert(
            "googleVertex".into(),
            json!({"taskType": "SEMANTIC_SIMILARITY", "outputDimensionality": 64})
                .as_object()
                .cloned()
                .unwrap(),
        );
        let parsed = parse_embedding_options(Some(&opts)).unwrap().unwrap();
        assert_eq!(parsed.task_type.as_deref(), Some("SEMANTIC_SIMILARITY"));
        assert_eq!(parsed.output_dimensionality, Some(64));
    }

    #[test]
    fn parse_options_legacy_vertex_key() {
        let mut opts = ProviderOptions::new();
        opts.insert(
            "vertex".into(),
            json!({"autoTruncate": false}).as_object().cloned().unwrap(),
        );
        let parsed = parse_embedding_options(Some(&opts)).unwrap().unwrap();
        assert_eq!(parsed.auto_truncate, Some(false));
    }

    #[test]
    fn parse_options_legacy_google_key() {
        let mut opts = ProviderOptions::new();
        opts.insert(
            "google".into(),
            json!({"title": "doc"}).as_object().cloned().unwrap(),
        );
        let parsed = parse_embedding_options(Some(&opts)).unwrap().unwrap();
        assert_eq!(parsed.title.as_deref(), Some("doc"));
    }
}
