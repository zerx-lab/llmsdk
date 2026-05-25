//! Gemini embeddings implementation.
//!
//! Mirrors `GoogleEmbeddingModel` from
//! `@ai-sdk/google/src/google-embedding-model.ts`. Routes single-value
//! calls to `:embedContent` and multi-value calls to `:batchEmbedContents`.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::embedding_model::{EmbedOptions, EmbedResult, EmbeddingModel};
use llmsdk_provider::shared::{Headers, ResponseInfo};
use llmsdk_provider_utils::http::{JsonRequest, post_json};
use serde_json::{Map, Value};

use crate::config::Inner;
use crate::error::rewrite_google_error;

use super::options::parse as parse_options;
use super::wire::{BatchEmbedResponse, SingleEmbedResponse};

/// Gemini embedding-model handle.
#[derive(Debug, Clone)]
pub struct GoogleEmbeddingModel {
    inner: Arc<Inner>,
    model_id: String,
}

impl GoogleEmbeddingModel {
    pub(crate) fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn merged_headers(&self, extra: Option<&Headers>) -> HashMap<String, Option<String>> {
        let mut h = self.inner.headers.clone();
        if let Some(extra) = extra {
            for (k, v) in extra {
                h.insert(k.clone(), v.clone());
            }
        }
        h
    }

    fn build_url(&self, method: &str) -> String {
        format!(
            "{}/models/{}:{}",
            self.inner.base_url, self.model_id, method
        )
    }
}

#[async_trait]
impl EmbeddingModel for GoogleEmbeddingModel {
    fn provider(&self) -> &str {
        &self.inner.provider
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_embeddings_per_call(&self) -> Option<u32> {
        Some(2048)
    }

    async fn supports_parallel_calls(&self) -> bool {
        true
    }

    async fn do_embed(&self, options: EmbedOptions) -> Result<EmbedResult, ProviderError> {
        let google_options = parse_options(options.provider_options.as_ref())?.unwrap_or_default();
        if options.values.len() > 2048 {
            return Err(ProviderError::too_many_embedding_values(
                2048,
                options.values.len(),
            ));
        }
        if let Some(content) = &google_options.content {
            if content.len() != options.values.len() {
                return Err(ProviderError::invalid_argument(
                    "providerOptions.google.content",
                    format!(
                        "must match the number of values ({}) but got {} entries",
                        options.values.len(),
                        content.len()
                    ),
                ));
            }
        }

        let headers = self.merged_headers(options.headers.as_ref());

        if options.values.len() == 1 {
            let parts = build_parts(
                &options.values[0],
                google_options.content.as_ref().and_then(|c| c[0].as_ref()),
            );
            let mut body = Map::new();
            body.insert(
                "model".into(),
                Value::String(format!("models/{}", self.model_id)),
            );
            let mut content_obj = Map::new();
            content_obj.insert("parts".into(), Value::Array(parts));
            body.insert("content".into(), Value::Object(content_obj));
            if let Some(d) = google_options.output_dimensionality {
                body.insert("outputDimensionality".into(), Value::from(d));
            }
            if let Some(t) = &google_options.task_type {
                body.insert("taskType".into(), Value::String(t.clone()));
            }

            let mut req = JsonRequest::new(self.build_url("embedContent"), Value::Object(body));
            req.headers = headers;
            let envelope = match post_json::<_, SingleEmbedResponse>(&self.inner.http, req).await {
                Ok(r) => r,
                Err(e) => return Err(rewrite_google_error(e)),
            };
            return Ok(EmbedResult {
                embeddings: vec![envelope.value.embedding.values],
                usage: None,
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
            });
        }

        // Batch endpoint.
        let mut requests: Vec<Value> = Vec::with_capacity(options.values.len());
        for (i, value) in options.values.iter().enumerate() {
            let extra = google_options.content.as_ref().and_then(|c| c[i].as_ref());
            let parts = build_parts(value, extra);
            let mut req_obj = Map::new();
            req_obj.insert(
                "model".into(),
                Value::String(format!("models/{}", self.model_id)),
            );
            let mut content_obj = Map::new();
            content_obj.insert("role".into(), Value::String("user".into()));
            content_obj.insert("parts".into(), Value::Array(parts));
            req_obj.insert("content".into(), Value::Object(content_obj));
            if let Some(d) = google_options.output_dimensionality {
                req_obj.insert("outputDimensionality".into(), Value::from(d));
            }
            if let Some(t) = &google_options.task_type {
                req_obj.insert("taskType".into(), Value::String(t.clone()));
            }
            requests.push(Value::Object(req_obj));
        }
        let mut body = Map::new();
        body.insert("requests".into(), Value::Array(requests));

        let mut req = JsonRequest::new(self.build_url("batchEmbedContents"), Value::Object(body));
        req.headers = headers;
        let envelope = match post_json::<_, BatchEmbedResponse>(&self.inner.http, req).await {
            Ok(r) => r,
            Err(e) => return Err(rewrite_google_error(e)),
        };

        Ok(EmbedResult {
            embeddings: envelope
                .value
                .embeddings
                .into_iter()
                .map(|e| e.values)
                .collect(),
            usage: None,
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

fn build_parts(text: &str, extra: Option<&Vec<Value>>) -> Vec<Value> {
    let mut parts: Vec<Value> = Vec::new();
    if !text.is_empty() {
        let mut t = Map::new();
        t.insert("text".into(), Value::String(text.to_owned()));
        parts.push(Value::Object(t));
    }
    if let Some(extra_parts) = extra {
        for p in extra_parts {
            parts.push(p.clone());
        }
    }
    if parts.is_empty() {
        // Always include at least one (empty) text part to match upstream.
        let mut t = Map::new();
        t.insert("text".into(), Value::String(String::new()));
        parts.push(Value::Object(t));
    }
    parts
}
