//! [`EmbeddingModel`] implementation for Bedrock embedding families.
// Rust guideline compliant 2026-05-25

use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::embedding_model::{
    EmbedOptions, EmbedResult, Embedding, EmbeddingModel, EmbeddingUsage,
};
use llmsdk_provider::shared::ResponseInfo;
use llmsdk_provider_utils::http::{RawRequest, post_raw};
use reqwest::Method;
use serde::Serialize;

use super::options::parse as parse_options;
use super::wire::{
    CohereRequest, EmbeddingResponse, NovaParams, NovaRequest, NovaTextParam, TitanRequest,
};
use crate::PROVIDER_ID;
use crate::config::Inner;

/// Bedrock embedding model handle.
#[derive(Debug, Clone)]
pub struct AmazonBedrockEmbeddingModel {
    inner: Arc<Inner>,
    model_id: String,
}

impl AmazonBedrockEmbeddingModel {
    pub(crate) fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn url(&self) -> String {
        let encoded = crate::chat::encode_path_segment(&self.model_id);
        format!("{}/model/{}/invoke", self.inner.runtime_base_url, encoded)
    }

    fn is_nova(&self) -> bool {
        self.model_id.starts_with("amazon.nova-") && self.model_id.contains("embed")
    }

    fn is_cohere(&self) -> bool {
        self.model_id.starts_with("cohere.embed-")
    }
}

#[async_trait]
impl EmbeddingModel for AmazonBedrockEmbeddingModel {
    fn provider(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_embeddings_per_call(&self) -> Option<u32> {
        Some(1)
    }

    async fn do_embed(&self, options: EmbedOptions) -> Result<EmbedResult, ProviderError> {
        if options.values.is_empty() {
            return Err(ProviderError::invalid_argument(
                "values",
                "Amazon Bedrock embedding models require at least one input value.",
            ));
        }
        if options.values.len() > 1 {
            return Err(ProviderError::invalid_argument(
                "values",
                "Amazon Bedrock embedding models accept at most one value per call.",
            ));
        }
        let value = &options.values[0];
        let parsed = parse_options(options.provider_options.as_ref());

        let body_bytes = if self.is_nova() {
            let req = NovaRequest {
                task_type: "SINGLE_EMBEDDING",
                single_embedding_params: NovaParams {
                    embedding_purpose: parsed
                        .embedding_purpose
                        .as_deref()
                        .unwrap_or("GENERIC_INDEX"),
                    embedding_dimension: parsed.embedding_dimension.unwrap_or(1024),
                    text: NovaTextParam {
                        truncation_mode: parsed.truncate.as_deref().unwrap_or("END"),
                        value,
                    },
                },
            };
            serialize_body(&req)?
        } else if self.is_cohere() {
            let req = CohereRequest {
                input_type: parsed.input_type.as_deref().unwrap_or("search_query"),
                texts: vec![value.as_str()],
                truncate: parsed.truncate.as_deref(),
                output_dimension: parsed.output_dimension,
            };
            serialize_body(&req)?
        } else {
            let req = TitanRequest {
                input_text: value,
                dimensions: parsed.dimensions,
                normalize: parsed.normalize,
            };
            serialize_body(&req)?
        };

        let url = self.url();
        let mut headers = self.inner.extra_headers.clone();
        if let Some(per_call) = options.headers.as_ref() {
            for (k, v) in per_call {
                headers.insert(k.clone(), v.clone());
            }
        }
        self.inner
            .auth
            .apply(&mut headers, &Method::POST, &url, &body_bytes)
            .await?;

        let mut raw = RawRequest::new(url.clone(), body_bytes, "application/json");
        raw.headers = headers;
        let response = post_raw::<EmbeddingResponse>(&self.inner.http, raw).await?;
        let (embedding, tokens): (Embedding, Option<u64>) = match response.value {
            EmbeddingResponse::Titan {
                embedding,
                input_text_token_count,
            } => (embedding, Some(input_text_token_count)),
            EmbeddingResponse::Nova {
                mut embeddings,
                input_token_count,
            } => {
                let first = embeddings.pop().ok_or_else(|| {
                    ProviderError::json_parse(
                        "<bedrock-embed-response>",
                        "Nova response missing embeddings[]".to_owned(),
                    )
                })?;
                (first.embedding, input_token_count)
            }
            EmbeddingResponse::CohereV3 { mut embeddings } => {
                let vec = embeddings.pop().ok_or_else(|| {
                    ProviderError::json_parse(
                        "<bedrock-embed-response>",
                        "Cohere v3 response missing embeddings[]".to_owned(),
                    )
                })?;
                (vec, None)
            }
            EmbeddingResponse::CohereV4 { embeddings } => {
                let mut floats = embeddings.float;
                let vec = floats.pop().ok_or_else(|| {
                    ProviderError::json_parse(
                        "<bedrock-embed-response>",
                        "Cohere v4 response missing float[]".to_owned(),
                    )
                })?;
                (vec, None)
            }
        };

        Ok(EmbedResult {
            embeddings: vec![embedding],
            usage: tokens.map(|t| EmbeddingUsage { tokens: Some(t) }),
            provider_metadata: None,
            request: None,
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

fn serialize_body<T: Serialize>(body: &T) -> Result<Vec<u8>, ProviderError> {
    serde_json::to_vec(body)
        .map_err(|e| ProviderError::json_parse("<bedrock-embed-request>", e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> Arc<Inner> {
        let bedrock = crate::AmazonBedrock::builder()
            .api_key("test-bearer")
            .region("us-east-1")
            .build()
            .unwrap();
        bedrock.inner
    }

    #[tokio::test]
    async fn empty_values_errors() {
        let model =
            AmazonBedrockEmbeddingModel::new(provider(), "amazon.titan-embed-text-v2:0".to_owned());
        let err = model
            .do_embed(EmbedOptions {
                values: vec![],
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("at least one"));
    }

    #[test]
    fn nova_dispatch_matches_prefix() {
        let model =
            AmazonBedrockEmbeddingModel::new(provider(), "amazon.nova-1-embed-v1:0".to_owned());
        assert!(model.is_nova());
        assert!(!model.is_cohere());
    }

    #[test]
    fn cohere_dispatch_matches_prefix() {
        let model =
            AmazonBedrockEmbeddingModel::new(provider(), "cohere.embed-english-v3".to_owned());
        assert!(model.is_cohere());
        assert!(!model.is_nova());
    }
}
