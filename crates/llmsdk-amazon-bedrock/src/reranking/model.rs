//! [`RerankingModel`] implementation for the Bedrock Rerank API.
// Rust guideline compliant 2026-05-25

use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::reranking_model::{
    RankingEntry, RerankingDocuments, RerankingModel, RerankingOptions, RerankingResult,
};
use llmsdk_provider::shared::{ResponseInfo, Warning};
use llmsdk_provider_utils::http::{RawRequest, post_raw};
use reqwest::Method;
use serde_json::Value;

use super::options::parse as parse_options;
use super::wire::{
    BedrockRerankingConfiguration, InlineDocumentSource, ModelConfiguration, RerankQuery,
    RerankRequest, RerankResponse, RerankSource, RerankingConfiguration, TextDocument, TextQuery,
};
use crate::PROVIDER_ID;
use crate::config::Inner;

/// Bedrock reranking model handle.
#[derive(Debug, Clone)]
pub struct AmazonBedrockRerankingModel {
    inner: Arc<Inner>,
    model_id: String,
}

impl AmazonBedrockRerankingModel {
    pub(crate) fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn url(&self) -> String {
        format!("{}/rerank", self.inner.agent_runtime_base_url)
    }

    fn model_arn(&self) -> String {
        format!(
            "arn:aws:bedrock:{}::foundation-model/{}",
            self.inner.region, self.model_id
        )
    }
}

#[async_trait]
impl RerankingModel for AmazonBedrockRerankingModel {
    fn provider(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn do_rerank(&self, options: RerankingOptions) -> Result<RerankingResult, ProviderError> {
        let parsed = parse_options(options.provider_options.as_ref());

        let sources = match &options.documents {
            RerankingDocuments::Text { values } => values
                .iter()
                .map(|text| RerankSource {
                    kind: "INLINE",
                    inline_document_source: InlineDocumentSource::Text {
                        kind: "TEXT",
                        text_document: TextDocument { text: text.clone() },
                    },
                })
                .collect(),
            RerankingDocuments::Object { values } => values
                .iter()
                .map(|obj| RerankSource {
                    kind: "INLINE",
                    inline_document_source: InlineDocumentSource::Json {
                        kind: "JSON",
                        json_document: Value::Object(obj.clone()),
                    },
                })
                .collect(),
        };

        let body = RerankRequest {
            next_token: parsed.next_token.clone(),
            queries: vec![RerankQuery {
                kind: "TEXT",
                text_query: TextQuery {
                    text: options.query.clone(),
                },
            }],
            reranking_configuration: RerankingConfiguration {
                kind: "BEDROCK_RERANKING_MODEL",
                bedrock_reranking_configuration: BedrockRerankingConfiguration {
                    model_configuration: ModelConfiguration {
                        model_arn: self.model_arn(),
                        additional_model_request_fields: parsed.additional_model_request_fields,
                    },
                    number_of_results: options.top_n,
                },
            },
            sources,
        };

        let body_bytes = serde_json::to_vec(&body)
            .map_err(|e| ProviderError::json_parse("<bedrock-rerank-request>", e.to_string()))?;
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

        let mut raw = RawRequest::new(url, body_bytes, "application/json");
        raw.headers = headers;
        let response = post_raw::<RerankResponse>(&self.inner.http, raw).await?;
        let value = response.value;

        let ranking = value
            .results
            .into_iter()
            .map(|r| RankingEntry {
                index: r.index,
                relevance_score: r.relevance_score,
            })
            .collect();

        Ok(RerankingResult {
            ranking,
            warnings: Vec::<Warning>::new(),
            provider_metadata: None,
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
