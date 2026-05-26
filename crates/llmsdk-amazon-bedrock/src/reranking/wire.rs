//! On-wire types for the Bedrock Rerank API.
// Rust guideline compliant 2026-05-25

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// `POST /rerank` request body.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct RerankRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "nextToken")]
    pub next_token: Option<String>,
    pub queries: Vec<RerankQuery>,
    #[serde(rename = "rerankingConfiguration")]
    pub reranking_configuration: RerankingConfiguration,
    pub sources: Vec<RerankSource>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RerankQuery {
    #[serde(rename = "type")]
    pub kind: &'static str,
    #[serde(rename = "textQuery")]
    pub text_query: TextQuery,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TextQuery {
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RerankingConfiguration {
    #[serde(rename = "type")]
    pub kind: &'static str,
    // AWS REST API requires the `amazonBedrockRerankingConfiguration` key —
    // see `bedrock-runtime-api Rerank` reference and upstream
    // `amazon-bedrock-reranking-api.ts:10`. The shorter `bedrockRerankingConfiguration`
    // is rejected by the service with a `ValidationException`.
    #[serde(rename = "amazonBedrockRerankingConfiguration")]
    pub bedrock_reranking_configuration: BedrockRerankingConfiguration,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct BedrockRerankingConfiguration {
    #[serde(rename = "modelConfiguration")]
    pub model_configuration: ModelConfiguration,
    #[serde(rename = "numberOfResults", skip_serializing_if = "Option::is_none")]
    pub number_of_results: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ModelConfiguration {
    #[serde(rename = "modelArn")]
    pub model_arn: String,
    #[serde(
        rename = "additionalModelRequestFields",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_model_request_fields: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RerankSource {
    #[serde(rename = "type")]
    pub kind: &'static str,
    #[serde(rename = "inlineDocumentSource")]
    pub inline_document_source: InlineDocumentSource,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum InlineDocumentSource {
    Text {
        #[serde(rename = "type")]
        kind: &'static str,
        #[serde(rename = "textDocument")]
        text_document: TextDocument,
    },
    Json {
        #[serde(rename = "type")]
        kind: &'static str,
        #[serde(rename = "jsonDocument")]
        json_document: Value,
    },
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TextDocument {
    pub text: String,
}

/// Response body shape.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RerankResponse {
    pub results: Vec<RerankResponseResult>,
    #[serde(default, rename = "nextToken")]
    pub _next_token: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) struct RerankResponseResult {
    pub index: u32,
    #[serde(rename = "relevanceScore")]
    pub relevance_score: f64,
}
