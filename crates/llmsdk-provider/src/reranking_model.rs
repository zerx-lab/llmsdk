//! Reranking model trait and supporting types.
//!
//! Mirrors `@ai-sdk/provider/src/reranking-model/v4/*`.
// Rust guideline compliant 2026-02-21

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::json::JsonObject;
use crate::shared::{Headers, ProviderMetadata, ProviderOptions, ResponseInfo, Warning};

/// Contract every reranking model implements.
///
/// Mirrors `RerankingModelV4`.
#[async_trait]
pub trait RerankingModel: Send + Sync + std::fmt::Debug {
    /// Provider id, e.g. `"cohere"`.
    fn provider(&self) -> &str;

    /// Provider-specific model id, e.g. `"rerank-english-v3.0"`.
    fn model_id(&self) -> &str;

    /// Specification version (currently `"v4"`).
    fn specification_version(&self) -> &'static str {
        "v4"
    }

    /// Rerank a list of documents against the given query.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::ProviderError`] when the upstream call fails or
    /// the response is malformed.
    async fn do_rerank(&self, options: RerankingOptions) -> Result<RerankingResult>;
}

/// Options for one [`RerankingModel::do_rerank`] call.
///
/// Mirrors `RerankingModelV4CallOptions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankingOptions {
    /// Documents to rerank.
    pub documents: RerankingDocuments,
    /// Query to rerank documents against.
    pub query: String,
    /// Limit returned documents to the top N.
    #[serde(default, rename = "topN", skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,
    /// Extra HTTP headers (HTTP providers only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<Headers>,
    /// Provider-specific options.
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_options: Option<ProviderOptions>,
}

/// Documents to rerank. Two-state tagged union over plain text or JSON objects.
///
/// Mirrors `RerankingModelV4CallOptions['documents']`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum RerankingDocuments {
    /// Plain-text documents.
    Text {
        /// Text values.
        values: Vec<String>,
    },
    /// JSON-object documents (Cohere passes them through with field filters).
    Object {
        /// Object values.
        values: Vec<JsonObject>,
    },
}

/// Result of [`RerankingModel::do_rerank`].
///
/// Mirrors `RerankingModelV4Result`.
#[derive(Debug, Clone)]
pub struct RerankingResult {
    /// Reranked documents (sorted by relevance descending).
    ///
    /// Each entry refers back to the document's index in the input list.
    pub ranking: Vec<RankingEntry>,
    /// Warnings for the call.
    pub warnings: Vec<Warning>,
    /// Provider-specific metadata.
    pub provider_metadata: Option<ProviderMetadata>,
    /// Optional response info (telemetry).
    pub response: Option<ResponseInfo>,
}

/// One entry in [`RerankingResult::ranking`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct RankingEntry {
    /// Index of the document in the input list before reranking.
    pub index: u32,
    /// Relevance score assigned by the model. Higher = more relevant.
    #[serde(rename = "relevanceScore")]
    pub relevance_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn options_roundtrip_text_documents() {
        let opts = RerankingOptions {
            documents: RerankingDocuments::Text {
                values: vec!["a".into(), "b".into()],
            },
            query: "q".into(),
            top_n: Some(3),
            headers: None,
            provider_options: None,
        };
        let j = serde_json::to_value(&opts).unwrap();
        assert_eq!(j["documents"]["type"], "text");
        assert_eq!(j["documents"]["values"][0], "a");
        assert_eq!(j["topN"], 3);
        let back: RerankingOptions = serde_json::from_value(j).unwrap();
        assert_eq!(back.top_n, Some(3));
    }

    #[test]
    fn documents_object_variant_kebab_tagged() {
        let docs = RerankingDocuments::Object {
            values: vec![json!({ "title": "x" }).as_object().cloned().unwrap()],
        };
        let j = serde_json::to_value(&docs).unwrap();
        assert_eq!(j["type"], "object");
    }

    #[test]
    fn ranking_entry_uses_camel_case_score() {
        let e = RankingEntry {
            index: 2,
            relevance_score: 0.87,
        };
        let j = serde_json::to_value(e).unwrap();
        assert_eq!(j["index"], 2);
        assert_eq!(j["relevanceScore"], 0.87);
    }
}
