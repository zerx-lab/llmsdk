//! On-wire request / response shapes for the three embedding families.
// Rust guideline compliant 2026-05-25

use serde::{Deserialize, Serialize};

// ----- request bodies (one per family) -----

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TitanRequest<'a> {
    /// Input text.
    #[serde(rename = "inputText")]
    pub input_text: &'a str,
    /// Optional output dimension.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// Normalize output embeddings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct NovaRequest<'a> {
    /// Always `"SINGLE_EMBEDDING"`.
    #[serde(rename = "taskType")]
    pub task_type: &'static str,
    /// Single-embedding parameters.
    #[serde(rename = "singleEmbeddingParams")]
    pub single_embedding_params: NovaParams<'a>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct NovaParams<'a> {
    #[serde(rename = "embeddingPurpose")]
    pub embedding_purpose: &'a str,
    #[serde(rename = "embeddingDimension")]
    pub embedding_dimension: u32,
    pub text: NovaTextParam<'a>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct NovaTextParam<'a> {
    #[serde(rename = "truncationMode")]
    pub truncation_mode: &'a str,
    pub value: &'a str,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CohereRequest<'a> {
    pub input_type: &'a str,
    pub texts: Vec<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimension: Option<u32>,
}

// ----- response shapes (untagged union over all four) -----

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum EmbeddingResponse {
    /// Titan-style `{ embedding, inputTextTokenCount }`.
    Titan {
        /// Single embedding.
        embedding: Vec<f32>,
        /// Token count.
        #[serde(rename = "inputTextTokenCount")]
        input_text_token_count: u64,
    },
    /// Nova-style `{ embeddings: [{ embeddingType, embedding }], inputTokenCount? }`.
    Nova {
        /// Single-element array of Nova embedding entries.
        embeddings: Vec<NovaEmbedding>,
        /// Optional token count.
        #[serde(default, rename = "inputTokenCount")]
        input_token_count: Option<u64>,
    },
    /// Cohere v3-style `{ embeddings: [[f32; N]] }`.
    CohereV3 {
        /// Embeddings array (single element after our `[values[0]]` send).
        embeddings: Vec<Vec<f32>>,
    },
    /// Cohere v4-style `{ embeddings: { float: [[f32; N]] } }`.
    CohereV4 {
        /// `{ float: [...] }` wrapper.
        embeddings: CohereV4Embeddings,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct NovaEmbedding {
    /// Always `"float"` upstream.
    #[serde(rename = "embeddingType")]
    pub _embedding_type: String,
    /// The actual vector.
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct CohereV4Embeddings {
    /// Float vectors.
    pub float: Vec<Vec<f32>>,
}
