//! Imagen `:predict` wire types.
// Rust guideline compliant 2026-05-25

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(crate) struct ImagenPredictResponse {
    #[serde(default)]
    pub predictions: Vec<ImagenPrediction>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ImagenPrediction {
    #[serde(rename = "bytesBase64Encoded")]
    pub bytes_base64_encoded: String,
    #[serde(default, rename = "mimeType")]
    pub mime_type: Option<String>,
}
