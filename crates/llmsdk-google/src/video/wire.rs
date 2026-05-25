//! Wire types for Veo `:predictLongRunning` + operation polling.
// Rust guideline compliant 2026-05-25

use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
pub(crate) struct OperationResponse {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub done: Option<bool>,
    #[serde(default)]
    pub error: Option<OperationError>,
    #[serde(default)]
    pub response: Option<OperationPayload>,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct OperationError {
    #[serde(default)]
    #[allow(dead_code, reason = "kept for parity with upstream")]
    pub code: Option<i64>,
    pub message: String,
    #[serde(default)]
    #[allow(dead_code, reason = "kept for parity with upstream")]
    pub status: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct OperationPayload {
    #[serde(default, rename = "generateVideoResponse")]
    pub generate_video_response: Option<GenerateVideoResponse>,
}

#[derive(Debug, Default, Clone, Deserialize)]
pub(crate) struct GenerateVideoResponse {
    #[serde(default, rename = "generatedSamples")]
    pub generated_samples: Option<Vec<GeneratedSample>>,
}

#[derive(Debug, Default, Clone, Deserialize)]
pub(crate) struct GeneratedSample {
    #[serde(default)]
    pub video: Option<SampleVideo>,
}

#[derive(Debug, Default, Clone, Deserialize)]
pub(crate) struct SampleVideo {
    #[serde(default)]
    pub uri: Option<String>,
}
