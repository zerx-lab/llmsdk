//! Response schema for Bedrock image generation.
// Rust guideline compliant 2026-05-25

use serde::Deserialize;
use serde_json::Value;

/// Bedrock image-generation response (Nova Canvas / Titan / SDXL share a
/// superset of the same shape).
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ImageResponse {
    /// Base64-encoded images (each entry is one PNG).
    #[serde(default)]
    pub images: Option<Vec<String>>,
    /// Status field for moderated requests.
    #[serde(default)]
    pub status: Option<String>,
    /// Details object (contains `"Moderation Reasons"`).
    #[serde(default)]
    pub details: Option<Value>,
}
