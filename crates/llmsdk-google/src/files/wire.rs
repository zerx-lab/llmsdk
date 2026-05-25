//! Wire types for the Gemini Files API.
// Rust guideline compliant 2026-05-25

use serde::Deserialize;

/// `{ file: { ... } }` envelope returned by both the upload-finalize call
/// and the operation polling endpoint.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct UploadFileEnvelope {
    pub file: GoogleFileResource,
}

/// File resource representation.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct GoogleFileResource {
    pub name: String,
    #[serde(default, rename = "displayName")]
    pub display_name: Option<String>,
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    #[serde(default, rename = "sizeBytes")]
    pub size_bytes: Option<String>,
    #[serde(default, rename = "createTime")]
    pub create_time: Option<String>,
    #[serde(default, rename = "updateTime")]
    pub update_time: Option<String>,
    #[serde(default, rename = "expirationTime")]
    pub expiration_time: Option<String>,
    #[serde(default, rename = "sha256Hash")]
    pub sha256_hash: Option<String>,
    pub uri: String,
    pub state: String,
}
