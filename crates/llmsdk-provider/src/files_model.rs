//! File upload model trait and supporting types.
//!
//! Mirrors `@ai-sdk/provider/src/files/v4/*`. Implementations let callers
//! upload a file once and receive a [`ProviderReference`] that can be reused
//! across subsequent calls without re-uploading the bytes.
//!
//! Only providers that expose a file-management endpoint implement this
//! trait (e.g. `Anthropic`'s `POST /v1/files`). The trait is intentionally
//! kept separate from [`crate::Provider`] so providers without a files
//! endpoint don't have to stub it out.
// Rust guideline compliant 2026-02-21

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::shared::{FileBytes, ProviderMetadata, ProviderOptions, ProviderReference, Warning};

/// Contract every file-upload model implements.
///
/// Mirrors `FilesV4`.
#[async_trait]
pub trait FilesModel: Send + Sync + std::fmt::Debug {
    /// Provider id, e.g. `"anthropic.files"`.
    fn provider(&self) -> &str;

    /// Specification version (currently `"v4"`).
    fn specification_version(&self) -> &'static str {
        "v4"
    }

    /// Upload a file to the provider and return a reusable reference.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::ProviderError`] when the upstream call fails or
    /// the response is malformed. Implementations should return
    /// [`crate::ProviderError::invalid_argument`] when given an
    /// [`UploadFileData`] variant the endpoint cannot handle.
    async fn upload_file(&self, options: UploadFileOptions) -> Result<UploadFileResult>;
}

/// Options for one [`FilesModel::upload_file`] call.
///
/// Mirrors `FilesV4UploadFileCallOptions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadFileOptions {
    /// File payload.
    pub data: UploadFileData,
    /// IANA media type (e.g. `"application/pdf"`).
    #[serde(rename = "mediaType")]
    pub media_type: String,
    /// Optional filename forwarded to the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    /// Provider-specific options.
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_options: Option<ProviderOptions>,
}

/// Payload variants accepted by [`FilesModel::upload_file`].
///
/// Mirrors `SharedV4FileDataData | SharedV4FileDataText` (the V4 spec
/// excludes URL / reference inputs because the call would have nothing
/// to upload).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum UploadFileData {
    /// Inline bytes or base64-encoded string.
    Data {
        /// Raw bytes or base64 string.
        data: FileBytes,
    },
    /// Inline UTF-8 text.
    Text {
        /// Text content.
        text: String,
    },
}

/// Result of [`FilesModel::upload_file`].
///
/// Mirrors `FilesV4UploadFileResult`.
#[derive(Debug, Clone)]
pub struct UploadFileResult {
    /// `{ providerId → fileId }` reference reusable in later calls.
    pub provider_reference: ProviderReference,
    /// Media type reported by the provider (may differ from input).
    pub media_type: Option<String>,
    /// Filename reported by the provider.
    pub filename: Option<String>,
    /// Provider-specific metadata.
    pub provider_metadata: Option<ProviderMetadata>,
    /// Warnings (e.g. setting coerced away).
    pub warnings: Vec<Warning>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn options_serde_roundtrip_data() {
        let opts = UploadFileOptions {
            data: UploadFileData::Data {
                data: FileBytes::Base64("aGVsbG8=".into()),
            },
            media_type: "text/plain".into(),
            filename: Some("hi.txt".into()),
            provider_options: None,
        };
        let json = serde_json::to_value(&opts).unwrap();
        assert_eq!(json["mediaType"], "text/plain");
        assert_eq!(json["data"]["type"], "data");
        assert_eq!(json["data"]["data"], "aGVsbG8=");
        let back: UploadFileOptions = serde_json::from_value(json).unwrap();
        assert_eq!(back.media_type, "text/plain");
    }

    #[test]
    fn options_serde_roundtrip_text() {
        let opts = UploadFileOptions {
            data: UploadFileData::Text {
                text: "hello".into(),
            },
            media_type: "text/plain".into(),
            filename: None,
            provider_options: None,
        };
        let json = serde_json::to_value(&opts).unwrap();
        assert_eq!(json["data"], json!({ "type": "text", "text": "hello" }));
    }

    #[test]
    fn upload_data_tagged_correctly() {
        let v = serde_json::to_value(UploadFileData::Data {
            data: FileBytes::Bytes(vec![1, 2, 3]),
        })
        .unwrap();
        assert_eq!(v["type"], "data");
    }
}
