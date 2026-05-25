//! Wire types for `POST /v1/files`.
//!
//! Mirrors `anthropic-files.ts` `anthropicUploadFileResponseSchema`.
// Rust guideline compliant 2026-02-21

use serde::Deserialize;

/// Successful response body from `POST /v1/files`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct WireUploadResponse {
    /// Server-assigned file id (used in subsequent calls).
    pub id: String,
    /// File mime type.
    pub mime_type: String,
    /// Filename echoed back.
    pub filename: String,
    /// Size in bytes.
    pub size_bytes: u64,
    /// RFC 3339 timestamp.
    pub created_at: String,
    /// Whether the file can be downloaded back.
    #[serde(default)]
    pub downloadable: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_response() {
        let json = serde_json::json!({
            "id": "file-abc123",
            "type": "file",
            "filename": "test.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 12_345_u64,
            "created_at": "2025-04-14T12:00:00Z"
        });
        let r: WireUploadResponse = serde_json::from_value(json).unwrap();
        assert_eq!(r.id, "file-abc123");
        assert_eq!(r.mime_type, "application/pdf");
        assert_eq!(r.size_bytes, 12_345);
        assert_eq!(r.downloadable, None);
    }

    #[test]
    fn parses_response_with_downloadable() {
        let json = serde_json::json!({
            "id": "file-1",
            "type": "file",
            "filename": "x.bin",
            "mime_type": "application/octet-stream",
            "size_bytes": 1_u64,
            "created_at": "2025-01-01T00:00:00Z",
            "downloadable": true
        });
        let r: WireUploadResponse = serde_json::from_value(json).unwrap();
        assert_eq!(r.downloadable, Some(true));
    }
}
