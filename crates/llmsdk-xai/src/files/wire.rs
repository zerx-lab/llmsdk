//! Wire types for xAI `POST /v1/files`.
//!
//! Mirrors `xaiFilesResponseSchema` in
//! `@ai-sdk/xai/src/files/xai-files-api.ts`. Only the fields actually
//! consumed by the Rust client are surfaced; the rest of the envelope
//! deserializes away.
// Rust guideline compliant 2026-05-25

use serde::Deserialize;

/// Successful response body from `POST /v1/files`.
///
/// All upstream fields except `id` are nullish; they only appear in
/// `provider_metadata.xai.*` when the server returns them.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct WireUploadResponse {
    /// Server-assigned file id (used in subsequent calls / `ProviderReference`).
    pub id: String,
    /// File object type (e.g. `"file"`). Echoed back; currently unused.
    #[serde(default)]
    #[allow(dead_code, reason = "captured for forward-compat / debugging")]
    pub object: Option<String>,
    /// File size in bytes.
    #[serde(default)]
    pub bytes: Option<u64>,
    /// RFC-3339-style unix timestamp (seconds since epoch).
    #[serde(default)]
    pub created_at: Option<u64>,
    /// Filename echoed back (may differ from input when xAI rewrites it).
    #[serde(default)]
    pub filename: Option<String>,
    /// Purpose tag (e.g. `"assistants"`).
    #[serde(default)]
    #[allow(dead_code, reason = "captured for forward-compat / debugging")]
    pub purpose: Option<String>,
    /// Processing status (e.g. `"processed"`).
    #[serde(default)]
    #[allow(dead_code, reason = "captured for forward-compat / debugging")]
    pub status: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_full_response() {
        let v = json!({
            "id": "file-abc123",
            "object": "file",
            "bytes": 512_u64,
            "created_at": 1_700_000_000_u64,
            "filename": "data.csv",
            "purpose": "assistants",
            "status": "processed",
        });
        let r: WireUploadResponse = serde_json::from_value(v).expect("parses");
        assert_eq!(r.id, "file-abc123");
        assert_eq!(r.bytes, Some(512));
        assert_eq!(r.created_at, Some(1_700_000_000));
        assert_eq!(r.filename.as_deref(), Some("data.csv"));
        assert_eq!(r.purpose.as_deref(), Some("assistants"));
    }

    #[test]
    fn parses_minimal_response() {
        // Only `id` is required; everything else is nullish.
        let v = json!({ "id": "file-1" });
        let r: WireUploadResponse = serde_json::from_value(v).expect("parses");
        assert_eq!(r.id, "file-1");
        assert!(r.bytes.is_none());
        assert!(r.filename.is_none());
        assert!(r.created_at.is_none());
    }

    #[test]
    fn parses_response_with_explicit_nulls() {
        // xAI returns `null` (not omitted) for fields that aren't filled.
        let v = json!({
            "id": "file-x",
            "object": "file",
            "bytes": null,
            "created_at": null,
            "filename": null,
        });
        let r: WireUploadResponse = serde_json::from_value(v).expect("parses");
        assert_eq!(r.id, "file-x");
        assert!(r.bytes.is_none());
        assert!(r.filename.is_none());
    }
}
