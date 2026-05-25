//! `Anthropic` Files API.
//!
//! Mirrors `@ai-sdk/anthropic/src/anthropic-files.ts`. Wraps
//! `POST /v1/files` with `anthropic-beta: files-api-2025-04-14`.
// Rust guideline compliant 2026-02-21

use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::error::Result;
use llmsdk_provider::shared::FileBytes;
use llmsdk_provider::{
    FilesModel, ProviderError, UploadFileData, UploadFileOptions, UploadFileResult,
};
use llmsdk_provider_utils::http::{RawRequest, post_raw};
use llmsdk_provider_utils::multipart::Multipart;
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::auth::apply_request_auth;
use crate::config::Inner;
use crate::error::rewrite_anthropic_error;
use crate::files::wire::WireUploadResponse;

const FILES_BETA_HEADER: &str = "files-api-2025-04-14";
const DEFAULT_FILENAME: &str = "blob";

/// `Anthropic` `Files` API handle.
///
/// Returned by [`crate::Anthropic::files`]. Implements
/// [`FilesModel`] for `POST /v1/files`.
#[derive(Debug, Clone)]
pub struct AnthropicFiles {
    inner: Arc<Inner>,
    provider: String,
}

impl AnthropicFiles {
    pub(crate) fn new(inner: Arc<Inner>, provider: String) -> Self {
        Self { inner, provider }
    }

    fn endpoint(&self) -> String {
        format!("{}/files", self.inner.base_url)
    }
}

#[async_trait]
impl FilesModel for AnthropicFiles {
    fn provider(&self) -> &str {
        &self.provider
    }

    async fn upload_file(&self, options: UploadFileOptions) -> Result<UploadFileResult> {
        let bytes = upload_data_to_bytes(&options.data)?;
        let filename = options
            .filename
            .clone()
            .unwrap_or_else(|| DEFAULT_FILENAME.to_owned());

        let mut mp = Multipart::new();
        mp.file("file", &filename, Some(&options.media_type), &bytes);
        let (boundary, body) = mp.finish();
        let content_type = format!("multipart/form-data; boundary={boundary}");

        let mut headers = self.inner.headers.clone();
        headers.insert("anthropic-beta".into(), Some(FILES_BETA_HEADER.to_owned()));

        let mut req = RawRequest::new(self.endpoint(), body, content_type);
        req.headers = headers;
        apply_request_auth(
            self.inner.request_auth.as_ref(),
            &mut req.headers,
            "POST",
            &req.url,
            &req.body,
        )
        .await?;

        let envelope = match post_raw::<WireUploadResponse>(&self.inner.http, req).await {
            Ok(r) => r,
            Err(err) => return Err(rewrite_anthropic_error(err)),
        };
        let resp = envelope.value;

        let mut provider_reference = std::collections::HashMap::new();
        provider_reference.insert("anthropic".to_owned(), resp.id.clone());

        let mut meta_obj = JsonMap::new();
        meta_obj.insert(
            "filename".to_owned(),
            JsonValue::String(resp.filename.clone()),
        );
        meta_obj.insert(
            "mimeType".to_owned(),
            JsonValue::String(resp.mime_type.clone()),
        );
        meta_obj.insert(
            "sizeBytes".to_owned(),
            JsonValue::Number(resp.size_bytes.into()),
        );
        meta_obj.insert(
            "createdAt".to_owned(),
            JsonValue::String(resp.created_at.clone()),
        );
        if let Some(d) = resp.downloadable {
            meta_obj.insert("downloadable".to_owned(), JsonValue::Bool(d));
        }
        let mut provider_metadata = std::collections::HashMap::new();
        provider_metadata.insert("anthropic".to_owned(), meta_obj);

        Ok(UploadFileResult {
            provider_reference,
            media_type: Some(resp.mime_type),
            filename: Some(resp.filename),
            provider_metadata: Some(provider_metadata),
            warnings: Vec::new(),
        })
    }
}

/// Decode an [`UploadFileData`] payload to raw bytes for the wire.
pub(crate) fn upload_data_to_bytes(data: &UploadFileData) -> Result<Vec<u8>> {
    match data {
        UploadFileData::Data { data: bytes } => match bytes {
            FileBytes::Bytes(b) => Ok(b.clone()),
            FileBytes::Base64(s) => base64_decode(s).map_err(|err| {
                ProviderError::type_validation(
                    "data.data",
                    JsonValue::String(s.clone()),
                    format!("invalid base64: {err}"),
                )
            }),
        },
        UploadFileData::Text { text } => Ok(text.clone().into_bytes()),
    }
}

/// Minimal RFC 4648 base64 decoder.
///
/// Mirrors the private helper in `convert_prompt.rs`'s sibling `OpenAI` crate
/// — kept inline to honor the project's no-new-deps rule.
pub(crate) fn base64_decode(input: &str) -> std::result::Result<Vec<u8>, Base64Error> {
    let bytes = input.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err(Base64Error::Length);
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks_exact(4) {
        let (b0, p0) = decode_byte(chunk[0])?;
        let (b1, p1) = decode_byte(chunk[1])?;
        let (b2, p2) = decode_byte(chunk[2])?;
        let (b3, p3) = decode_byte(chunk[3])?;
        if p0 || p1 {
            return Err(Base64Error::Padding);
        }
        let n =
            (u32::from(b0) << 18) | (u32::from(b1) << 12) | (u32::from(b2) << 6) | u32::from(b3);
        out.push(((n >> 16) & 0xFF) as u8);
        if !p2 {
            out.push(((n >> 8) & 0xFF) as u8);
        }
        if !p3 {
            if p2 {
                return Err(Base64Error::Padding);
            }
            out.push((n & 0xFF) as u8);
        }
    }
    Ok(out)
}

fn decode_byte(b: u8) -> std::result::Result<(u8, bool), Base64Error> {
    Ok(match b {
        b'A'..=b'Z' => (b - b'A', false),
        b'a'..=b'z' => (b - b'a' + 26, false),
        b'0'..=b'9' => (b - b'0' + 52, false),
        b'+' => (62, false),
        b'/' => (63, false),
        b'=' => (0, true),
        _ => return Err(Base64Error::Byte(b)),
    })
}

/// Reasons [`base64_decode`] can fail.
#[derive(Debug)]
pub(crate) enum Base64Error {
    Length,
    Padding,
    Byte(u8),
}

impl std::fmt::Display for Base64Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Length => f.write_str("input length is not a multiple of 4"),
            Self::Padding => f.write_str("misplaced padding"),
            Self::Byte(b) => write!(f, "non-alphabet byte 0x{b:02x}"),
        }
    }
}

impl std::error::Error for Base64Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_bytes_passes_through() {
        let r = upload_data_to_bytes(&UploadFileData::Data {
            data: FileBytes::Bytes(vec![1, 2, 3]),
        })
        .unwrap();
        assert_eq!(r, vec![1, 2, 3]);
    }

    #[test]
    fn data_base64_decodes() {
        let r = upload_data_to_bytes(&UploadFileData::Data {
            data: FileBytes::Base64("aGVsbG8=".into()),
        })
        .unwrap();
        assert_eq!(r, b"hello");
    }

    #[test]
    fn data_base64_rejects_invalid() {
        let err = upload_data_to_bytes(&UploadFileData::Data {
            data: FileBytes::Base64("not_padded".into()),
        })
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("base64"), "expected base64 error, got: {msg}");
    }

    #[test]
    fn text_encodes_utf8() {
        let r = upload_data_to_bytes(&UploadFileData::Text {
            text: "héllo".into(),
        })
        .unwrap();
        assert_eq!(r, "héllo".as_bytes());
    }
}
