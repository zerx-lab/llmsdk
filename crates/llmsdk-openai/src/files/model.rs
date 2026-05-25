//! `OpenAI` Files API.
//!
//! Mirrors `@ai-sdk/openai/src/files/openai-files.ts`. Wraps
//! `POST /v1/files` with `multipart/form-data`.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::error::Result;
use llmsdk_provider::shared::FileBytes;
use llmsdk_provider::{
    FilesModel, ProviderError, UploadFileData, UploadFileOptions, UploadFileResult,
};
use llmsdk_provider_utils::http::{RawRequest, post_raw};
use llmsdk_provider_utils::multipart::Multipart;
use serde::Deserialize;
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::config::Inner;
use crate::error::rewrite_openai_error;
use crate::files::wire::WireFileResponse;

const DEFAULT_PURPOSE: &str = "assistants";
const DEFAULT_FILENAME: &str = "blob";

/// `OpenAI` Files API handle.
///
/// Returned by [`crate::OpenAi::files`]. Implements [`FilesModel`] for
/// `POST /v1/files`.
#[derive(Debug, Clone)]
pub struct OpenAiFiles {
    inner: Arc<Inner>,
    provider: String,
}

impl OpenAiFiles {
    /// Construct from a fully assembled [`Inner`]. Public for cross-crate
    /// composition. End-users should prefer the provider builder's
    /// `files()` factory.
    #[must_use]
    pub fn new(inner: Arc<Inner>, provider: String) -> Self {
        Self { inner, provider }
    }

    fn endpoint(&self) -> String {
        self.inner.endpoint("/files", "")
    }
}

/// Parsed `provider_options["openai"]` slot for [`OpenAiFiles::upload_file`].
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct OpenAiFilesOptions {
    /// `purpose` form-field; defaults to `"assistants"`.
    purpose: Option<String>,
    /// `expires_after` form-field (seconds).
    expires_after: Option<u64>,
}

fn parse_options(opts: Option<&llmsdk_provider::shared::ProviderOptions>) -> OpenAiFilesOptions {
    let Some(map) = opts else {
        return OpenAiFilesOptions::default();
    };
    let Some(slot) = map.get("openai") else {
        return OpenAiFilesOptions::default();
    };
    serde_json::from_value::<OpenAiFilesOptions>(serde_json::Value::Object(slot.clone()))
        .unwrap_or_default()
}

#[async_trait]
impl FilesModel for OpenAiFiles {
    fn provider(&self) -> &str {
        &self.provider
    }

    async fn upload_file(&self, options: UploadFileOptions) -> Result<UploadFileResult> {
        let provider_opts = parse_options(options.provider_options.as_ref());

        let bytes = upload_data_to_bytes(&options.data)?;
        let filename = options
            .filename
            .clone()
            .unwrap_or_else(|| DEFAULT_FILENAME.to_owned());

        let mut mp = Multipart::new();
        mp.file("file", &filename, Some(&options.media_type), &bytes);
        mp.text(
            "purpose",
            &provider_opts
                .purpose
                .unwrap_or_else(|| DEFAULT_PURPOSE.to_owned()),
        );
        if let Some(secs) = provider_opts.expires_after {
            mp.text("expires_after", &secs.to_string());
        }
        let (boundary, body) = mp.finish();
        let content_type = format!("multipart/form-data; boundary={boundary}");

        let url = self.endpoint();
        let mut req_headers = self.inner.headers.clone();
        self.inner
            .sign_if_needed(&mut req_headers, "POST", &url, &body)
            .await?;
        let mut req = RawRequest::new(url, body, content_type);
        req.headers = req_headers;

        let envelope = match post_raw::<WireFileResponse>(&self.inner.http, req).await {
            Ok(r) => r,
            Err(err) => return Err(rewrite_openai_error(err)),
        };
        let resp = envelope.value;

        let mut provider_reference: HashMap<String, String> = HashMap::new();
        provider_reference.insert("openai".to_owned(), resp.id);

        let mut meta = JsonMap::new();
        if let Some(name) = &resp.filename {
            meta.insert("filename".to_owned(), JsonValue::String(name.clone()));
        }
        if let Some(purpose) = &resp.purpose {
            meta.insert("purpose".to_owned(), JsonValue::String(purpose.clone()));
        }
        if let Some(bytes) = resp.bytes {
            meta.insert("bytes".to_owned(), JsonValue::Number(bytes.into()));
        }
        if let Some(ts) = resp.created_at {
            meta.insert("createdAt".to_owned(), JsonValue::Number(ts.into()));
        }
        if let Some(status) = resp.status {
            meta.insert("status".to_owned(), JsonValue::String(status));
        }
        if let Some(exp) = resp.expires_at {
            meta.insert("expiresAt".to_owned(), JsonValue::Number(exp.into()));
        }
        let mut provider_metadata = HashMap::new();
        provider_metadata.insert("openai".to_owned(), meta);

        Ok(UploadFileResult {
            provider_reference,
            media_type: Some(options.media_type),
            filename: resp.filename.or(options.filename),
            provider_metadata: Some(provider_metadata),
            warnings: Vec::new(),
        })
    }
}

/// Decode an [`UploadFileData`] payload to raw bytes.
fn upload_data_to_bytes(data: &UploadFileData) -> Result<Vec<u8>> {
    match data {
        UploadFileData::Data { data } => match data {
            FileBytes::Bytes(b) => Ok(b.clone()),
            FileBytes::Base64(s) => decode_base64(s).map_err(|err| {
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

/// Minimal RFC 4648 base64 decoder (kept inline to honor the no-new-deps rule).
fn decode_base64(input: &str) -> std::result::Result<Vec<u8>, &'static str> {
    let bytes = input.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err("length not a multiple of 4");
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks_exact(4) {
        let (b0, _) = decode_byte(chunk[0])?;
        let (b1, _) = decode_byte(chunk[1])?;
        let (b2, p2) = decode_byte(chunk[2])?;
        let (b3, p3) = decode_byte(chunk[3])?;
        out.push((b0 << 2) | (b1 >> 4));
        if !p2 {
            out.push(((b1 & 0x0F) << 4) | (b2 >> 2));
        }
        if !p3 {
            out.push(((b2 & 0x03) << 6) | b3);
        }
    }
    Ok(out)
}

fn decode_byte(c: u8) -> std::result::Result<(u8, bool), &'static str> {
    match c {
        b'A'..=b'Z' => Ok((c - b'A', false)),
        b'a'..=b'z' => Ok((c - b'a' + 26, false)),
        b'0'..=b'9' => Ok((c - b'0' + 52, false)),
        b'+' => Ok((62, false)),
        b'/' => Ok((63, false)),
        b'=' => Ok((0, true)),
        _ => Err("invalid base64 byte"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trip() {
        assert_eq!(decode_base64("aGVsbG8=").unwrap(), b"hello".to_vec());
        assert_eq!(decode_base64("Zm9v").unwrap(), b"foo".to_vec());
        assert_eq!(decode_base64("Zm9vYmFy").unwrap(), b"foobar".to_vec());
    }

    #[test]
    fn upload_data_decodes_base64() {
        let data = UploadFileData::Data {
            data: FileBytes::Base64("aGVsbG8=".into()),
        };
        assert_eq!(upload_data_to_bytes(&data).unwrap(), b"hello");
    }

    #[test]
    fn upload_data_text_passes_through() {
        let data = UploadFileData::Text {
            text: "hello".into(),
        };
        assert_eq!(upload_data_to_bytes(&data).unwrap(), b"hello");
    }
}
