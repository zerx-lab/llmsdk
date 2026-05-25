//! xAI Files API model implementation.
//!
//! Mirrors `@ai-sdk/xai/src/files/xai-files.ts` — wraps
//! `POST {base_url}/files` (multipart/form-data) to upload a file and
//! return a reusable [`UploadFileResult`].
// Rust guideline compliant 2026-05-25

use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::error::Result;
use llmsdk_provider::shared::FileBytes;
use llmsdk_provider::{FilesModel, UploadFileData, UploadFileOptions, UploadFileResult};
use llmsdk_provider_utils::http::{RawRequest, post_raw};
use llmsdk_provider_utils::multipart::Multipart;
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::PROVIDER_ID;
use crate::config::Inner;
use crate::files::options::parse_xai_files_options;
use crate::files::wire::WireUploadResponse;

/// Default filename when the caller does not supply one, matching the
/// upstream upstream behaviour where `FormData.append('file', blob)` sets
/// `blob` as the implicit filename.
const DEFAULT_FILENAME: &str = "blob";

/// xAI Files API handle.
///
/// Returned by [`crate::Xai::files`]. Implements [`FilesModel`] for
/// `POST {base_url}/files`. Cheap to clone; the underlying HTTP client and
/// headers are shared with the parent provider.
///
/// # Provider options
///
/// Set under `provider_options["xai"]`:
///
/// - `teamId` (string, optional) — forwarded as the `team_id` form field.
///
/// Unknown keys are accepted silently (matches the upstream
/// `.passthrough()` zod schema).
///
/// # Examples
///
/// ```no_run
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// use llmsdk_provider::shared::FileBytes;
/// use llmsdk_provider::{FilesModel, UploadFileData, UploadFileOptions};
/// use llmsdk_xai::Xai;
///
/// let xai = Xai::from_env()?;
/// let r = xai
///     .files()
///     .upload_file(UploadFileOptions {
///         data: UploadFileData::Data {
///             data: FileBytes::Bytes(b"PDF-1.4 ...".to_vec()),
///         },
///         media_type: "application/pdf".into(),
///         filename: Some("report.pdf".into()),
///         provider_options: None,
///     })
///     .await?;
/// assert!(r.provider_reference.contains_key("xai"));
/// # Ok(()) }
/// ```
#[derive(Debug, Clone)]
pub struct XaiFiles {
    inner: Arc<Inner>,
    provider: String,
}

impl XaiFiles {
    pub(crate) fn new(inner: Arc<Inner>) -> Self {
        Self {
            inner,
            provider: format!("{PROVIDER_ID}.files"),
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/files", self.inner.base_url)
    }
}

#[async_trait]
impl FilesModel for XaiFiles {
    fn provider(&self) -> &str {
        &self.provider
    }

    async fn upload_file(&self, options: UploadFileOptions) -> Result<UploadFileResult> {
        let xai_opts = parse_xai_files_options(options.provider_options.as_ref())?;

        let bytes = upload_data_to_bytes(&options.data)?;

        let filename_for_form = options
            .filename
            .clone()
            .unwrap_or_else(|| DEFAULT_FILENAME.to_owned());

        let mut mp = Multipart::new();
        // Upstream sends `Content-Type: <mediaType>` on the file part via
        // `new Blob([fileBytes], { type: mediaType })`. We mirror that.
        mp.file(
            "file",
            &filename_for_form,
            Some(&options.media_type),
            &bytes,
        );
        if let Some(team_id) = xai_opts.team_id.as_deref() {
            mp.text("team_id", team_id);
        }
        let (boundary, body) = mp.finish();
        let content_type = format!("multipart/form-data; boundary={boundary}");

        let mut req = RawRequest::new(self.endpoint(), body, content_type);
        req.headers = self.inner.headers.clone();

        let envelope = post_raw::<WireUploadResponse>(&self.inner.http, req).await?;
        let resp = envelope.value;

        let mut provider_reference = std::collections::HashMap::new();
        provider_reference.insert(PROVIDER_ID.to_owned(), resp.id.clone());

        // Upstream rule: result.filename = response.filename ?? input.filename,
        // and only included when *one of them* is non-null.
        let echo_filename = resp.filename.clone().or_else(|| options.filename.clone());
        // Upstream rule: result.mediaType is included only when the caller
        // supplied a mediaType (always true on our trait — the field is
        // `String`, not `Option<String>`).
        let echo_media_type = Some(options.media_type.clone());

        // providerMetadata.xai: include filename/bytes/createdAt only when
        // the server returned them. Mirrors the spread-with-conditional
        // pattern in xai-files.ts.
        let mut meta_obj = JsonMap::new();
        if let Some(f) = resp.filename.as_ref() {
            meta_obj.insert("filename".to_owned(), JsonValue::String(f.clone()));
        }
        if let Some(b) = resp.bytes {
            meta_obj.insert("bytes".to_owned(), JsonValue::Number(b.into()));
        }
        if let Some(t) = resp.created_at {
            meta_obj.insert("createdAt".to_owned(), JsonValue::Number(t.into()));
        }
        let mut provider_metadata = std::collections::HashMap::new();
        provider_metadata.insert(PROVIDER_ID.to_owned(), meta_obj);

        Ok(UploadFileResult {
            provider_reference,
            media_type: echo_media_type,
            filename: echo_filename,
            provider_metadata: Some(provider_metadata),
            warnings: Vec::new(),
        })
    }
}

/// Decode an [`UploadFileData`] payload to the raw bytes we PUT on the wire.
///
/// Mirrors upstream's `convertInlineFileDataToUint8Array`.
fn upload_data_to_bytes(data: &UploadFileData) -> Result<Vec<u8>> {
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

// ---- base64 decoding ---------------------------------------------------
//
// Inline RFC 4648 decoder mirroring the one in `src/image/wire.rs`.
// Duplicated here so the `files` module can be lifted out without dragging
// the `image` module along, and to honor the project's no-new-deps rule
// (no `base64` crate). Same approach as `llmsdk_anthropic::files::model`.

fn base64_decode(input: &str) -> std::result::Result<Vec<u8>, Base64Error> {
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

#[derive(Debug)]
enum Base64Error {
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
        .expect("decodes");
        assert_eq!(r, vec![1, 2, 3]);
    }

    #[test]
    fn data_base64_decodes() {
        let r = upload_data_to_bytes(&UploadFileData::Data {
            data: FileBytes::Base64("dGVzdA==".into()),
        })
        .expect("decodes");
        assert_eq!(r, b"test");
    }

    #[test]
    fn data_base64_rejects_invalid() {
        let err = upload_data_to_bytes(&UploadFileData::Data {
            data: FileBytes::Base64("not_padded".into()),
        })
        .unwrap_err();
        assert!(format!("{err}").contains("base64"));
    }

    #[test]
    fn text_encodes_utf8() {
        let r = upload_data_to_bytes(&UploadFileData::Text {
            text: "héllo".into(),
        })
        .expect("decodes");
        assert_eq!(r, "héllo".as_bytes());
    }
}
