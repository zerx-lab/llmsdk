//! Gemini Files API implementation (resumable upload protocol).
//!
//! Mirrors `GoogleFiles` from `@ai-sdk/google/src/google-files.ts`.
//!
//! Two-step protocol:
//!
//! 1. **Init**: `POST {origin}/upload/v1beta/files` with headers
//!    `X-Goog-Upload-Protocol: resumable`, `X-Goog-Upload-Command: start`,
//!    `X-Goog-Upload-Header-Content-Length`, `X-Goog-Upload-Header-Content-Type`
//!    and JSON body `{"file":{"display_name":"..."}}`. The response carries
//!    the upload URL in the `x-goog-upload-url` header.
//! 2. **Upload+finalize**: `POST <upload_url>` with the raw file bytes,
//!    plus `X-Goog-Upload-Offset: 0` and `X-Goog-Upload-Command: upload, finalize`.
//!    The response body is `{ file: GoogleFileResource }`.
//!
//! If the returned file is in `PROCESSING` state, the model polls
//! `GET /<file.name>` every `pollIntervalMs` until it transitions to
//! `ACTIVE` (or fails).
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use llmsdk_provider::error::Result;
use llmsdk_provider::shared::{FileBytes, ProviderMetadata, ProviderReference};
use llmsdk_provider::{
    FilesModel, ProviderError, UploadFileData, UploadFileOptions, UploadFileResult,
};
use llmsdk_provider_utils::http::{JsonRequest, RawRequest, get_json, post_json, post_raw};
use serde_json::{Map, Value};

use crate::PROVIDER_ID;
use crate::base64::decode as base64_decode;
use crate::config::Inner;
use crate::error::rewrite_google_error;

use super::options::parse as parse_files_options;
use super::wire::{GoogleFileResource, UploadFileEnvelope};

const DEFAULT_POLL_INTERVAL_MS: u64 = 2_000;
const DEFAULT_POLL_TIMEOUT_MS: u64 = 300_000;

/// Gemini Files API handle.
#[derive(Debug, Clone)]
pub struct GoogleFiles {
    inner: Arc<Inner>,
    provider: String,
}

impl GoogleFiles {
    pub(crate) fn new(inner: Arc<Inner>) -> Self {
        let provider = format!("{}.files", inner.provider);
        Self { inner, provider }
    }
}

#[async_trait]
impl FilesModel for GoogleFiles {
    fn provider(&self) -> &str {
        &self.provider
    }

    async fn upload_file(&self, options: UploadFileOptions) -> Result<UploadFileResult> {
        let google_options =
            parse_files_options(options.provider_options.as_ref())?.unwrap_or_default();

        let warnings = Vec::new();
        let bytes = upload_data_to_bytes(&options.data)?;
        let display_name = google_options.display_name.as_deref();

        // Strip trailing /v1beta from baseURL to derive the upload host.
        let base_origin = self
            .inner
            .base_url
            .strip_suffix("/v1beta")
            .unwrap_or(&self.inner.base_url);
        let init_url = format!("{base_origin}/upload/v1beta/files");

        let init_body = {
            let mut o = Map::new();
            let mut file = Map::new();
            if let Some(name) = display_name {
                file.insert("display_name".into(), Value::String(name.to_owned()));
            }
            o.insert("file".into(), Value::Object(file));
            Value::Object(o)
        };

        // Init request: extra protocol headers.
        let mut init_headers = self.inner.headers.clone();
        init_headers.insert("X-Goog-Upload-Protocol".into(), Some("resumable".into()));
        init_headers.insert("X-Goog-Upload-Command".into(), Some("start".into()));
        init_headers.insert(
            "X-Goog-Upload-Header-Content-Length".into(),
            Some(bytes.len().to_string()),
        );
        init_headers.insert(
            "X-Goog-Upload-Header-Content-Type".into(),
            Some(options.media_type.clone()),
        );

        // `post_json` captures headers; the init body is `{}`-ish JSON we
        // do not need, so decode as a permissive `Value`.
        let mut init_req = JsonRequest::new(init_url.clone(), init_body);
        init_req.headers = init_headers;
        let init_response = post_json::<_, Value>(&self.inner.http, init_req)
            .await
            .map_err(rewrite_google_error)?;

        let upload_url = init_response
            .headers
            .get("x-goog-upload-url")
            .cloned()
            .ok_or_else(|| {
                ProviderError::api_call_builder(&init_url, "No upload URL returned").build()
            })?;

        // Step 2: upload + finalize.
        let mut upload_headers: HashMap<String, Option<String>> = HashMap::new();
        upload_headers.insert("X-Goog-Upload-Offset".into(), Some("0".into()));
        upload_headers.insert(
            "X-Goog-Upload-Command".into(),
            Some("upload, finalize".into()),
        );
        // Re-add the API key header (the upload URL may strip it).
        if let Some(key) = self.inner.headers.get("x-goog-api-key") {
            upload_headers.insert("x-goog-api-key".into(), key.clone());
        }
        let mut req = RawRequest::new(upload_url.clone(), bytes, options.media_type.clone());
        req.headers = upload_headers;
        let envelope = post_raw::<UploadFileEnvelope>(&self.inner.http, req)
            .await
            .map_err(rewrite_google_error)?;
        let mut file = envelope.value.file;

        // Step 3: poll while PROCESSING.
        let interval = Duration::from_millis(
            google_options
                .poll_interval_ms
                .unwrap_or(DEFAULT_POLL_INTERVAL_MS),
        );
        let timeout = Duration::from_millis(
            google_options
                .poll_timeout_ms
                .unwrap_or(DEFAULT_POLL_TIMEOUT_MS),
        );
        let deadline = Instant::now() + timeout;
        while file.state == "PROCESSING" {
            if Instant::now() > deadline {
                return Err(ProviderError::api_call_builder(
                    format!("{}/{}", self.inner.base_url, file.name),
                    format!("File processing timed out after {} ms", timeout.as_millis()),
                )
                .build());
            }
            tokio::time::sleep(interval).await;
            let poll_url = format!("{}/{}", self.inner.base_url, file.name);
            let polled =
                get_json::<GoogleFileResource, _>(&self.inner.http, &poll_url, &self.inner.headers)
                    .await
                    .map_err(rewrite_google_error)?;
            file = polled.value;
        }

        if file.state == "FAILED" {
            return Err(ProviderError::api_call_builder(
                format!("{}/{}", self.inner.base_url, file.name),
                format!("File processing failed for {}", file.name),
            )
            .build());
        }

        let mut provider_reference = ProviderReference::new();
        provider_reference.insert(PROVIDER_ID.into(), file.uri.clone());

        let mut meta = Map::new();
        meta.insert("name".into(), Value::String(file.name.clone()));
        if let Some(n) = &file.display_name {
            meta.insert("displayName".into(), Value::String(n.clone()));
        }
        meta.insert("mimeType".into(), Value::String(file.mime_type.clone()));
        if let Some(sb) = &file.size_bytes {
            meta.insert("sizeBytes".into(), Value::String(sb.clone()));
        }
        meta.insert("state".into(), Value::String(file.state.clone()));
        meta.insert("uri".into(), Value::String(file.uri.clone()));
        if let Some(t) = &file.create_time {
            meta.insert("createTime".into(), Value::String(t.clone()));
        }
        if let Some(t) = &file.update_time {
            meta.insert("updateTime".into(), Value::String(t.clone()));
        }
        if let Some(t) = &file.expiration_time {
            meta.insert("expirationTime".into(), Value::String(t.clone()));
        }
        if let Some(h) = &file.sha256_hash {
            meta.insert("sha256Hash".into(), Value::String(h.clone()));
        }
        let mut provider_metadata = ProviderMetadata::new();
        provider_metadata.insert(PROVIDER_ID.into(), meta);

        Ok(UploadFileResult {
            provider_reference,
            media_type: Some(file.mime_type),
            filename: options.filename,
            provider_metadata: Some(provider_metadata),
            warnings,
        })
    }
}

fn upload_data_to_bytes(data: &UploadFileData) -> Result<Vec<u8>> {
    match data {
        UploadFileData::Data { data } => match data {
            FileBytes::Bytes(b) => Ok(b.clone()),
            FileBytes::Base64(s) => base64_decode(s).map_err(|err| {
                ProviderError::type_validation(
                    "data.data",
                    Value::String(s.clone()),
                    format!("invalid base64: {err}"),
                )
            }),
        },
        UploadFileData::Text { text } => Ok(text.clone().into_bytes()),
    }
}
