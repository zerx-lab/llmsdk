//! `OpenAI` Skills API.
//!
//! Mirrors `@ai-sdk/openai/src/skills/openai-skills.ts`. Wraps
//! `POST /v1/skills` with `multipart/form-data` (one `files[]` part per file).
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::error::Result;
use llmsdk_provider::shared::{FileBytes, Warning};
use llmsdk_provider::{
    ProviderError, SkillsModel, UploadFileData, UploadSkillOptions, UploadSkillResult,
};
use llmsdk_provider_utils::http::{RawRequest, post_raw};
use llmsdk_provider_utils::multipart::Multipart;
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::config::Inner;
use crate::error::rewrite_openai_error;
use crate::skills::wire::WireSkillResponse;

/// `OpenAI` Skills API handle.
#[derive(Debug, Clone)]
pub struct OpenAiSkills {
    inner: Arc<Inner>,
    provider: String,
}

impl OpenAiSkills {
    /// Construct from a fully assembled [`Inner`]. Public for cross-crate
    /// composition; end-users should prefer the provider builder's
    /// `skills()` factory.
    #[must_use]
    pub fn new(inner: Arc<Inner>, provider: String) -> Self {
        Self { inner, provider }
    }

    fn endpoint(&self) -> String {
        self.inner.endpoint("/skills", "")
    }
}

#[async_trait]
impl SkillsModel for OpenAiSkills {
    fn provider(&self) -> &str {
        &self.provider
    }

    async fn upload_skill(&self, options: UploadSkillOptions) -> Result<UploadSkillResult> {
        let mut warnings = Vec::new();
        // OpenAI Skills API doesn't store a display title — mirror upstream warning.
        if options.display_title.is_some() {
            warnings.push(Warning::Unsupported {
                feature: "displayTitle".to_owned(),
                details: None,
            });
        }

        let mut mp = Multipart::new();
        for file in &options.files {
            let bytes = skill_file_to_bytes(&file.data, &file.path)?;
            // ai-sdk uses the file's relative path as the "filename" — preserves
            // the bundle layout server-side.
            mp.file("files[]", &file.path, None, &bytes);
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

        let envelope = match post_raw::<WireSkillResponse>(&self.inner.http, req).await {
            Ok(r) => r,
            Err(err) => return Err(rewrite_openai_error(err)),
        };
        let resp = envelope.value;

        let mut provider_reference: HashMap<String, String> = HashMap::new();
        provider_reference.insert("openai".to_owned(), resp.id);

        let mut meta = JsonMap::new();
        if let Some(v) = &resp.default_version {
            meta.insert("defaultVersion".to_owned(), JsonValue::String(v.clone()));
        }
        if let Some(ts) = resp.created_at {
            meta.insert("createdAt".to_owned(), JsonValue::Number(ts.into()));
        }
        if let Some(ts) = resp.updated_at {
            meta.insert("updatedAt".to_owned(), JsonValue::Number(ts.into()));
        }
        let mut provider_metadata = HashMap::new();
        provider_metadata.insert("openai".to_owned(), meta);

        Ok(UploadSkillResult {
            provider_reference,
            display_title: options.display_title,
            name: resp.name,
            description: resp.description,
            latest_version: resp.latest_version,
            provider_metadata: Some(provider_metadata),
            warnings,
        })
    }
}

fn skill_file_to_bytes(data: &UploadFileData, path: &str) -> Result<Vec<u8>> {
    match data {
        UploadFileData::Data { data } => match data {
            FileBytes::Bytes(b) => Ok(b.clone()),
            FileBytes::Base64(s) => decode_base64(s).map_err(|err| {
                ProviderError::type_validation(
                    format!("files[{path}].data"),
                    JsonValue::String(s.clone()),
                    format!("invalid base64: {err}"),
                )
            }),
        },
        UploadFileData::Text { text } => Ok(text.clone().into_bytes()),
    }
}

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
    use llmsdk_provider::SkillFile;

    #[test]
    fn text_file_passes_through_as_utf8() {
        let f = SkillFile {
            path: "main.py".into(),
            data: UploadFileData::Text {
                text: "print(1)".into(),
            },
        };
        assert_eq!(skill_file_to_bytes(&f.data, &f.path).unwrap(), b"print(1)");
    }

    #[test]
    fn binary_file_passes_through() {
        let f = SkillFile {
            path: "blob.bin".into(),
            data: UploadFileData::Data {
                data: FileBytes::Bytes(vec![0, 1, 2]),
            },
        };
        assert_eq!(
            skill_file_to_bytes(&f.data, &f.path).unwrap(),
            vec![0, 1, 2]
        );
    }
}
