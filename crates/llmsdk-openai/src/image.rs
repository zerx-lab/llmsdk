//! `OpenAI` Image Generation model.
//!
//! Mirrors `@ai-sdk/openai/src/image/*`. M8 scope:
//!
//! - text-to-image via `POST /v1/images/generations`
//! - DALL-E 3 + gpt-image-1\* models (different default `response_format`
//!   handling per family)
//! - provider options: `quality` / `style` / `background` / `outputFormat` /
//!   `outputCompression` / `moderation` / `user`
//! - response: `b64_json` decoded into raw bytes; `revised_prompt` /
//!   `created` / wire metadata captured under
//!   `provider_metadata.openai.images[]`
//!
//! # Out of scope (deferred — see `todo.md`)
//!
//! - image editing (`POST /v1/images/edits`) — requires `files` / `mask`
//!   on `ImageOptions`, which the M1 trait does not yet expose
//! - image variations (`POST /v1/images/variations`) — same reason
//! - inline `Usage` reporting on `ImageResult` — not in trait yet; the
//!   provider-side usage is preserved verbatim under `provider_metadata`
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::image_model::{
    GeneratedImage, ImageModel, ImageOptions, ImageResult, ImageUsage as ProviderImageUsage,
    ImageUsageInputDetails,
};
use llmsdk_provider::language_model::FilePart;
use llmsdk_provider::shared::{
    FileBytes, FileData, Headers, ProviderMetadata, RequestInfo, ResponseInfo, Warning,
};
use llmsdk_provider_utils::http::{JsonRequest, RawRequest, post_json, post_raw};
use llmsdk_provider_utils::multipart::Multipart;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as JsonValue};

use crate::PROVIDER_ID;
use crate::config::Inner;
use crate::error::rewrite_openai_error;

/// `OpenAI` image-generation model handle.
///
/// Cheap to clone — shares the provider's HTTP client and auth state.
#[derive(Debug, Clone)]
pub struct OpenAiImageModel {
    inner: Arc<Inner>,
    model_id: String,
}

impl OpenAiImageModel {
    /// Construct from a fully assembled [`Inner`].
    ///
    /// Public for cross-crate composition (Azure `OpenAI`). End-users should
    /// prefer [`crate::OpenAi::image`].
    #[must_use]
    pub fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn endpoint(&self) -> String {
        self.inner.endpoint("/images/generations", &self.model_id)
    }

    fn edits_endpoint(&self) -> String {
        self.inner.endpoint("/images/edits", &self.model_id)
    }

    fn variations_endpoint(&self) -> String {
        self.inner.endpoint("/images/variations", &self.model_id)
    }
}

#[async_trait]
impl ImageModel for OpenAiImageModel {
    fn provider(&self) -> &str {
        self.inner.provider_id()
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_images_per_call(&self) -> Option<u32> {
        Some(max_images_for(&self.model_id))
    }

    async fn do_generate(&self, options: ImageOptions) -> Result<ImageResult, ProviderError> {
        let mode = route_endpoint(&options);

        match mode {
            EndpointMode::Generate => self.do_generate_text2image(options).await,
            EndpointMode::Edit => self.do_edit(options).await,
            EndpointMode::Variation => self.do_variation(options).await,
        }
    }
}

/// Pick the correct endpoint based on which fields `options` has populated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EndpointMode {
    /// Text → image (`POST /v1/images/generations`).
    Generate,
    /// Image + prompt → image (`POST /v1/images/edits`).
    Edit,
    /// Image → image (`POST /v1/images/variations`).
    Variation,
}

fn route_endpoint(options: &ImageOptions) -> EndpointMode {
    let has_files = options.files.as_ref().is_some_and(|f| !f.is_empty());
    if !has_files {
        return EndpointMode::Generate;
    }
    if options.prompt.trim().is_empty() {
        EndpointMode::Variation
    } else {
        EndpointMode::Edit
    }
}

impl OpenAiImageModel {
    async fn do_generate_text2image(
        &self,
        options: ImageOptions,
    ) -> Result<ImageResult, ProviderError> {
        let (request, warnings) = build_request(&self.model_id, &options);

        let request_body_value = serde_json::to_value(&request).ok();
        let body_bytes = serde_json::to_vec(&request).unwrap_or_default();

        let mut request_headers = self.inner.headers.clone();
        if let Some(headers) = options.headers {
            for (name, value) in headers {
                request_headers.insert(name, value);
            }
        }

        let url = self.endpoint();
        self.inner
            .sign_if_needed(&mut request_headers, "POST", &url, &body_bytes)
            .await?;
        let mut http_request = JsonRequest::new(url, request);
        http_request.headers = request_headers;

        let response = match post_json::<_, ImageResponse>(&self.inner.http, http_request).await {
            Ok(r) => r,
            Err(err) => return Err(rewrite_openai_error(err)),
        };

        let resp = response.value;
        let response_headers = response.headers;
        let output_format_response = resp.output_format.clone();

        // Decode the base64 payloads. A single bad payload fails the whole
        // call — partial success would silently drop output, which is worse.
        let mut images = Vec::with_capacity(resp.data.len());
        for (idx, item) in resp.data.iter().enumerate() {
            let bytes = base64_decode(&item.b64_json).map_err(|err| {
                ProviderError::type_validation(
                    format!("data[{idx}].b64_json"),
                    serde_json::Value::String(item.b64_json.clone()),
                    format!("OpenAI returned invalid base64 in image data: {err}"),
                )
            })?;
            let media_type = guess_media_type(output_format_response.as_deref(), &bytes);
            images.push(GeneratedImage {
                // `Vec<u8>` -> `bytes::Bytes` via `From`; avoids needing
                // a direct `bytes` crate import in this provider.
                bytes: bytes.into(),
                media_type,
            });
        }

        let provider_metadata = build_provider_metadata(&resp);

        Ok(ImageResult {
            images,
            warnings,
            usage: resp.usage.as_ref().map(into_provider_usage),
            provider_metadata: Some(provider_metadata),
            request: Some(RequestInfo {
                body: request_body_value,
            }),
            response: Some(ResponseInfo {
                id: resp.created.map(|c| format!("openai-img-{c}")),
                timestamp: resp.created.map(|c| c.to_string()),
                model_id: Some(self.model_id.clone()),
                headers: Some(headers_to_provider(response_headers)),
                ..ResponseInfo::default()
            }),
        })
    }

    async fn do_edit(&self, options: ImageOptions) -> Result<ImageResult, ProviderError> {
        let openai = parse_provider_options(options.provider_options.as_ref());
        let mut warnings = Vec::new();
        collect_unsupported_warnings(&options, &mut warnings);

        let files = options.files.unwrap_or_default();
        if files.is_empty() {
            return Err(ProviderError::invalid_argument(
                "files",
                "image edit endpoint requires at least one source file",
            ));
        }
        let mut mp = Multipart::new();
        mp.text("model", &self.model_id);
        mp.text("prompt", &options.prompt);
        if let Some(n) = options.n {
            mp.text("n", &n.to_string());
        }
        if let Some(size) = &options.size {
            mp.text("size", size);
        }
        if let Some(q) = &openai.quality {
            mp.text("quality", q);
        }
        if let Some(fi) = &openai.input_fidelity {
            mp.text("input_fidelity", fi);
        }
        if let Some(u) = &openai.user {
            mp.text("user", u);
        }
        // `image` (file) — repeated when multiple sources are sent.
        for (idx, fp) in files.iter().enumerate() {
            let bytes = file_part_bytes(fp)?;
            let filename = fp
                .filename
                .clone()
                .unwrap_or_else(|| format!("image-{idx}.png"));
            mp.file("image[]", &filename, Some(fp.media_type.as_str()), &bytes);
        }
        if let Some(mask) = &options.mask {
            let bytes = file_part_bytes(mask)?;
            mp.file(
                "mask",
                mask.filename.as_deref().unwrap_or("mask.png"),
                Some(mask.media_type.as_str()),
                &bytes,
            );
        }
        self.send_multipart(self.edits_endpoint(), mp, options.headers, warnings)
            .await
    }

    async fn do_variation(&self, options: ImageOptions) -> Result<ImageResult, ProviderError> {
        let openai = parse_provider_options(options.provider_options.as_ref());
        let mut warnings = Vec::new();
        collect_unsupported_warnings(&options, &mut warnings);
        if !options.prompt.trim().is_empty() {
            warnings.push(Warning::Unsupported {
                feature: "prompt".to_owned(),
                details: Some("OpenAI image variations endpoint ignores prompt".to_owned()),
            });
        }

        let files = options.files.unwrap_or_default();
        let first = files.into_iter().next().ok_or_else(|| {
            ProviderError::invalid_argument(
                "files",
                "image variation endpoint requires exactly one source file",
            )
        })?;
        let mut mp = Multipart::new();
        mp.text("model", &self.model_id);
        if let Some(n) = options.n {
            mp.text("n", &n.to_string());
        }
        if let Some(size) = &options.size {
            mp.text("size", size);
        }
        if let Some(u) = &openai.user {
            mp.text("user", u);
        }
        let bytes = file_part_bytes(&first)?;
        mp.file(
            "image",
            first.filename.as_deref().unwrap_or("image.png"),
            Some(first.media_type.as_str()),
            &bytes,
        );
        self.send_multipart(self.variations_endpoint(), mp, options.headers, warnings)
            .await
    }

    async fn send_multipart(
        &self,
        url: String,
        mp: Multipart,
        per_call_headers: Option<llmsdk_provider::shared::Headers>,
        warnings: Vec<Warning>,
    ) -> Result<ImageResult, ProviderError> {
        let (boundary, body) = mp.finish();
        let content_type = format!("multipart/form-data; boundary={boundary}");

        let mut request_headers = self.inner.headers.clone();
        if let Some(headers) = per_call_headers {
            for (name, value) in headers {
                request_headers.insert(name, value);
            }
        }

        self.inner
            .sign_if_needed(&mut request_headers, "POST", &url, &body)
            .await?;
        let mut req = RawRequest::new(url, body, content_type);
        req.headers = request_headers;

        let resp_outcome = post_raw::<ImageResponse>(&self.inner.http, req).await;
        let resp_envelope = match resp_outcome {
            Ok(r) => r,
            Err(err) => return Err(rewrite_openai_error(err)),
        };

        let resp = resp_envelope.value;
        let response_headers = resp_envelope.headers;
        let output_format_response = resp.output_format.clone();

        let mut images = Vec::with_capacity(resp.data.len());
        for (idx, item) in resp.data.iter().enumerate() {
            let bytes = base64_decode(&item.b64_json).map_err(|err| {
                ProviderError::type_validation(
                    format!("data[{idx}].b64_json"),
                    serde_json::Value::String(item.b64_json.clone()),
                    format!("OpenAI returned invalid base64 in image data: {err}"),
                )
            })?;
            let media_type = guess_media_type(output_format_response.as_deref(), &bytes);
            images.push(GeneratedImage {
                bytes: bytes.into(),
                media_type,
            });
        }

        let provider_metadata = build_provider_metadata(&resp);

        Ok(ImageResult {
            images,
            warnings,
            usage: resp.usage.as_ref().map(into_provider_usage),
            provider_metadata: Some(provider_metadata),
            request: None,
            response: Some(ResponseInfo {
                id: resp.created.map(|c| format!("openai-img-{c}")),
                timestamp: resp.created.map(|c| c.to_string()),
                model_id: Some(self.model_id.clone()),
                headers: Some(headers_to_provider(response_headers)),
                ..ResponseInfo::default()
            }),
        })
    }
}

/// Surface `aspectRatio` / `seed` warnings for `do_edit` / `do_variation` too.
fn collect_unsupported_warnings(options: &ImageOptions, warnings: &mut Vec<Warning>) {
    if options.aspect_ratio.is_some() {
        warnings.push(Warning::Unsupported {
            feature: "aspectRatio".to_owned(),
            details: Some("OpenAI image endpoints do not support aspect ratio.".to_owned()),
        });
    }
    if options.seed.is_some() {
        warnings.push(Warning::Unsupported {
            feature: "seed".to_owned(),
            details: Some("OpenAI image endpoints do not support seed.".to_owned()),
        });
    }
}

fn file_part_bytes(fp: &FilePart) -> Result<Vec<u8>, ProviderError> {
    match &fp.data {
        FileData::Data { data: bytes } => match bytes {
            FileBytes::Bytes(b) => Ok(b.clone()),
            FileBytes::Base64(s) => base64_decode(s).map_err(|err| {
                ProviderError::type_validation(
                    "file.data",
                    serde_json::Value::String(s.clone()),
                    format!("invalid base64: {err}"),
                )
            }),
        },
        FileData::Text { text } => Ok(text.clone().into_bytes()),
        FileData::Url { .. } | FileData::Reference { .. } => Err(ProviderError::unsupported(
            "OpenAI image endpoints require inline file bytes; URL / Reference variants are not yet downloaded",
        )),
    }
}

fn into_provider_usage(u: &ImageUsage) -> ProviderImageUsage {
    ProviderImageUsage {
        input_tokens: u.input_tokens,
        output_tokens: u.output_tokens,
        input_tokens_details: u
            .input_tokens_details
            .as_ref()
            .map(|d| ImageUsageInputDetails {
                text_tokens: d.text_tokens,
                image_tokens: d.image_tokens,
            }),
    }
}

/// Per-model `n` ceiling. Defaults to 1 for unknown ids — the safe choice.
///
/// Source: <https://platform.openai.com/docs/guides/images>.
fn max_images_for(model_id: &str) -> u32 {
    if model_id == "dall-e-3" || model_id.starts_with("chatgpt-image-") {
        1
    } else if model_id.starts_with("gpt-image-")
        || model_id == "dall-e-2"
        || model_id.starts_with("chatgpt-image")
    {
        10
    } else {
        1
    }
}

/// Models that default to base64 `response_format` and do not accept the
/// `response_format` field on the request.
fn has_default_b64_response_format(model_id: &str) -> bool {
    model_id.starts_with("gpt-image-") || model_id.starts_with("chatgpt-image-")
}

/// Build the wire request, collecting warnings for ignored settings.
fn build_request(model_id: &str, options: &ImageOptions) -> (ImageRequest, Vec<Warning>) {
    let mut warnings = Vec::new();

    if options.aspect_ratio.is_some() {
        warnings.push(Warning::Unsupported {
            feature: "aspectRatio".to_owned(),
            details: Some(
                "OpenAI image models do not support aspect ratio. Use `size` instead.".to_owned(),
            ),
        });
    }

    if options.seed.is_some() {
        warnings.push(Warning::Unsupported {
            feature: "seed".to_owned(),
            details: Some("OpenAI image models do not support a seed.".to_owned()),
        });
    }

    let openai = parse_provider_options(options.provider_options.as_ref());

    let response_format = if has_default_b64_response_format(model_id) {
        None
    } else {
        Some(ResponseFormat::B64Json)
    };

    let request = ImageRequest {
        model: model_id.to_owned(),
        prompt: options.prompt.clone(),
        n: options.n,
        size: options.size.clone(),
        response_format,
        quality: openai.quality,
        style: openai.style,
        background: openai.background,
        moderation: openai.moderation,
        output_format: openai.output_format,
        output_compression: openai.output_compression,
        user: openai.user,
    };

    (request, warnings)
}

/// Typed view of `provider_options["openai"]` for image calls.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct OpenAiImageOptions {
    /// `standard` | `hd` | `low` | `medium` | `high` | `auto`.
    quality: Option<String>,
    /// DALL-E 3 only: `vivid` | `natural`.
    style: Option<String>,
    /// `transparent` | `opaque` | `auto`.
    background: Option<String>,
    /// gpt-image-1 only: `auto` | `low`.
    moderation: Option<String>,
    /// `png` | `jpeg` | `webp`. Used both to set the wire field and to
    /// pick the response `media_type`.
    output_format: Option<String>,
    /// 0-100 (jpeg/webp only).
    output_compression: Option<u32>,
    /// End-user identifier.
    user: Option<String>,
    /// Edits-endpoint only: how closely to follow the input image
    /// (`"low"` / `"medium"` / `"high"`).
    input_fidelity: Option<String>,
}

/// Parse the `openai` slot of `provider_options`, returning defaults on
/// absence / shape mismatch (forgiving like ai-sdk).
fn parse_provider_options(
    options: Option<&llmsdk_provider::shared::ProviderOptions>,
) -> OpenAiImageOptions {
    let Some(map) = options else {
        return OpenAiImageOptions::default();
    };
    let Some(slot) = map.get(PROVIDER_ID) else {
        return OpenAiImageOptions::default();
    };
    serde_json::from_value::<OpenAiImageOptions>(JsonValue::Object(slot.clone()))
        .unwrap_or_default()
}

/// Build the `openai.images[]` provider-metadata payload.
///
/// Each image gets a per-position object with whatever wire fields the
/// upstream surfaced (`revised_prompt`, `created`, `size`, etc.) — matches
/// ai-sdk's shape.
fn build_provider_metadata(resp: &ImageResponse) -> ProviderMetadata {
    let images: Vec<JsonValue> = resp
        .data
        .iter()
        .map(|item| {
            let mut obj = Map::new();
            if let Some(rp) = &item.revised_prompt {
                obj.insert("revisedPrompt".to_owned(), JsonValue::String(rp.clone()));
            }
            if let Some(created) = resp.created {
                obj.insert("created".to_owned(), JsonValue::from(created));
            }
            if let Some(size) = &resp.size {
                obj.insert("size".to_owned(), JsonValue::String(size.clone()));
            }
            if let Some(quality) = &resp.quality {
                obj.insert("quality".to_owned(), JsonValue::String(quality.clone()));
            }
            if let Some(bg) = &resp.background {
                obj.insert("background".to_owned(), JsonValue::String(bg.clone()));
            }
            if let Some(of) = &resp.output_format {
                obj.insert("outputFormat".to_owned(), JsonValue::String(of.clone()));
            }
            JsonValue::Object(obj)
        })
        .collect();

    let mut openai = Map::new();
    openai.insert("images".to_owned(), JsonValue::Array(images));
    if let Some(usage) = &resp.usage
        && let Ok(u) = serde_json::to_value(usage)
    {
        openai.insert("usage".to_owned(), u);
    }

    let mut pm = ProviderMetadata::new();
    pm.insert(PROVIDER_ID.to_owned(), openai);
    pm
}

/// Best-effort media-type detection.
///
/// 1. If the server reported `output_format`, trust it.
/// 2. Otherwise sniff magic bytes (PNG, JPEG, WEBP, GIF).
/// 3. Otherwise default to `image/png` (DALL-E default).
fn guess_media_type(output_format: Option<&str>, bytes: &[u8]) -> String {
    if let Some(fmt) = output_format {
        return format!("image/{fmt}");
    }
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return "image/png".to_owned();
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return "image/jpeg".to_owned();
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return "image/webp".to_owned();
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return "image/gif".to_owned();
    }
    "image/png".to_owned()
}

fn headers_to_provider(raw: HashMap<String, String>) -> Headers {
    raw.into_iter().map(|(k, v)| (k, Some(v))).collect()
}

// ---- wire types -------------------------------------------------------

/// `POST /v1/images/generations` request body.
#[derive(Debug, Clone, Serialize)]
struct ImageRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    style: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    background: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    moderation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_compression: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum ResponseFormat {
    B64Json,
}

/// `POST /v1/images/generations` response body (subset).
#[derive(Debug, Clone, Deserialize, Serialize)]
struct ImageResponse {
    #[serde(default)]
    created: Option<u64>,
    data: Vec<ImageData>,
    #[serde(default)]
    background: Option<String>,
    #[serde(default)]
    output_format: Option<String>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    quality: Option<String>,
    #[serde(default)]
    usage: Option<ImageUsage>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ImageData {
    b64_json: String,
    #[serde(default)]
    revised_prompt: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ImageUsage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
    #[serde(default)]
    input_tokens_details: Option<InputTokensDetails>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct InputTokensDetails {
    #[serde(default)]
    image_tokens: Option<u64>,
    #[serde(default)]
    text_tokens: Option<u64>,
}

// ---- base64 decoding --------------------------------------------------

/// Minimal RFC 4648 base64 decoder.
///
/// We intentionally avoid a third-party `base64` dependency to honor the
/// project's no-new-deps rule. Accepts standard alphabet (`+/`) with
/// optional `=` padding; rejects whitespace, urlsafe alphabet, and any
/// non-alphabet byte with [`Base64Error`].
fn base64_decode(input: &str) -> Result<Vec<u8>, Base64Error> {
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
        // Padding may only appear at positions 2 and/or 3, never earlier.
        if p0 || p1 {
            return Err(Base64Error::Padding);
        }
        let n =
            (u32::from(b0) << 18) | (u32::from(b1) << 12) | (u32::from(b2) << 6) | u32::from(b3);
        // Mask before casting so clippy's truncation lint is satisfied;
        // the masks are no-ops at runtime (each byte fits in 8 bits).
        out.push(((n >> 16) & 0xFF) as u8);
        if !p2 {
            out.push(((n >> 8) & 0xFF) as u8);
        }
        if !p3 {
            // p2 without p3 is illegal: data after padding.
            if p2 {
                return Err(Base64Error::Padding);
            }
            out.push((n & 0xFF) as u8);
        }
    }
    Ok(out)
}

/// Map one base64 byte to its 6-bit value; the `bool` flags `=` (padding).
fn decode_byte(b: u8) -> Result<(u8, bool), Base64Error> {
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
    use serde_json::json;

    #[test]
    fn max_images_for_known_models() {
        assert_eq!(max_images_for("dall-e-3"), 1);
        assert_eq!(max_images_for("dall-e-2"), 10);
        assert_eq!(max_images_for("gpt-image-1"), 10);
        assert_eq!(max_images_for("gpt-image-1-mini"), 10);
        assert_eq!(max_images_for("chatgpt-image-latest"), 1);
        // Unknown id defaults to the safe value.
        assert_eq!(max_images_for("custom-image-alias"), 1);
    }

    #[test]
    fn dall_e_3_requests_b64_response_format() {
        let opts = ImageOptions {
            prompt: "hi".into(),
            ..Default::default()
        };
        let (req, warnings) = build_request("dall-e-3", &opts);
        assert!(matches!(req.response_format, Some(ResponseFormat::B64Json)));
        assert!(warnings.is_empty());
    }

    #[test]
    fn gpt_image_1_omits_response_format_field() {
        let opts = ImageOptions {
            prompt: "hi".into(),
            ..Default::default()
        };
        let (req, _) = build_request("gpt-image-1", &opts);
        assert!(req.response_format.is_none());
    }

    #[test]
    fn aspect_ratio_emits_warning() {
        let opts = ImageOptions {
            prompt: "hi".into(),
            aspect_ratio: Some("16:9".into()),
            ..Default::default()
        };
        let (_, warnings) = build_request("dall-e-3", &opts);
        assert!(matches!(
            &warnings[0],
            Warning::Unsupported { feature, .. } if feature == "aspectRatio"
        ));
    }

    #[test]
    fn seed_emits_warning() {
        let opts = ImageOptions {
            prompt: "hi".into(),
            seed: Some(42),
            ..Default::default()
        };
        let (_, warnings) = build_request("dall-e-3", &opts);
        assert!(warnings.iter().any(|w| matches!(
            w,
            Warning::Unsupported { feature, .. } if feature == "seed"
        )));
    }

    #[test]
    fn provider_options_pass_through_to_wire() {
        let mut anth = Map::new();
        anth.insert("quality".into(), json!("hd"));
        anth.insert("style".into(), json!("vivid"));
        anth.insert("outputFormat".into(), json!("png"));
        let mut po = llmsdk_provider::shared::ProviderOptions::new();
        po.insert(PROVIDER_ID.into(), anth);
        let opts = ImageOptions {
            prompt: "hi".into(),
            provider_options: Some(po),
            ..Default::default()
        };
        let (req, _) = build_request("dall-e-3", &opts);
        assert_eq!(req.quality.as_deref(), Some("hd"));
        assert_eq!(req.style.as_deref(), Some("vivid"));
        assert_eq!(req.output_format.as_deref(), Some("png"));
    }

    #[test]
    fn base64_round_trips_known_vectors() {
        // RFC 4648 test vectors.
        let cases: &[(&str, &[u8])] = &[
            ("", b""),
            ("Zg==", b"f"),
            ("Zm8=", b"fo"),
            ("Zm9v", b"foo"),
            ("Zm9vYg==", b"foob"),
            ("Zm9vYmE=", b"fooba"),
            ("Zm9vYmFy", b"foobar"),
        ];
        for (encoded, raw) in cases {
            let decoded = base64_decode(encoded).expect("valid base64");
            assert_eq!(&decoded, raw, "vector {encoded}");
        }
    }

    #[test]
    fn base64_rejects_invalid_input() {
        assert!(base64_decode("abc").is_err()); // wrong length
        assert!(base64_decode("ab=c").is_err()); // misplaced padding
        assert!(base64_decode("ab!d").is_err()); // bad byte
    }

    #[test]
    fn guess_media_type_uses_server_output_format_first() {
        assert_eq!(guess_media_type(Some("png"), b""), "image/png");
        assert_eq!(guess_media_type(Some("jpeg"), b""), "image/jpeg");
        assert_eq!(guess_media_type(Some("webp"), b""), "image/webp");
    }

    #[test]
    fn guess_media_type_sniffs_png_magic() {
        let png = b"\x89PNG\r\n\x1a\nrest...";
        assert_eq!(guess_media_type(None, png), "image/png");
    }

    #[test]
    fn guess_media_type_sniffs_jpeg_magic() {
        let jpeg = b"\xFF\xD8\xFFrest...";
        assert_eq!(guess_media_type(None, jpeg), "image/jpeg");
    }

    #[test]
    fn guess_media_type_defaults_when_unknown() {
        assert_eq!(guess_media_type(None, b"unknown"), "image/png");
    }

    #[test]
    fn route_endpoint_picks_correct_mode() {
        // text → generate
        assert_eq!(
            route_endpoint(&ImageOptions {
                prompt: "cat".into(),
                ..Default::default()
            }),
            EndpointMode::Generate
        );
        // image + prompt → edit
        let mut with_files = ImageOptions {
            prompt: "make it red".into(),
            ..Default::default()
        };
        with_files.files = Some(vec![FilePart {
            filename: Some("in.png".into()),
            data: FileData::Data {
                data: FileBytes::Bytes(vec![1, 2, 3]),
            },
            media_type: "image/png".into(),
            provider_options: None,
        }]);
        assert_eq!(route_endpoint(&with_files), EndpointMode::Edit);
        // image without prompt → variation
        let mut no_prompt = with_files.clone();
        no_prompt.prompt = String::new();
        assert_eq!(route_endpoint(&no_prompt), EndpointMode::Variation);
    }

    #[test]
    fn into_provider_usage_maps_all_fields() {
        let wire = ImageUsage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            total_tokens: Some(30),
            input_tokens_details: Some(InputTokensDetails {
                image_tokens: Some(5),
                text_tokens: Some(5),
            }),
        };
        let mapped = into_provider_usage(&wire);
        assert_eq!(mapped.input_tokens, Some(10));
        assert_eq!(mapped.output_tokens, Some(20));
        let det = mapped.input_tokens_details.unwrap();
        assert_eq!(det.image_tokens, Some(5));
        assert_eq!(det.text_tokens, Some(5));
    }
}
