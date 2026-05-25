//! Vertex AI image generation (Imagen `:predict` + Gemini image
//! delegation via the language model).
//!
//! Mirrors `google-vertex-image-model.ts`. Two backends:
//!
//! 1. Imagen (`imagen-*`): `POST {publishers/google}/models/{id}:predict`
//!    with Vertex's `instances[]` + `parameters` wire and optional edit
//!    `referenceImages[]` payload.
//! 2. Gemini image-output (`gemini-*-image*` / `gemini-2.5-flash-image`):
//!    delegated to the language model with
//!    `provider_options.googleVertex.responseModalities=["IMAGE"]`.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use llmsdk_provider::ProviderError;
use llmsdk_provider::image_model::{
    GeneratedImage, ImageModel, ImageOptions, ImageResult, ImageUsage,
};
use llmsdk_provider::language_model::{
    CallOptions, Content, FilePart, LanguageModel, Message, UserPart,
};
use llmsdk_provider::shared::{
    FileBytes, FileData, Headers, ProviderMetadata, ProviderOptions, ResponseInfo, Warning,
};
use llmsdk_provider_utils::http::{JsonRequest, post_json};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::PROVIDER_ID_IMAGE;
use crate::auth::cloud_platform_token;
use crate::config::{VertexAuthMode, VertexInner};
use crate::language::GoogleVertexLanguageModel;

const IMAGEN_MAX_PER_CALL: u32 = 4;
const GEMINI_IMAGE_MAX_PER_CALL: u32 = 10;

/// Vertex image-model handle.
#[derive(Debug, Clone)]
pub struct GoogleVertexImageModel {
    inner: Arc<VertexInner>,
    model_id: String,
}

impl GoogleVertexImageModel {
    pub(crate) fn new(inner: Arc<VertexInner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn is_gemini(&self) -> bool {
        self.model_id.starts_with("gemini-")
    }

    async fn merged_headers(
        &self,
        per_call: Option<&Headers>,
    ) -> Result<HashMap<String, Option<String>>, ProviderError> {
        let mut headers = self.inner.extra_headers.clone();
        match &self.inner.auth {
            VertexAuthMode::Express { api_key } => {
                headers.insert("x-goog-api-key".into(), Some(api_key.clone()));
            }
            VertexAuthMode::OAuth { token_provider, .. } => {
                let token = cloud_platform_token(token_provider.as_ref()).await?;
                headers.insert("Authorization".into(), Some(format!("Bearer {token}")));
            }
        }
        if let Some(h) = per_call {
            for (k, v) in h {
                headers.insert(k.clone(), v.clone());
            }
        }
        Ok(headers)
    }
}

#[async_trait]
impl ImageModel for GoogleVertexImageModel {
    fn provider(&self) -> &str {
        PROVIDER_ID_IMAGE
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_images_per_call(&self) -> Option<u32> {
        if self.is_gemini() {
            Some(GEMINI_IMAGE_MAX_PER_CALL)
        } else {
            Some(IMAGEN_MAX_PER_CALL)
        }
    }

    async fn do_generate(&self, options: ImageOptions) -> Result<ImageResult, ProviderError> {
        if self.is_gemini() {
            self.do_generate_gemini(options).await
        } else {
            self.do_generate_imagen(options).await
        }
    }
}

impl GoogleVertexImageModel {
    async fn do_generate_imagen(
        &self,
        options: ImageOptions,
    ) -> Result<ImageResult, ProviderError> {
        let mut warnings: Vec<Warning> = Vec::new();
        if options.size.is_some() {
            warnings.push(Warning::UnsupportedSetting {
                setting: "size".into(),
                details: Some(
                    "This model does not support the `size` option. Use `aspectRatio` instead."
                        .into(),
                ),
            });
        }

        let vertex_opts =
            parse_image_options(options.provider_options.as_ref())?.unwrap_or_default();
        let is_edit_mode = options.files.as_ref().is_some_and(|v| !v.is_empty());

        let mut parameters = Map::new();
        if let Some(n) = options.n {
            parameters.insert("sampleCount".into(), Value::from(n));
        }
        if let Some(ar) = &options.aspect_ratio {
            parameters.insert("aspectRatio".into(), Value::String(ar.clone()));
        }
        if let Some(seed) = options.seed {
            parameters.insert("seed".into(), Value::from(seed));
        }
        if let Some(np) = &vertex_opts.negative_prompt {
            parameters.insert("negativePrompt".into(), Value::String(np.clone()));
        }
        if let Some(pg) = &vertex_opts.person_generation {
            parameters.insert("personGeneration".into(), Value::String(pg.clone()));
        }
        if let Some(ss) = &vertex_opts.safety_setting {
            parameters.insert("safetySetting".into(), Value::String(ss.clone()));
        }
        if let Some(aw) = vertex_opts.add_watermark {
            parameters.insert("addWatermark".into(), Value::Bool(aw));
        }
        if let Some(uri) = &vertex_opts.storage_uri {
            parameters.insert("storageUri".into(), Value::String(uri.clone()));
        }
        if let Some(sz) = &vertex_opts.sample_image_size {
            parameters.insert("sampleImageSize".into(), Value::String(sz.clone()));
        }
        for (k, v) in &vertex_opts.extras {
            parameters.insert(k.clone(), v.clone());
        }

        let mut instance = Map::new();
        instance.insert("prompt".into(), Value::String(options.prompt.clone()));

        if is_edit_mode {
            let mut reference_images: Vec<Value> = Vec::new();
            let mut next_id = 1u32;
            if let Some(files) = &options.files {
                for file in files {
                    reference_images.push(reference_image_object(
                        "REFERENCE_TYPE_RAW",
                        next_id,
                        file_base64(file)?,
                        None,
                    ));
                    next_id += 1;
                }
            }
            if let Some(mask) = &options.mask {
                let mut mask_config = Map::new();
                let mode = vertex_opts
                    .edit
                    .as_ref()
                    .and_then(|e| e.mask_mode.clone())
                    .unwrap_or_else(|| "MASK_MODE_USER_PROVIDED".into());
                mask_config.insert("maskMode".into(), Value::String(mode));
                if let Some(d) = vertex_opts.edit.as_ref().and_then(|e| e.mask_dilation) {
                    mask_config.insert("dilation".into(), Value::from(d));
                }
                reference_images.push(reference_image_object(
                    "REFERENCE_TYPE_MASK",
                    next_id,
                    file_base64(mask)?,
                    Some(Value::Object(mask_config)),
                ));
            }
            instance.insert("referenceImages".into(), Value::Array(reference_images));

            let edit_mode = vertex_opts
                .edit
                .as_ref()
                .and_then(|e| e.mode.clone())
                .unwrap_or_else(|| "EDIT_MODE_INPAINT_INSERTION".into());
            parameters.insert("editMode".into(), Value::String(edit_mode));
            if let Some(steps) = vertex_opts.edit.as_ref().and_then(|e| e.base_steps) {
                let mut edit_config = Map::new();
                edit_config.insert("baseSteps".into(), Value::from(steps));
                parameters.insert("editConfig".into(), Value::Object(edit_config));
            }
        }

        let mut body = Map::new();
        body.insert(
            "instances".into(),
            Value::Array(vec![Value::Object(instance)]),
        );
        body.insert("parameters".into(), Value::Object(parameters));

        let url = format!(
            "{}/models/{}:predict",
            self.inner.publishers_google_base(),
            self.model_id
        );
        let mut req = JsonRequest::new(url, Value::Object(body));
        req.headers = self.merged_headers(options.headers.as_ref()).await?;

        let envelope = post_json::<_, ImagenPredictResponse>(&self.inner.http, req).await?;
        let predictions = envelope.value.predictions.unwrap_or_default();

        let mut images: Vec<GeneratedImage> = Vec::with_capacity(predictions.len());
        let mut image_meta: Vec<Value> = Vec::new();
        for pred in predictions {
            let decoded = decode_base64(&pred.bytes_base64_encoded)?;
            images.push(GeneratedImage {
                bytes: Bytes::from(decoded),
                media_type: pred.mime_type.unwrap_or_else(|| "image/png".into()),
            });
            let mut meta = Map::new();
            if let Some(p) = pred.prompt {
                meta.insert("revisedPrompt".into(), Value::String(p));
            }
            image_meta.push(Value::Object(meta));
        }

        let mut vertex_payload = Map::new();
        vertex_payload.insert("images".into(), Value::Array(image_meta));
        let mut provider_metadata = ProviderMetadata::new();
        provider_metadata.insert("googleVertex".into(), vertex_payload.clone());
        provider_metadata.insert("vertex".into(), vertex_payload);

        Ok(ImageResult {
            images,
            warnings,
            usage: None,
            provider_metadata: Some(provider_metadata),
            request: None,
            response: Some(ResponseInfo {
                model_id: Some(self.model_id.clone()),
                headers: Some(
                    envelope
                        .headers
                        .into_iter()
                        .map(|(k, v)| (k, Some(v)))
                        .collect(),
                ),
                ..Default::default()
            }),
        })
    }

    async fn do_generate_gemini(
        &self,
        options: ImageOptions,
    ) -> Result<ImageResult, ProviderError> {
        let mut warnings: Vec<Warning> = Vec::new();
        if options.mask.is_some() {
            return Err(ProviderError::unsupported(
                "Gemini image models do not support mask-based image editing.",
            ));
        }
        if let Some(n) = options.n
            && n > 1
        {
            return Err(ProviderError::unsupported(
                "Gemini image models do not support generating a set number of images per call. Use n=1.",
            ));
        }
        if options.size.is_some() {
            warnings.push(Warning::UnsupportedSetting {
                setting: "size".into(),
                details: Some("Use `aspectRatio` instead.".into()),
            });
        }

        let vertex_opts =
            parse_image_options(options.provider_options.as_ref())?.unwrap_or_default();
        let user_content = build_user_content(&options);

        // googleVertex passthrough → language model
        let mut vertex_po = Map::new();
        vertex_po.insert(
            "responseModalities".into(),
            Value::Array(vec![Value::String("IMAGE".into())]),
        );
        if let Some(ar) = options.aspect_ratio.clone() {
            let mut ic = Map::new();
            ic.insert("aspectRatio".into(), Value::String(ar));
            vertex_po.insert("imageConfig".into(), Value::Object(ic));
        }
        for (k, v) in &vertex_opts.extras {
            vertex_po.insert(k.clone(), v.clone());
        }
        if let Some(pg) = &vertex_opts.person_generation {
            vertex_po.insert("personGeneration".into(), Value::String(pg.clone()));
        }
        let mut provider_options = ProviderOptions::new();
        provider_options.insert("googleVertex".into(), vertex_po.clone());
        provider_options.insert("vertex".into(), vertex_po);

        let call = CallOptions {
            prompt: vec![Message::User {
                content: user_content,
                provider_options: None,
            }],
            seed: options.seed,
            provider_options: Some(provider_options),
            tools: None,
            tool_choice: None,
            headers: options.headers.clone(),
            include_raw_chunks: None,
            reasoning: None,
            response_format: None,
            ..Default::default()
        };

        let lm = GoogleVertexLanguageModel::new(Arc::clone(&self.inner), self.model_id.clone());
        let result = lm.do_generate(call).await?;

        let mut images: Vec<GeneratedImage> = Vec::new();
        for c in result.content {
            if let Content::File(f) = c
                && f.media_type.starts_with("image/")
            {
                let bytes = match f.data {
                    FileData::Data {
                        data: FileBytes::Base64(s),
                    } => Bytes::from(decode_base64(&s)?),
                    FileData::Data {
                        data: FileBytes::Bytes(b),
                    } => Bytes::from(b),
                    _ => continue,
                };
                images.push(GeneratedImage {
                    bytes,
                    media_type: f.media_type,
                });
            }
        }

        let mut image_meta = Vec::new();
        for _ in &images {
            image_meta.push(Value::Object(Map::new()));
        }
        let mut payload = Map::new();
        payload.insert("images".into(), Value::Array(image_meta));
        let mut provider_metadata = ProviderMetadata::new();
        provider_metadata.insert("googleVertex".into(), payload.clone());
        provider_metadata.insert("vertex".into(), payload);

        let usage = result.usage;
        let total_in = usage.input_tokens.total;
        let total_out = usage.output_tokens.total;
        let image_usage = if total_in.is_some() || total_out.is_some() {
            Some(ImageUsage {
                input_tokens: total_in,
                output_tokens: total_out,
                input_tokens_details: None,
            })
        } else {
            None
        };

        Ok(ImageResult {
            images,
            warnings,
            usage: image_usage,
            provider_metadata: Some(provider_metadata),
            request: None,
            response: Some(ResponseInfo {
                model_id: Some(self.model_id.clone()),
                headers: result
                    .response
                    .as_ref()
                    .and_then(|r| r.metadata.headers.clone()),
                ..Default::default()
            }),
        })
    }
}

fn reference_image_object(
    kind: &str,
    id: u32,
    bytes_b64: String,
    mask_config: Option<Value>,
) -> Value {
    let mut img = Map::new();
    img.insert("bytesBase64Encoded".into(), Value::String(bytes_b64));
    let mut out = Map::new();
    out.insert("referenceType".into(), Value::String(kind.into()));
    out.insert("referenceId".into(), Value::from(id));
    out.insert("referenceImage".into(), Value::Object(img));
    if let Some(m) = mask_config {
        out.insert("maskImageConfig".into(), m);
    }
    Value::Object(out)
}

fn file_base64(file: &FilePart) -> Result<String, ProviderError> {
    match &file.data {
        FileData::Data {
            data: FileBytes::Base64(s),
        } => Ok(s.clone()),
        FileData::Data {
            data: FileBytes::Bytes(b),
        } => Ok(encode_base64(b)),
        FileData::Url { .. } | FileData::Reference { .. } | FileData::Text { .. } => {
            Err(ProviderError::unsupported(
                "URL- / reference- / text- based files are not supported for Google Vertex image editing. \
             Please provide raw image bytes or a base64 string directly.",
            ))
        }
    }
}

fn build_user_content(options: &ImageOptions) -> Vec<UserPart> {
    let mut out: Vec<UserPart> = Vec::new();
    if !options.prompt.is_empty() {
        out.push(UserPart::Text(llmsdk_provider::language_model::TextPart {
            text: options.prompt.clone(),
            provider_options: None,
        }));
    }
    if let Some(files) = &options.files {
        for f in files {
            out.push(UserPart::File(FilePart {
                filename: f.filename.clone(),
                data: f.data.clone(),
                media_type: if f.media_type.is_empty() {
                    "image/*".into()
                } else {
                    f.media_type.clone()
                },
                provider_options: None,
            }));
        }
    }
    out
}

fn decode_base64(s: &str) -> Result<Vec<u8>, ProviderError> {
    // Minimal RFC 4648 §4 decoder (no padding tolerance variations).
    let clean: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    let table = base64_table();
    let mut out: Vec<u8> = Vec::with_capacity(clean.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut buf_bits: u32 = 0;
    for ch in clean.chars() {
        if ch == '=' {
            break;
        }
        let val = table
            .iter()
            .position(|c| *c == ch)
            .ok_or_else(|| ProviderError::json_parse(s.to_owned(), "invalid base64 character"))?;
        // `val` is bounded by 63 (table length 64); u32 conversion is lossless.
        buf = (buf << 6) | u32::try_from(val).unwrap_or(0);
        buf_bits += 6;
        if buf_bits >= 8 {
            buf_bits -= 8;
            out.push(u8::try_from((buf >> buf_bits) & 0xff).unwrap_or(0));
        }
    }
    Ok(out)
}

/// Re-export for sibling modules (video).
pub(crate) fn encode_base64_for_video(bytes: &[u8]) -> String {
    encode_base64(bytes)
}

fn encode_base64(bytes: &[u8]) -> String {
    let table = base64_table();
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let b0 = bytes[i];
        let b1 = bytes[i + 1];
        let b2 = bytes[i + 2];
        out.push(table[(b0 >> 2) as usize]);
        out.push(table[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize]);
        out.push(table[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize]);
        out.push(table[(b2 & 0x3f) as usize]);
        i += 3;
    }
    match bytes.len() - i {
        1 => {
            let b0 = bytes[i];
            out.push(table[(b0 >> 2) as usize]);
            out.push(table[((b0 & 0x03) << 4) as usize]);
            out.push('=');
            out.push('=');
        }
        2 => {
            let b0 = bytes[i];
            let b1 = bytes[i + 1];
            out.push(table[(b0 >> 2) as usize]);
            out.push(table[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize]);
            out.push(table[((b1 & 0x0f) << 2) as usize]);
            out.push('=');
        }
        _ => {}
    }
    out
}

fn base64_table() -> [char; 64] {
    let alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = ['A'; 64];
    for (i, c) in alphabet.chars().enumerate() {
        out[i] = c;
    }
    out
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct VertexImageOptions {
    #[serde(default, rename = "negativePrompt")]
    negative_prompt: Option<String>,
    #[serde(default, rename = "personGeneration")]
    person_generation: Option<String>,
    #[serde(default, rename = "safetySetting")]
    safety_setting: Option<String>,
    #[serde(default, rename = "addWatermark")]
    add_watermark: Option<bool>,
    #[serde(default, rename = "storageUri")]
    storage_uri: Option<String>,
    #[serde(default, rename = "sampleImageSize")]
    sample_image_size: Option<String>,
    #[serde(default)]
    edit: Option<VertexImageEditOptions>,
    /// Pass-through for keys not modeled explicitly.
    #[serde(flatten)]
    extras: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct VertexImageEditOptions {
    #[serde(default, rename = "baseSteps")]
    base_steps: Option<u32>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default, rename = "maskMode")]
    mask_mode: Option<String>,
    #[serde(default, rename = "maskDilation")]
    mask_dilation: Option<f32>,
}

fn parse_image_options(
    provider_options: Option<&ProviderOptions>,
) -> Result<Option<VertexImageOptions>, ProviderError> {
    let Some(opts) = provider_options else {
        return Ok(None);
    };
    for key in ["googleVertex", "vertex"] {
        if let Some(payload) = opts.get(key) {
            let value = Value::Object(payload.clone());
            let parsed: VertexImageOptions =
                serde_json::from_value(value.clone()).map_err(|e| {
                    ProviderError::type_validation(
                        format!("provider_options.{key}"),
                        value,
                        e.to_string(),
                    )
                })?;
            return Ok(Some(parsed));
        }
    }
    Ok(None)
}

#[derive(Debug, Clone, Deserialize)]
struct ImagenPredictResponse {
    #[serde(default)]
    predictions: Option<Vec<ImagenPrediction>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ImagenPrediction {
    #[serde(rename = "bytesBase64Encoded")]
    bytes_base64_encoded: String,
    #[serde(default, rename = "mimeType")]
    mime_type: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trip() {
        let original = b"hello world!";
        let encoded = encode_base64(original);
        let decoded = decode_base64(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn base64_decodes_padded() {
        // "Man" → "TWFu"
        assert_eq!(decode_base64("TWFu").unwrap(), b"Man");
    }
}
