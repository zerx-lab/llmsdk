//! [`ImageModel`] implementation for Bedrock (Nova Canvas + family).
// Rust guideline compliant 2026-05-25

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use llmsdk_provider::ProviderError;
use llmsdk_provider::image_model::{GeneratedImage, ImageModel, ImageOptions, ImageResult};
use llmsdk_provider::language_model::FilePart;
use llmsdk_provider::shared::{FileBytes, FileData, ResponseInfo, Warning};
use llmsdk_provider_utils::http::{RawRequest, post_raw};
use reqwest::Method;
use serde_json::{Map, Value};

use super::options::parse as parse_options;
use super::wire::ImageResponse;
use crate::PROVIDER_ID;
use crate::config::Inner;

/// Bedrock image model handle.
#[derive(Debug, Clone)]
pub struct AmazonBedrockImageModel {
    inner: Arc<Inner>,
    model_id: String,
}

impl AmazonBedrockImageModel {
    pub(crate) fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn url(&self) -> String {
        let encoded = crate::chat::encode_path_segment(&self.model_id);
        format!("{}/model/{}/invoke", self.inner.runtime_base_url, encoded)
    }

    fn max_images_static(&self) -> u32 {
        match self.model_id.as_str() {
            "amazon.nova-canvas-v1:0" => 5,
            _ => 1,
        }
    }
}

#[async_trait]
impl ImageModel for AmazonBedrockImageModel {
    fn provider(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_images_per_call(&self) -> Option<u32> {
        Some(self.max_images_static())
    }

    async fn do_generate(&self, options: ImageOptions) -> Result<ImageResult, ProviderError> {
        let mut warnings: Vec<Warning> = Vec::new();
        let (width, height) = options
            .size
            .as_deref()
            .and_then(|s| {
                let mut parts = s.splitn(2, 'x');
                let w = parts.next()?.parse::<u32>().ok()?;
                let h = parts.next()?.parse::<u32>().ok()?;
                Some((Some(w), Some(h)))
            })
            .unwrap_or((None, None));

        let bedrock_opts = parse_options(options.provider_options.as_ref());

        if options.aspect_ratio.is_some() {
            warnings.push(Warning::Unsupported {
                feature: "aspectRatio".into(),
                details: Some("Bedrock image models do not accept aspect ratio; use size.".into()),
            });
        }

        let mut image_generation_config = Map::new();
        if let Some(w) = width {
            image_generation_config.insert("width".to_owned(), Value::from(w));
        }
        if let Some(h) = height {
            image_generation_config.insert("height".to_owned(), Value::from(h));
        }
        if let Some(seed) = options.seed {
            image_generation_config.insert("seed".to_owned(), Value::from(seed));
        }
        if let Some(n) = options.n {
            image_generation_config.insert("numberOfImages".to_owned(), Value::from(n));
        }
        if let Some(q) = &bedrock_opts.quality {
            image_generation_config.insert("quality".to_owned(), Value::String(q.clone()));
        }
        if let Some(cfg) = bedrock_opts.cfg_scale {
            image_generation_config.insert("cfgScale".to_owned(), Value::from(cfg));
        }

        let has_files = options.files.as_ref().is_some_and(|f| !f.is_empty());
        let mut args = Map::new();

        if has_files {
            let files = options.files.as_ref().unwrap();
            let has_mask = options.mask.is_some();
            let has_mask_prompt = bedrock_opts.mask_prompt.is_some();
            let task_type = bedrock_opts.task_type.clone().unwrap_or_else(|| {
                if has_mask || has_mask_prompt {
                    "INPAINTING".to_owned()
                } else {
                    "IMAGE_VARIATION".to_owned()
                }
            });
            let source_b64 = file_to_base64(&files[0])?;
            match task_type.as_str() {
                "INPAINTING" => {
                    let mut params = Map::new();
                    params.insert("image".to_owned(), Value::String(source_b64));
                    if !options.prompt.is_empty() {
                        params.insert("text".to_owned(), Value::String(options.prompt.clone()));
                    }
                    if let Some(neg) = &bedrock_opts.negative_text {
                        params.insert("negativeText".to_owned(), Value::String(neg.clone()));
                    }
                    if has_mask {
                        let mask_b64 = file_to_base64(options.mask.as_ref().unwrap())?;
                        params.insert("maskImage".to_owned(), Value::String(mask_b64));
                    } else if let Some(mp) = &bedrock_opts.mask_prompt {
                        params.insert("maskPrompt".to_owned(), Value::String(mp.clone()));
                    }
                    args.insert("taskType".to_owned(), Value::String("INPAINTING".into()));
                    args.insert("inPaintingParams".to_owned(), Value::Object(params));
                    args.insert(
                        "imageGenerationConfig".to_owned(),
                        Value::Object(image_generation_config),
                    );
                }
                "OUTPAINTING" => {
                    let mut params = Map::new();
                    params.insert("image".to_owned(), Value::String(source_b64));
                    if !options.prompt.is_empty() {
                        params.insert("text".to_owned(), Value::String(options.prompt.clone()));
                    }
                    if let Some(neg) = &bedrock_opts.negative_text {
                        params.insert("negativeText".to_owned(), Value::String(neg.clone()));
                    }
                    if let Some(mode) = &bedrock_opts.out_painting_mode {
                        params.insert("outPaintingMode".to_owned(), Value::String(mode.clone()));
                    }
                    if has_mask {
                        let mask_b64 = file_to_base64(options.mask.as_ref().unwrap())?;
                        params.insert("maskImage".to_owned(), Value::String(mask_b64));
                    } else if let Some(mp) = &bedrock_opts.mask_prompt {
                        params.insert("maskPrompt".to_owned(), Value::String(mp.clone()));
                    }
                    args.insert("taskType".to_owned(), Value::String("OUTPAINTING".into()));
                    args.insert("outPaintingParams".to_owned(), Value::Object(params));
                    args.insert(
                        "imageGenerationConfig".to_owned(),
                        Value::Object(image_generation_config),
                    );
                }
                "BACKGROUND_REMOVAL" => {
                    let mut params = Map::new();
                    params.insert("image".to_owned(), Value::String(source_b64));
                    args.insert(
                        "taskType".to_owned(),
                        Value::String("BACKGROUND_REMOVAL".into()),
                    );
                    args.insert("backgroundRemovalParams".to_owned(), Value::Object(params));
                }
                _ => {
                    let mut params = Map::new();
                    let mut images: Vec<Value> = Vec::with_capacity(files.len());
                    images.push(Value::String(source_b64));
                    for extra in files.iter().skip(1) {
                        images.push(Value::String(file_to_base64(extra)?));
                    }
                    params.insert("images".to_owned(), Value::Array(images));
                    if !options.prompt.is_empty() {
                        params.insert("text".to_owned(), Value::String(options.prompt.clone()));
                    }
                    if let Some(neg) = &bedrock_opts.negative_text {
                        params.insert("negativeText".to_owned(), Value::String(neg.clone()));
                    }
                    if let Some(strength) = bedrock_opts.similarity_strength {
                        params.insert("similarityStrength".to_owned(), Value::from(strength));
                    }
                    args.insert(
                        "taskType".to_owned(),
                        Value::String("IMAGE_VARIATION".into()),
                    );
                    args.insert("imageVariationParams".to_owned(), Value::Object(params));
                    args.insert(
                        "imageGenerationConfig".to_owned(),
                        Value::Object(image_generation_config),
                    );
                }
            }
        } else {
            let mut text_to_image = Map::new();
            text_to_image.insert("text".to_owned(), Value::String(options.prompt.clone()));
            if let Some(neg) = &bedrock_opts.negative_text {
                text_to_image.insert("negativeText".to_owned(), Value::String(neg.clone()));
            }
            if let Some(style) = &bedrock_opts.style {
                text_to_image.insert("style".to_owned(), Value::String(style.clone()));
            }
            args.insert("taskType".to_owned(), Value::String("TEXT_IMAGE".into()));
            args.insert("textToImageParams".to_owned(), Value::Object(text_to_image));
            args.insert(
                "imageGenerationConfig".to_owned(),
                Value::Object(image_generation_config),
            );
        }

        let body_bytes = serde_json::to_vec(&Value::Object(args))
            .map_err(|e| ProviderError::json_parse("<bedrock-image-request>", e.to_string()))?;
        let url = self.url();
        let mut headers = self.inner.extra_headers.clone();
        if let Some(per_call) = options.headers.as_ref() {
            for (k, v) in per_call {
                headers.insert(k.clone(), v.clone());
            }
        }
        self.inner
            .auth
            .apply(&mut headers, &Method::POST, &url, &body_bytes)
            .await?;

        let mut raw = RawRequest::new(url.clone(), body_bytes, "application/json");
        raw.headers = headers;
        let response = post_raw::<ImageResponse>(&self.inner.http, raw).await?;
        let value = response.value;

        if value.status.as_deref() == Some("Request Moderated") {
            let mut reasons = "Unknown".to_owned();
            if let Some(details) = value.details.as_ref()
                && let Some(arr) = details.get("Moderation Reasons").and_then(Value::as_array)
            {
                reasons = arr
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ");
            }
            return Err(ProviderError::api_call_builder(
                &url,
                format!("Amazon Bedrock request was moderated: {reasons}"),
            )
            .build());
        }

        let images_b64 = value.images.unwrap_or_default();
        if images_b64.is_empty() {
            return Err(ProviderError::api_call_builder(
                &url,
                "Amazon Bedrock returned no images.",
            )
            .build());
        }

        let mut generated: Vec<GeneratedImage> = Vec::with_capacity(images_b64.len());
        for b64 in images_b64 {
            let bytes = decode_base64(&b64).map_err(|e| {
                ProviderError::api_call_builder(
                    &url,
                    format!("invalid base64 in Bedrock image response: {e}"),
                )
                .build()
            })?;
            generated.push(GeneratedImage {
                bytes: Bytes::from(bytes),
                media_type: "image/png".to_owned(),
            });
        }

        Ok(ImageResult {
            images: generated,
            warnings,
            usage: None,
            provider_metadata: None,
            request: None,
            response: Some(ResponseInfo {
                headers: Some(
                    response
                        .headers
                        .into_iter()
                        .map(|(k, v)| (k, Some(v)))
                        .collect(),
                ),
                model_id: Some(self.model_id.clone()),
                ..ResponseInfo::default()
            }),
        })
    }
}

fn file_to_base64(file: &FilePart) -> Result<String, ProviderError> {
    match &file.data {
        FileData::Data { data } => Ok(match data {
            FileBytes::Base64(s) => s.clone(),
            FileBytes::Bytes(b) => crate::chat::base64_encode_public(b),
        }),
        FileData::Url { .. } => Err(ProviderError::invalid_argument(
            "files",
            "Amazon Bedrock image editing requires inline image bytes; URL-sourced files are not supported.",
        )),
        FileData::Reference { .. } => Err(ProviderError::invalid_argument(
            "files",
            "Amazon Bedrock image editing requires inline image bytes; provider references are not supported.",
        )),
        FileData::Text { .. } => Err(ProviderError::invalid_argument(
            "files",
            "Amazon Bedrock image editing requires inline image bytes; text inputs are not supported.",
        )),
    }
}

/// Minimal base64 decoder. Mirrors `convertBase64ToUint8Array` upstream.
fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Result<u8, String> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'),
            b'a'..=b'z' => Ok(c - b'a' + 26),
            b'0'..=b'9' => Ok(c - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            b'=' => Ok(0xff),
            _ => Err(format!("invalid base64 byte: {c}")),
        }
    }
    let clean: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if !clean.len().is_multiple_of(4) {
        return Err("base64 input length is not a multiple of 4".into());
    }
    let mut out = Vec::with_capacity(clean.len() / 4 * 3);
    for chunk in clean.chunks(4) {
        let b0 = val(chunk[0])?;
        let b1 = val(chunk[1])?;
        let b2 = val(chunk[2])?;
        let b3 = val(chunk[3])?;
        out.push((b0 << 2) | (b1 >> 4));
        if b2 != 0xff {
            out.push(((b1 & 0x0f) << 4) | (b2 >> 2));
        }
        if b3 != 0xff {
            out.push(((b2 & 0x03) << 6) | b3);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_round_trip() {
        let input = "Zm9vYmFy";
        let bytes = decode_base64(input).unwrap();
        assert_eq!(bytes, b"foobar");
    }
}
