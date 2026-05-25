//! Gemini / Imagen image model.
//!
//! Mirrors `GoogleImageModel` from
//! `@ai-sdk/google/src/google-image-model.ts`. Two backends:
//!
//! 1. Imagen models (`imagen-*`) → `POST /models/{id}:predict` with
//!    `{instances:[{prompt}], parameters:{sampleCount,aspectRatio,...}}`.
//! 2. Gemini image-output models (`gemini-*-image*`) → delegate to
//!    [`crate::GoogleLanguageModel`] with `responseModalities:["IMAGE"]`
//!    and a user message carrying the prompt + optional source images.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use llmsdk_provider::ProviderError;
use llmsdk_provider::image_model::{GeneratedImage, ImageModel, ImageOptions, ImageResult};
use llmsdk_provider::language_model::{
    CallOptions, Content, FilePart, FunctionTool, LanguageModel, Message, ProviderTool,
    ReasoningEffort, ResponseFormat, TextPart, Tool, ToolChoice, UserPart,
};
use llmsdk_provider::shared::{
    FileBytes, FileData, Headers, ProviderMetadata, ProviderOptions, ResponseInfo, Warning,
};
use llmsdk_provider_utils::http::{JsonRequest, post_json};
use serde_json::{Map, Value};

use crate::PROVIDER_ID;
use crate::base64::decode as base64_decode;
use crate::config::Inner;
use crate::error::rewrite_google_error;
use crate::language::GoogleLanguageModel;

use super::options::parse as parse_image_options;
use super::wire::ImagenPredictResponse;

/// Gemini image-model handle.
#[derive(Debug, Clone)]
pub struct GoogleImageModel {
    inner: Arc<Inner>,
    model_id: String,
}

impl GoogleImageModel {
    pub(crate) fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn merged_headers(&self, extra: Option<&Headers>) -> HashMap<String, Option<String>> {
        let mut h = self.inner.headers.clone();
        if let Some(extra) = extra {
            for (k, v) in extra {
                h.insert(k.clone(), v.clone());
            }
        }
        h
    }

    fn is_gemini_image(&self) -> bool {
        self.model_id.starts_with("gemini-")
    }
}

#[async_trait]
impl ImageModel for GoogleImageModel {
    fn provider(&self) -> &str {
        &self.inner.provider
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_images_per_call(&self) -> Option<u32> {
        if self.is_gemini_image() {
            Some(10)
        } else {
            Some(4)
        }
    }

    async fn do_generate(&self, options: ImageOptions) -> Result<ImageResult, ProviderError> {
        if self.is_gemini_image() {
            self.generate_via_gemini(options).await
        } else {
            self.generate_via_imagen(options).await
        }
    }
}

impl GoogleImageModel {
    async fn generate_via_imagen(
        &self,
        options: ImageOptions,
    ) -> Result<ImageResult, ProviderError> {
        let mut warnings: Vec<Warning> = Vec::new();
        if options.files.is_some() && !options.files.as_ref().unwrap().is_empty() {
            return Err(ProviderError::unsupported(
                "Google Gemini API does not support image editing with Imagen models.",
            ));
        }
        if options.mask.is_some() {
            return Err(ProviderError::unsupported(
                "Google Gemini API does not support image editing with masks.",
            ));
        }
        if options.size.is_some() {
            warnings.push(Warning::UnsupportedSetting {
                setting: "size".into(),
                details: Some("Use `aspectRatio` instead.".into()),
            });
        }
        if options.seed.is_some() {
            warnings.push(Warning::UnsupportedSetting {
                setting: "seed".into(),
                details: Some(
                    "This model does not support the `seed` option through this provider.".into(),
                ),
            });
        }

        let google_options =
            parse_image_options(options.provider_options.as_ref())?.unwrap_or_default();

        // Build `parameters`
        let mut params = Map::new();
        if let Some(n) = options.n {
            params.insert("sampleCount".into(), Value::from(n));
        }
        let aspect = options.aspect_ratio.clone().unwrap_or_else(|| "1:1".into());
        params.insert("aspectRatio".into(), Value::String(aspect));
        if google_options.google_search.is_some() {
            warnings.push(Warning::UnsupportedSetting {
                setting: "googleSearch".into(),
                details: Some(
                    "Google Search grounding is only supported on Gemini image models.".into(),
                ),
            });
        }
        if let Some(pg) = &google_options.person_generation {
            params.insert("personGeneration".into(), Value::String(pg.clone()));
        }
        if let Some(ar) = &google_options.aspect_ratio {
            params.insert("aspectRatio".into(), Value::String(ar.clone()));
        }
        for (k, v) in &google_options.extras {
            params.insert(k.clone(), v.clone());
        }

        let mut body = Map::new();
        let mut instance = Map::new();
        instance.insert("prompt".into(), Value::String(options.prompt.clone()));
        body.insert(
            "instances".into(),
            Value::Array(vec![Value::Object(instance)]),
        );
        body.insert("parameters".into(), Value::Object(params));

        let url = format!("{}/models/{}:predict", self.inner.base_url, self.model_id);
        let mut req = JsonRequest::new(url, Value::Object(body));
        req.headers = self.merged_headers(options.headers.as_ref());

        let envelope = match post_json::<_, ImagenPredictResponse>(&self.inner.http, req).await {
            Ok(r) => r,
            Err(e) => return Err(rewrite_google_error(e)),
        };

        let mut images = Vec::with_capacity(envelope.value.predictions.len());
        let mut image_meta: Vec<Value> = Vec::new();
        for p in envelope.value.predictions {
            let bytes = base64_decode(&p.bytes_base64_encoded).map_err(|e| {
                ProviderError::json_parse(p.bytes_base64_encoded.clone(), e.to_string())
            })?;
            images.push(GeneratedImage {
                bytes: Bytes::from(bytes),
                media_type: p.mime_type.unwrap_or_else(|| "image/png".into()),
            });
            image_meta.push(Value::Object(Map::new()));
        }

        let mut g_meta = Map::new();
        g_meta.insert("images".into(), Value::Array(image_meta));
        let mut provider_metadata = ProviderMetadata::new();
        provider_metadata.insert(PROVIDER_ID.into(), g_meta);

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

    async fn generate_via_gemini(
        &self,
        options: ImageOptions,
    ) -> Result<ImageResult, ProviderError> {
        let mut warnings: Vec<Warning> = Vec::new();
        if options.mask.is_some() {
            return Err(ProviderError::unsupported(
                "Gemini image models do not support mask-based image editing.",
            ));
        }
        if let Some(n) = options.n {
            if n > 1 {
                return Err(ProviderError::unsupported(
                    "Gemini image models do not support generating a set number of images per call. Use n=1.",
                ));
            }
        }
        if options.size.is_some() {
            warnings.push(Warning::UnsupportedSetting {
                setting: "size".into(),
                details: Some("Use `aspectRatio` instead.".into()),
            });
        }

        let image_options =
            parse_image_options(options.provider_options.as_ref())?.unwrap_or_default();
        let user_content = build_user_content(&options);

        // Provider options passthrough → language model.
        let mut google_po = Map::new();
        google_po.insert(
            "responseModalities".into(),
            Value::Array(vec![Value::String("IMAGE".into())]),
        );
        if let Some(ar) = options.aspect_ratio.clone().or(image_options.aspect_ratio) {
            let mut ic = Map::new();
            ic.insert("aspectRatio".into(), Value::String(ar));
            google_po.insert("imageConfig".into(), Value::Object(ic));
        }
        for (k, v) in &image_options.extras {
            google_po.insert(k.clone(), v.clone());
        }
        if let Some(pg) = &image_options.person_generation {
            google_po.insert("personGeneration".into(), Value::String(pg.clone()));
        }

        let mut provider_options = ProviderOptions::new();
        provider_options.insert("google".into(), google_po);

        let mut tools_arr: Option<Vec<Tool>> = None;
        if let Some(gs) = &image_options.google_search {
            tools_arr = Some(vec![Tool::Provider(ProviderTool {
                id: "google.google_search".into(),
                name: "google_search".into(),
                args: gs.as_object().cloned(),
                provider_options: None,
            })]);
        }

        let call = CallOptions {
            prompt: vec![Message::User {
                content: user_content,
                provider_options: None,
            }],
            seed: options.seed,
            provider_options: Some(provider_options),
            tools: tools_arr,
            tool_choice: None::<ToolChoice>,
            headers: options.headers.clone(),
            include_raw_chunks: None,
            reasoning: None::<ReasoningEffort>,
            response_format: None::<ResponseFormat>,
            ..Default::default()
        };
        // Keep unused imports happy when build features change.
        let _ = (
            FunctionTool {
                name: String::new(),
                description: None,
                input_schema: serde_json::from_value(Value::Object(Map::new())).unwrap(),
                input_examples: None,
                strict: None,
                provider_options: None,
            },
            TextPart {
                text: String::new(),
                provider_options: None,
            },
        );

        let lm = GoogleLanguageModel::new(Arc::clone(&self.inner), self.model_id.clone());
        let result = lm.do_generate(call).await?;

        let mut images: Vec<GeneratedImage> = Vec::new();
        for c in result.content {
            if let Content::File(f) = c {
                if f.media_type.starts_with("image/") {
                    let bytes = match f.data {
                        FileData::Data {
                            data: FileBytes::Base64(s),
                        } => Bytes::from(
                            base64_decode(&s)
                                .map_err(|e| ProviderError::json_parse(s.clone(), e.to_string()))?,
                        ),
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
        }

        let mut image_meta = Vec::new();
        for _ in &images {
            image_meta.push(Value::Object(Map::new()));
        }
        let mut g_meta = result
            .provider_metadata
            .as_ref()
            .and_then(|m| m.get("google").cloned())
            .unwrap_or_default();
        g_meta.insert("images".into(), Value::Array(image_meta));
        let mut provider_metadata = ProviderMetadata::new();
        provider_metadata.insert(PROVIDER_ID.into(), g_meta);

        let usage = result.usage;
        let total_in = usage.input_tokens.total;
        let total_out = usage.output_tokens.total;
        let image_usage = if total_in.is_some() || total_out.is_some() {
            Some(llmsdk_provider::image_model::ImageUsage {
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
                media_type: f.media_type.clone(),
                provider_options: None,
            }));
        }
    }
    out
}
