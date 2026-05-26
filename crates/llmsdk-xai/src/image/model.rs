//! [`ImageModel`] implementation for xAI image generation.
//!
//! Mirrors `XaiImageModel` from `@ai-sdk/xai/src/xai-image-model.ts`. Entry:
//! [`XaiImageModel::new`] via [`crate::Xai::image`].
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::image_model::{GeneratedImage, ImageModel, ImageOptions, ImageResult};
use llmsdk_provider::language_model::FilePart;
use llmsdk_provider::shared::{
    FileBytes, FileData, Headers, ProviderMetadata, RequestInfo, ResponseInfo, Warning,
};
use llmsdk_provider_utils::http::{JsonRequest, post_json};
use serde_json::{Map, Value as JsonValue};

use crate::PROVIDER_ID;
use crate::config::Inner;

use super::options::parse as parse_xai_options;
use super::wire::{
    Base64Error, ImageData, ImageReference, ImageRequest, ImageResponse, ResponseFormat,
    base64_decode, base64_encode,
};

/// xAI image-generation model handle.
///
/// Cheap to clone — shares the provider's HTTP client and auth state via
/// [`Xai`](crate::Xai)'s `Arc`. Supports both `grok-imagine-image` and
/// `grok-imagine-image-pro` (and any future `grok-*-image*` id passed
/// through verbatim).
#[derive(Debug, Clone)]
pub struct XaiImageModel {
    inner: Arc<Inner>,
    model_id: String,
}

/// Maximum images that a single xAI image call may request.
///
/// Hard-coded to mirror upstream `XaiImageModel.maxImagesPerCall = 3`.
const MAX_IMAGES_PER_CALL: u32 = 3;

impl XaiImageModel {
    /// Construct from shared provider state and a model id.
    pub(crate) fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn generations_endpoint(&self) -> String {
        format!("{}/images/generations", self.inner.base_url)
    }

    fn edits_endpoint(&self) -> String {
        format!("{}/images/edits", self.inner.base_url)
    }
}

#[async_trait]
impl ImageModel for XaiImageModel {
    fn provider(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_images_per_call(&self) -> Option<u32> {
        Some(MAX_IMAGES_PER_CALL)
    }

    async fn do_generate(&self, options: ImageOptions) -> Result<ImageResult, ProviderError> {
        let (request, endpoint, warnings) = build_request(&self.model_id, &options)?;
        let request_body_value = serde_json::to_value(&request).ok();

        let url = match endpoint {
            EndpointMode::Generate => self.generations_endpoint(),
            EndpointMode::Edit => self.edits_endpoint(),
        };

        let mut request_headers = self.inner.headers.clone();
        if let Some(headers) = options.headers {
            for (name, value) in headers {
                request_headers.insert(name, value);
            }
        }

        let mut http_request = JsonRequest::new(url.clone(), request);
        http_request.headers = request_headers;

        let response = post_json::<_, ImageResponse>(&self.inner.http, http_request).await?;
        let resp = response.value;
        let response_headers = response.headers;

        let images = collect_images(&self.inner.http, &resp.data).await?;
        let provider_metadata = build_provider_metadata(&resp);

        Ok(ImageResult {
            images,
            warnings,
            usage: None,
            provider_metadata: Some(provider_metadata),
            request: Some(RequestInfo {
                body: request_body_value,
            }),
            response: Some(ResponseInfo {
                id: None,
                timestamp: None,
                model_id: Some(self.model_id.clone()),
                headers: Some(headers_to_provider(response_headers)),
                ..ResponseInfo::default()
            }),
        })
    }
}

/// Which endpoint to hit for a given [`ImageOptions`].
///
/// Empty / absent `files` → text → image. One or more `files` → image edit.
/// Variations have no dedicated endpoint on xAI (upstream parity).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EndpointMode {
    Generate,
    Edit,
}

/// Decide which endpoint the call routes to.
fn route_endpoint(options: &ImageOptions) -> EndpointMode {
    let has_files = options.files.as_ref().is_some_and(|f| !f.is_empty());
    if has_files {
        EndpointMode::Edit
    } else {
        EndpointMode::Generate
    }
}

/// Build the wire request, the endpoint to use, and warnings for ignored
/// settings.
fn build_request(
    model_id: &str,
    options: &ImageOptions,
) -> Result<(ImageRequest, EndpointMode, Vec<Warning>), ProviderError> {
    let mut warnings = Vec::new();

    if options.size.is_some() {
        warnings.push(Warning::Unsupported {
            feature: "size".to_owned(),
            details: Some(
                "This model does not support the `size` option. Use `aspectRatio` instead."
                    .to_owned(),
            ),
        });
    }
    if options.seed.is_some() {
        warnings.push(Warning::Unsupported {
            feature: "seed".to_owned(),
            details: None,
        });
    }
    if options.mask.is_some() {
        warnings.push(Warning::Unsupported {
            feature: "mask".to_owned(),
            details: None,
        });
    }

    let xai = parse_xai_options(options.provider_options.as_ref());
    let mode = route_endpoint(options);

    // Convert source files to data URIs (or pass URL refs through).
    let image_urls = match options.files.as_deref() {
        Some(files) if !files.is_empty() => files
            .iter()
            .map(file_to_url_or_data_uri)
            .collect::<Result<Vec<_>, _>>()?,
        _ => Vec::new(),
    };

    // Per upstream: top-level `aspectRatio` wins; `provider_options.aspect_ratio`
    // only fills in when the top-level slot is absent.
    let aspect_ratio = options.aspect_ratio.clone().or(xai.aspect_ratio.clone());

    let (image, images) = build_image_references(&image_urls);

    // Mirror upstream zod enums in xai-image-model-options.ts:6-7:
    // `quality: z.enum(['low','medium','high'])` and
    // `resolution: z.enum(['1k','2k'])`. Reject invalid values with an
    // `Unsupported` warning rather than forwarding them to the server
    // (which would 4xx with an opaque error). Mirrors zod's strict-enum
    // behavior; we degrade gracefully (warn + drop) instead of panicking
    // because llmsdk's parse path is forgiving by design.
    let quality = match xai.quality.as_deref() {
        None => None,
        Some(v @ ("low" | "medium" | "high")) => Some(v.to_owned()),
        Some(other) => {
            warnings.push(Warning::Unsupported {
                feature: "xai.quality".into(),
                details: Some(format!(
                    "xai quality \"{other}\" is not a recognized preset (\"low\" / \"medium\" / \"high\"); ignored."
                )),
            });
            None
        }
    };
    let resolution = match xai.resolution.as_deref() {
        None => None,
        Some(v @ ("1k" | "2k")) => Some(v.to_owned()),
        Some(other) => {
            warnings.push(Warning::Unsupported {
                feature: "xai.resolution".into(),
                details: Some(format!(
                    "xai resolution \"{other}\" is not a recognized preset (\"1k\" / \"2k\"); ignored."
                )),
            });
            None
        }
    };

    let request = ImageRequest {
        model: model_id.to_owned(),
        prompt: options.prompt.clone(),
        n: options.n,
        response_format: ResponseFormat::B64Json,
        aspect_ratio,
        output_format: xai.output_format.clone(),
        sync_mode: xai.sync_mode,
        resolution,
        quality,
        user: xai.user.clone(),
        image,
        images,
    };

    Ok((request, mode, warnings))
}

/// Split the prepared URL list into `image` (single) vs `images` (many),
/// matching upstream's wire shape:
///
/// - 0 urls → both `None`
/// - 1 url → `image = { url, type: 'image_url' }`
/// - ≥2 urls → `images = [{ url, type: 'image_url' }, ...]`
fn build_image_references(
    urls: &[String],
) -> (Option<ImageReference>, Option<Vec<ImageReference>>) {
    match urls.len() {
        0 => (None, None),
        1 => (Some(ImageReference::image_url(urls[0].clone())), None),
        _ => (
            None,
            Some(
                urls.iter()
                    .map(|u| ImageReference::image_url(u.clone()))
                    .collect(),
            ),
        ),
    }
}

/// Mirror upstream `convertImageModelFileToDataUri`: URL stays URL,
/// inline bytes / base64 become `data:<media>;base64,<payload>`.
/// `Text` / `Reference` variants are not supported as image sources.
fn file_to_url_or_data_uri(fp: &FilePart) -> Result<String, ProviderError> {
    match &fp.data {
        FileData::Url { url } => Ok(url.clone()),
        FileData::Data { data } => {
            let payload = match data {
                FileBytes::Base64(s) => s.clone(),
                FileBytes::Bytes(b) => base64_encode(b),
            };
            Ok(format!("data:{};base64,{}", fp.media_type, payload))
        }
        FileData::Text { .. } | FileData::Reference { .. } => Err(ProviderError::unsupported(
            "xAI image edits require inline file bytes or a URL; \
             text / reference variants are not supported as sources",
        )),
    }
}

/// Materialize one [`GeneratedImage`] per upstream entry.
///
/// xAI returns either `b64_json` inline or a `url` to download. Per upstream
/// behavior, when **all** entries carry `b64_json` we skip URL downloads
/// entirely; otherwise we issue one GET per entry.
async fn collect_images(
    http: &llmsdk_provider_utils::http::HttpClient,
    data: &[ImageData],
) -> Result<Vec<GeneratedImage>, ProviderError> {
    let all_inline = !data.is_empty() && data.iter().all(|d| d.b64_json.is_some());
    let mut out = Vec::with_capacity(data.len());

    if all_inline {
        for (idx, item) in data.iter().enumerate() {
            let b64 = item
                .b64_json
                .as_deref()
                .ok_or_else(|| missing_payload_error(idx))?;
            let bytes = decode_b64_field(idx, b64)?;
            let media_type = sniff_media_type(&bytes);
            out.push(GeneratedImage {
                bytes: bytes.into(),
                media_type,
            });
        }
        return Ok(out);
    }

    for (idx, item) in data.iter().enumerate() {
        let bytes = if let Some(b64) = &item.b64_json {
            decode_b64_field(idx, b64)?
        } else if let Some(url) = &item.url {
            download_image(http, url).await?
        } else {
            return Err(missing_payload_error(idx));
        };
        let media_type = sniff_media_type(&bytes);
        out.push(GeneratedImage {
            bytes: bytes.into(),
            media_type,
        });
    }
    Ok(out)
}

fn missing_payload_error(idx: usize) -> ProviderError {
    ProviderError::type_validation(
        format!("data[{idx}]"),
        JsonValue::Null,
        "xAI returned an image entry with neither `b64_json` nor `url`",
    )
}

fn decode_b64_field(idx: usize, b64: &str) -> Result<Vec<u8>, ProviderError> {
    base64_decode(b64).map_err(|err: Base64Error| {
        ProviderError::type_validation(
            format!("data[{idx}].b64_json"),
            JsonValue::String(b64.to_owned()),
            format!("xAI returned invalid base64 in image data: {err}"),
        )
    })
}

/// GET an image URL and return its raw bytes.
async fn download_image(
    http: &llmsdk_provider_utils::http::HttpClient,
    url: &str,
) -> Result<Vec<u8>, ProviderError> {
    let response = http
        .reqwest()
        .get(url)
        .send()
        .await
        .map_err(|e| ProviderError::api_call(url, format!("image download failed: {e}")))?;
    let status = response.status();
    if !status.is_success() {
        return Err(ProviderError::api_call(
            url,
            format!("image download returned HTTP {status}"),
        ));
    }
    let bytes = response
        .bytes()
        .await
        .map_err(|e| ProviderError::api_call(url, format!("image read failed: {e}")))?;
    Ok(bytes.to_vec())
}

/// Build the `provider_metadata.xai` payload (`images[]` + optional
/// `costInUsdTicks`).
fn build_provider_metadata(resp: &ImageResponse) -> ProviderMetadata {
    let images: Vec<JsonValue> = resp
        .data
        .iter()
        .map(|item| {
            let mut obj = Map::new();
            if let Some(rp) = &item.revised_prompt {
                obj.insert("revisedPrompt".to_owned(), JsonValue::String(rp.clone()));
            }
            JsonValue::Object(obj)
        })
        .collect();

    let mut xai = Map::new();
    xai.insert("images".to_owned(), JsonValue::Array(images));
    if let Some(usage) = &resp.usage
        && let Some(ticks) = usage.cost_in_usd_ticks
    {
        xai.insert("costInUsdTicks".to_owned(), JsonValue::from(ticks));
    }

    let mut pm = ProviderMetadata::new();
    pm.insert(PROVIDER_ID.to_owned(), xai);
    pm
}

/// Best-effort media-type detection by sniffing magic bytes. Defaults to
/// `image/jpeg` (xAI default output).
fn sniff_media_type(bytes: &[u8]) -> String {
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
    "image/jpeg".to_owned()
}

fn headers_to_provider(raw: HashMap<String, String>) -> Headers {
    raw.into_iter().map(|(k, v)| (k, Some(v))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn opts_with_xai(value: &JsonValue) -> ImageOptions {
        let mut po = llmsdk_provider::shared::ProviderOptions::new();
        po.insert(PROVIDER_ID.into(), value.as_object().cloned().unwrap());
        ImageOptions {
            prompt: "a hat".into(),
            provider_options: Some(po),
            ..Default::default()
        }
    }

    #[test]
    fn route_endpoint_picks_generate_without_files() {
        let opts = ImageOptions {
            prompt: "cat".into(),
            ..Default::default()
        };
        assert_eq!(route_endpoint(&opts), EndpointMode::Generate);
    }

    #[test]
    fn route_endpoint_picks_edit_with_files() {
        let opts = ImageOptions {
            prompt: "make it red".into(),
            files: Some(vec![FilePart {
                filename: None,
                data: FileData::Url {
                    url: "https://x.ai/img.png".into(),
                },
                media_type: "image/png".into(),
                provider_options: None,
            }]),
            ..Default::default()
        };
        assert_eq!(route_endpoint(&opts), EndpointMode::Edit);
    }

    #[test]
    fn size_seed_mask_each_emit_a_warning() {
        let opts = ImageOptions {
            prompt: "hi".into(),
            size: Some("1024x1024".into()),
            seed: Some(7),
            mask: Some(FilePart {
                filename: None,
                data: FileData::Data {
                    data: FileBytes::Bytes(vec![0xFF]),
                },
                media_type: "image/png".into(),
                provider_options: None,
            }),
            ..Default::default()
        };
        let (_, _, warnings) = build_request("grok-imagine-image", &opts).unwrap();
        let names: Vec<&str> = warnings
            .iter()
            .filter_map(|w| match w {
                Warning::Unsupported { feature, .. } => Some(feature.as_str()),
                _ => None,
            })
            .collect();
        assert!(names.contains(&"size"));
        assert!(names.contains(&"seed"));
        assert!(names.contains(&"mask"));
    }

    #[test]
    fn provider_options_pass_through_to_wire() {
        let opts = opts_with_xai(&json!({
            "aspect_ratio": "16:9",
            "output_format": "png",
            "sync_mode": true,
            "resolution": "2k",
            "quality": "high",
            "user": "alice@example.com"
        }));
        let (req, mode, _) = build_request("grok-imagine-image", &opts).unwrap();
        assert_eq!(mode, EndpointMode::Generate);
        assert_eq!(req.aspect_ratio.as_deref(), Some("16:9"));
        assert_eq!(req.output_format.as_deref(), Some("png"));
        assert_eq!(req.sync_mode, Some(true));
        assert_eq!(req.resolution.as_deref(), Some("2k"));
        assert_eq!(req.quality.as_deref(), Some("high"));
        assert_eq!(req.user.as_deref(), Some("alice@example.com"));
    }

    #[test]
    fn top_level_aspect_ratio_wins_over_provider_option() {
        let mut opts = opts_with_xai(&json!({"aspect_ratio": "16:9"}));
        opts.aspect_ratio = Some("1:1".into());
        let (req, _, _) = build_request("grok-imagine-image", &opts).unwrap();
        assert_eq!(req.aspect_ratio.as_deref(), Some("1:1"));
    }

    #[test]
    fn single_file_lands_on_image_field() {
        let opts = ImageOptions {
            prompt: "make it red".into(),
            files: Some(vec![FilePart {
                filename: None,
                data: FileData::Data {
                    data: FileBytes::Bytes(b"foo".to_vec()),
                },
                media_type: "image/png".into(),
                provider_options: None,
            }]),
            ..Default::default()
        };
        let (req, mode, _) = build_request("grok-imagine-image", &opts).unwrap();
        assert_eq!(mode, EndpointMode::Edit);
        let image = req.image.expect("single image set");
        assert_eq!(image.ref_type, "image_url");
        assert_eq!(image.url, "data:image/png;base64,Zm9v");
        assert!(req.images.is_none());
    }

    #[test]
    fn multiple_files_land_on_images_array() {
        let mk = |bytes: Vec<u8>| FilePart {
            filename: None,
            data: FileData::Data {
                data: FileBytes::Bytes(bytes),
            },
            media_type: "image/png".into(),
            provider_options: None,
        };
        let opts = ImageOptions {
            prompt: "swap them".into(),
            files: Some(vec![
                mk(b"foo".to_vec()),
                FilePart {
                    filename: None,
                    data: FileData::Url {
                        url: "https://x.ai/img.png".into(),
                    },
                    media_type: "image/png".into(),
                    provider_options: None,
                },
            ]),
            ..Default::default()
        };
        let (req, _, _) = build_request("grok-imagine-image", &opts).unwrap();
        assert!(req.image.is_none());
        let arr = req.images.expect("images array set");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0].url, "data:image/png;base64,Zm9v");
        // URL variant passes through unchanged.
        assert_eq!(arr[1].url, "https://x.ai/img.png");
    }

    #[test]
    fn text_file_data_is_rejected_with_clear_error() {
        let opts = ImageOptions {
            prompt: "edit".into(),
            files: Some(vec![FilePart {
                filename: None,
                data: FileData::Text {
                    text: "hello".into(),
                },
                media_type: "image/png".into(),
                provider_options: None,
            }]),
            ..Default::default()
        };
        let err = build_request("grok-imagine-image", &opts).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.to_lowercase().contains("text"), "got: {msg}");
    }

    #[test]
    fn sniff_media_type_detects_png_jpeg_webp_gif_and_defaults_jpeg() {
        assert_eq!(sniff_media_type(b"\x89PNG\r\n\x1a\nrest"), "image/png");
        assert_eq!(sniff_media_type(&[0xFF, 0xD8, 0xFF, 0xE0]), "image/jpeg");
        assert_eq!(sniff_media_type(b"GIF87aXYZ"), "image/gif");
        let mut webp = b"RIFF\0\0\0\0WEBPVP8 ".to_vec();
        webp.extend_from_slice(&[0; 4]);
        assert_eq!(sniff_media_type(&webp), "image/webp");
        assert_eq!(sniff_media_type(b"unknown blob"), "image/jpeg");
    }

    #[test]
    fn build_provider_metadata_includes_revised_prompt_and_cost() {
        let resp = ImageResponse {
            data: vec![
                ImageData {
                    url: None,
                    b64_json: Some("Zg==".into()),
                    revised_prompt: Some("revised".into()),
                },
                ImageData {
                    url: Some("https://x.ai/img.png".into()),
                    b64_json: None,
                    revised_prompt: None,
                },
            ],
            usage: Some(super::super::wire::ImageUsage {
                cost_in_usd_ticks: Some(99),
            }),
        };
        let pm = build_provider_metadata(&resp);
        let xai = pm.get(PROVIDER_ID).expect("xai slot");
        let images = xai.get("images").and_then(|v| v.as_array()).unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!(images[0]["revisedPrompt"], "revised");
        // Second entry had no revisedPrompt — slot is an empty object,
        // matching upstream's spread `...(item.revised_prompt ? ... : {})`.
        assert!(images[1].as_object().is_some_and(serde_json::Map::is_empty));
        assert_eq!(xai["costInUsdTicks"], 99);
    }
}
