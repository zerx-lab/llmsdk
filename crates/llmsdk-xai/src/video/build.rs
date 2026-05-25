//! Mode-aware request body construction for `XaiVideoModel`.
//!
//! Mirrors `doGenerate` lines 40-247 of `@ai-sdk/xai/src/xai-video-model.ts`,
//! split off from `model.rs` to keep the model file under the project's
//! 400-line soft cap. The two entry points are [`resolve_mode`] (which
//! decides which of the four xAI video modes the call hits) and
//! [`build_body`] (which assembles the wire request + warnings).
// Rust guideline compliant 2026-05-25

use llmsdk_provider::ProviderError;
use llmsdk_provider::shared::Warning;
use llmsdk_provider::video_model::{VideoFile, VideoOptions};

use super::options::{XaiVideoMode, XaiVideoOptions};
use super::wire::{VideoRequest, VideoSourceRef, base64_encode};

/// Map a CSS-style resolution string to xAI's preset name.
/// Mirrors `RESOLUTION_MAP` from `xai-video-model.ts`.
pub(crate) fn map_top_level_resolution(s: &str) -> Option<&'static str> {
    match s {
        "1280x720" => Some("720p"),
        "854x480" | "640x480" => Some("480p"),
        _ => None,
    }
}

/// Decide which mode the call routes to.
///
/// Mirrors `resolveVideoMode` in `xai-video-model.ts`:
///
/// 1. Explicit `mode` always wins.
/// 2. Otherwise a non-empty `videoUrl` ⇒ `edit-video` (legacy shape).
/// 3. Otherwise a non-empty `referenceImageUrls` ⇒ `reference-to-video`
///    (legacy shape).
/// 4. Otherwise text-to-video (returns `None`).
pub(crate) fn resolve_mode(xai: &XaiVideoOptions) -> Option<XaiVideoMode> {
    if let Some(m) = xai.mode {
        return Some(m);
    }
    if xai.video_url.is_some() {
        return Some(XaiVideoMode::EditVideo);
    }
    if xai
        .reference_image_urls
        .as_ref()
        .is_some_and(|v| !v.is_empty())
    {
        return Some(XaiVideoMode::ReferenceToVideo);
    }
    None
}

/// Build the wire body + the warnings list, gated on the resolved mode.
///
/// Mirrors `doGenerate` lines 90-247 of `xai-video-model.ts`.
pub(crate) fn build_body(
    model_id: &str,
    options: &VideoOptions,
    xai: &XaiVideoOptions,
    mode: Option<XaiVideoMode>,
) -> Result<(VideoRequest, Vec<Warning>), ProviderError> {
    let is_edit = mode == Some(XaiVideoMode::EditVideo);
    let is_extension = mode == Some(XaiVideoMode::ExtendVideo);
    let has_reference_images = mode == Some(XaiVideoMode::ReferenceToVideo);

    let mut warnings = collect_unsupported_warnings(options, xai, is_edit, is_extension);

    let allow_duration = !is_edit;
    let allow_aspect_ratio = !is_edit && !is_extension;
    let allow_resolution = !is_edit && !is_extension;

    let mut body = VideoRequest {
        model: model_id.to_owned(),
        prompt: options.prompt.clone().unwrap_or_default(),
        extras: xai.extras.clone(),
        ..Default::default()
    };

    if allow_duration {
        body.duration = options.duration_seconds;
    }
    if allow_aspect_ratio {
        body.aspect_ratio.clone_from(&options.aspect_ratio);
    }
    if allow_resolution {
        apply_resolution(&mut body, options, xai, &mut warnings);
    }

    // Edit / Extend share the same `video: { url }` shape.
    if is_edit || is_extension {
        let url = xai.video_url.as_deref().ok_or_else(|| {
            ProviderError::invalid_argument(
                "providerOptions.xai.videoUrl",
                "videoUrl is required for edit-video and extend-video modes",
            )
        })?;
        body.video = Some(VideoSourceRef {
            url: url.to_owned(),
        });
    }

    if let Some(image) = &options.image {
        body.image = Some(image_to_source_ref(image));
    }

    if has_reference_images {
        let urls = xai
            .reference_image_urls
            .as_ref()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| {
                ProviderError::invalid_argument(
                    "providerOptions.xai.referenceImageUrls",
                    "referenceImageUrls is required for reference-to-video mode",
                )
            })?;
        body.reference_images = Some(
            urls.iter()
                .map(|u| VideoSourceRef { url: u.clone() })
                .collect(),
        );
    }

    Ok((body, warnings))
}

/// Choose the wire `resolution` value when the call permits it.
///
/// `provider_options.xai.resolution` wins; otherwise the top-level
/// `options.resolution` runs through [`map_top_level_resolution`]. An
/// unrecognized top-level value yields a warning and drops the field.
fn apply_resolution(
    body: &mut VideoRequest,
    options: &VideoOptions,
    xai: &XaiVideoOptions,
    warnings: &mut Vec<Warning>,
) {
    if let Some(r) = &xai.resolution {
        body.resolution = Some(r.clone());
    } else if let Some(raw) = options.resolution.as_deref() {
        if let Some(mapped) = map_top_level_resolution(raw) {
            body.resolution = Some(mapped.to_owned());
        } else {
            warnings.push(Warning::UnsupportedSetting {
                setting: "resolution".into(),
                details: Some(format!(
                    "Unrecognized resolution \"{raw}\". \
                     Use providerOptions.xai.resolution with \"480p\" or \"720p\" instead."
                )),
            });
        }
    }
}

/// Collect every "the call carried a setting xAI does not support" warning.
///
/// Each branch mirrors one warning from `xai-video-model.ts` lines 92-164.
fn collect_unsupported_warnings(
    options: &VideoOptions,
    xai: &XaiVideoOptions,
    is_edit: bool,
    is_extension: bool,
) -> Vec<Warning> {
    let mut warnings = Vec::new();
    if options.fps.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "fps".into(),
            details: Some("xAI video models do not support custom FPS.".into()),
        });
    }
    if options.seed.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "seed".into(),
            details: Some("xAI video models do not support seed.".into()),
        });
    }
    if options.n > 1 {
        warnings.push(Warning::UnsupportedSetting {
            setting: "n".into(),
            details: Some(
                "xAI video models do not support generating multiple videos per call. \
                 Only 1 video will be generated."
                    .into(),
            ),
        });
    }
    if is_edit && options.duration_seconds.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "duration".into(),
            details: Some("xAI video editing does not support custom duration.".into()),
        });
    }
    if is_edit && options.aspect_ratio.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "aspectRatio".into(),
            details: Some("xAI video editing does not support custom aspect ratio.".into()),
        });
    }
    if is_edit && (xai.resolution.is_some() || options.resolution.is_some()) {
        warnings.push(Warning::UnsupportedSetting {
            setting: "resolution".into(),
            details: Some("xAI video editing does not support custom resolution.".into()),
        });
    }
    if is_extension && options.aspect_ratio.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "aspectRatio".into(),
            details: Some("xAI video extension does not support custom aspect ratio.".into()),
        });
    }
    if is_extension && (xai.resolution.is_some() || options.resolution.is_some()) {
        warnings.push(Warning::UnsupportedSetting {
            setting: "resolution".into(),
            details: Some("xAI video extension does not support custom resolution.".into()),
        });
    }
    warnings
}

/// Convert a [`VideoFile`] input to the nested xAI `image: { url }` wire shape.
///
/// URL inputs pass through unchanged; inline bytes / base64 become a
/// `data:<media>;base64,<payload>` URI to match upstream.
fn image_to_source_ref(file: &VideoFile) -> VideoSourceRef {
    match file {
        VideoFile::Url { url, .. } => VideoSourceRef { url: url.clone() },
        VideoFile::File {
            media_type, data, ..
        } => {
            let payload = match data {
                llmsdk_provider::shared::FileBytes::Base64(s) => s.clone(),
                llmsdk_provider::shared::FileBytes::Bytes(b) => base64_encode(b),
            };
            VideoSourceRef {
                url: format!("data:{media_type};base64,{payload}"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PROVIDER_ID;
    use llmsdk_provider::shared::{FileBytes, ProviderOptions};
    use serde_json::{Value as JsonValue, json};

    fn po(value: &JsonValue) -> ProviderOptions {
        let mut po = ProviderOptions::new();
        po.insert(PROVIDER_ID.into(), value.as_object().cloned().unwrap());
        po
    }

    fn parse(value: &JsonValue) -> XaiVideoOptions {
        let map = po(value);
        super::super::options::parse(Some(&map))
    }

    #[test]
    fn resolve_mode_explicit_wins_over_legacy_shape() {
        let opts = parse(&json!({
            "mode": "extend-video",
            "videoUrl": "https://x.ai/in.mp4",
            "referenceImageUrls": ["https://x.ai/a.png"]
        }));
        assert_eq!(resolve_mode(&opts), Some(XaiVideoMode::ExtendVideo));
    }

    #[test]
    fn resolve_mode_legacy_video_url_detects_edit() {
        let opts = parse(&json!({"videoUrl": "https://x.ai/a.mp4"}));
        assert_eq!(resolve_mode(&opts), Some(XaiVideoMode::EditVideo));
    }

    #[test]
    fn resolve_mode_legacy_reference_images_detects_r2v() {
        let opts = parse(&json!({"referenceImageUrls": ["https://x.ai/a.png"]}));
        assert_eq!(resolve_mode(&opts), Some(XaiVideoMode::ReferenceToVideo));
    }

    #[test]
    fn resolve_mode_text_to_video_when_empty() {
        let opts = super::super::options::parse(None);
        assert_eq!(resolve_mode(&opts), None);
    }

    #[test]
    fn warnings_emitted_for_fps_seed_and_n_gt_one() {
        let opts = VideoOptions {
            prompt: Some("a".into()),
            n: 2,
            fps: Some(30),
            seed: Some(1),
            ..Default::default()
        };
        let xai = super::super::options::parse(None);
        let (_, warnings) = build_body("grok-imagine-video", &opts, &xai, None).unwrap();
        let names: Vec<&str> = warnings
            .iter()
            .filter_map(|w| match w {
                Warning::UnsupportedSetting { setting, .. } => Some(setting.as_str()),
                _ => None,
            })
            .collect();
        assert!(names.contains(&"fps"));
        assert!(names.contains(&"seed"));
        assert!(names.contains(&"n"));
    }

    #[test]
    fn edit_mode_warns_on_duration_aspect_resolution_and_drops_them() {
        let opts = VideoOptions {
            prompt: Some("edit".into()),
            n: 1,
            duration_seconds: Some(5.0),
            aspect_ratio: Some("16:9".into()),
            resolution: Some("1280x720".into()),
            ..Default::default()
        };
        let xai = parse(&json!({"mode": "edit-video", "videoUrl": "https://x.ai/in.mp4"}));
        let (body, warnings) = build_body(
            "grok-imagine-video",
            &opts,
            &xai,
            Some(XaiVideoMode::EditVideo),
        )
        .unwrap();
        assert!(body.duration.is_none());
        assert!(body.aspect_ratio.is_none());
        assert!(body.resolution.is_none());
        let kinds: Vec<&str> = warnings
            .iter()
            .filter_map(|w| match w {
                Warning::UnsupportedSetting { setting, .. } => Some(setting.as_str()),
                _ => None,
            })
            .collect();
        assert!(kinds.contains(&"duration"));
        assert!(kinds.contains(&"aspectRatio"));
        assert!(kinds.contains(&"resolution"));
        let video = body.video.expect("edit-video sets video.url");
        assert_eq!(video.url, "https://x.ai/in.mp4");
    }

    #[test]
    fn text_to_video_top_level_resolution_mapping() {
        let opts = VideoOptions {
            prompt: Some("a".into()),
            n: 1,
            resolution: Some("854x480".into()),
            ..Default::default()
        };
        let xai = super::super::options::parse(None);
        let (body, _) = build_body("grok-imagine-video", &opts, &xai, None).unwrap();
        assert_eq!(body.resolution.as_deref(), Some("480p"));
    }

    #[test]
    fn text_to_video_unrecognized_resolution_warns_and_drops() {
        let opts = VideoOptions {
            prompt: Some("a".into()),
            n: 1,
            resolution: Some("4k".into()),
            ..Default::default()
        };
        let xai = super::super::options::parse(None);
        let (body, warnings) = build_body("grok-imagine-video", &opts, &xai, None).unwrap();
        assert!(body.resolution.is_none());
        assert!(warnings.iter().any(|w| matches!(
            w,
            Warning::UnsupportedSetting { setting, .. } if setting == "resolution"
        )));
    }

    #[test]
    fn provider_options_resolution_wins_over_top_level() {
        let opts = VideoOptions {
            prompt: Some("a".into()),
            n: 1,
            resolution: Some("1280x720".into()),
            ..Default::default()
        };
        let xai = parse(&json!({"resolution": "480p"}));
        let (body, _) = build_body("grok-imagine-video", &opts, &xai, None).unwrap();
        assert_eq!(body.resolution.as_deref(), Some("480p"));
    }

    #[test]
    fn r2v_mode_emits_reference_images_array() {
        let opts = VideoOptions {
            prompt: Some("r2v".into()),
            n: 1,
            ..Default::default()
        };
        let xai = parse(&json!({
            "mode": "reference-to-video",
            "referenceImageUrls": ["https://x.ai/a.png", "https://x.ai/b.png"]
        }));
        let (body, _) = build_body(
            "grok-imagine-video",
            &opts,
            &xai,
            Some(XaiVideoMode::ReferenceToVideo),
        )
        .unwrap();
        let arr = body.reference_images.expect("reference_images set");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0].url, "https://x.ai/a.png");
        assert!(body.video.is_none());
    }

    #[test]
    fn image_input_url_passes_through_to_body_image() {
        let opts = VideoOptions {
            prompt: Some("i2v".into()),
            n: 1,
            image: Some(VideoFile::Url {
                url: "https://x.ai/in.png".into(),
                provider_options: None,
            }),
            ..Default::default()
        };
        let xai = super::super::options::parse(None);
        let (body, _) = build_body("grok-imagine-video", &opts, &xai, None).unwrap();
        let img = body.image.expect("image set");
        assert_eq!(img.url, "https://x.ai/in.png");
    }

    #[test]
    fn image_input_bytes_become_data_uri() {
        let opts = VideoOptions {
            prompt: Some("i2v".into()),
            n: 1,
            image: Some(VideoFile::File {
                media_type: "image/png".into(),
                data: FileBytes::Bytes(b"foo".to_vec()),
                provider_options: None,
            }),
            ..Default::default()
        };
        let xai = super::super::options::parse(None);
        let (body, _) = build_body("grok-imagine-video", &opts, &xai, None).unwrap();
        let img = body.image.expect("image set");
        assert_eq!(img.url, "data:image/png;base64,Zm9v");
    }

    #[test]
    fn unknown_provider_options_flatten_onto_request_root() {
        let opts = VideoOptions {
            prompt: Some("a".into()),
            n: 1,
            ..Default::default()
        };
        let xai = parse(&json!({"watermark": "off"}));
        let (body, _) = build_body("grok-imagine-video", &opts, &xai, None).unwrap();
        let value = serde_json::to_value(&body).unwrap();
        assert_eq!(value["watermark"], "off");
    }
}
