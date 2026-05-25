//! xAI Image Generation API implementation.
//!
//! Mirrors `@ai-sdk/xai/src/xai-image-model.ts` plus the supporting modules
//! (`xai-image-model-options.ts`, `xai-image-settings.ts`).
//!
//! # Endpoints
//!
//! - `POST {base_url}/images/generations` — text → image
//! - `POST {base_url}/images/edits` — image + prompt → image
//!   (selected automatically when [`ImageOptions::files`] is non-empty)
//!
//! Both wire shapes are xAI-specific (not OpenAI-compatible). Responses
//! prefer `b64_json` payloads inline; a `url` fallback is downloaded over
//! HTTP to honor the v4 `GeneratedImage::bytes` contract.
//!
//! [`ImageOptions::files`]: llmsdk_provider::image_model::ImageOptions::files
// Rust guideline compliant 2026-05-25

mod model;
mod options;
mod wire;

pub use model::XaiImageModel;
