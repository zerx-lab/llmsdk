//! xAI Video Generation API implementation.
//!
//! Mirrors `@ai-sdk/xai/src/xai-video-model.ts` plus the supporting modules
//! (`xai-video-model-options.ts`, `xai-video-settings.ts`). First concrete
//! implementation of the workspace-wide
//! [`VideoModel`](llmsdk_provider::VideoModel) trait.
//!
//! # Endpoints
//!
//! - `POST {base_url}/videos/generations` — text → video (and R2V)
//! - `POST {base_url}/videos/edits`       — `edit-video` mode
//! - `POST {base_url}/videos/extensions`  — `extend-video` mode
//! - `GET  {base_url}/videos/{request_id}` — long-running operation poll
//!
//! All four operations are **asynchronous**: the POST returns a `request_id`,
//! and the model polls `GET /videos/{request_id}` every `pollIntervalMs`
//! milliseconds until the job reports `status: "done"` (or one of the failure
//! states `expired` / `failed`). The default poll interval is `5000` ms and
//! the default total timeout is `600000` ms, matching upstream.
// Rust guideline compliant 2026-05-25

mod build;
mod model;
mod options;
mod timestamp;
mod wire;

pub use model::XaiVideoModel;
