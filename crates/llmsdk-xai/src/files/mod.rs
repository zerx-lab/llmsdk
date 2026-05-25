//! xAI Files API (`POST /v1/files`).
//!
//! Mirrors `@ai-sdk/xai/src/files/*`. Used to upload a file once and
//! reference its server-side id from subsequent requests via the
//! `provider_reference.xai` slot returned in
//! [`UploadFileResult`](llmsdk_provider::UploadFileResult).
// Rust guideline compliant 2026-05-25

mod model;
mod options;
mod wire;

pub use model::XaiFiles;
