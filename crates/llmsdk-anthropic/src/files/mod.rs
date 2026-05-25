//! `Anthropic` Files API (`POST /v1/files`).
//!
//! Mirrors `@ai-sdk/anthropic/src/anthropic-files.ts`. Used to upload a file
//! once and reference its server-side id from `Messages` requests (as
//! `FileData::Reference { reference: { "anthropic": "<id>" } }`).
// Rust guideline compliant 2026-02-21

mod model;
mod wire;

pub use model::AnthropicFiles;
pub(crate) use model::upload_data_to_bytes;
