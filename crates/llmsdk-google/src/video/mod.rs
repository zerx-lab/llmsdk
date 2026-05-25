//! Veo video generation (`:predictLongRunning` + LRO polling).
//!
//! Mirrors `@ai-sdk/google/src/google-video-model.ts`.
// Rust guideline compliant 2026-05-25

mod model;
mod options;
mod wire;

pub use model::GoogleVideoModel;
