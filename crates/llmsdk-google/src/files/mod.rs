//! Gemini Files API (`POST /upload/v1beta/files`, resumable protocol).
//!
//! Mirrors `@ai-sdk/google/src/google-files.ts`.
// Rust guideline compliant 2026-05-25

mod model;
mod options;
mod wire;

pub use model::GoogleFiles;
