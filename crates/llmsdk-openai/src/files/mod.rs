//! `OpenAI` Files API (`POST /v1/files`).
//!
//! Mirrors `@ai-sdk/openai/src/files/*`. Uploads a file once and returns a
//! reusable [`ProviderReference`] keyed by `"openai"` -> `<file id>`.
//!
//! [`ProviderReference`]: llmsdk_provider::shared::ProviderReference
// Rust guideline compliant 2026-02-21

mod model;
mod wire;

pub use model::OpenAiFiles;
