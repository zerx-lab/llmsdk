//! Gemini text embeddings (`:embedContent` and `:batchEmbedContents`).
//!
//! Mirrors `@ai-sdk/google/src/google-embedding-model.ts`.
// Rust guideline compliant 2026-05-25

mod model;
mod options;
mod wire;

pub use model::GoogleEmbeddingModel;
