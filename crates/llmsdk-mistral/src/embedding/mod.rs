//! Mistral Embeddings API implementation.
//!
//! Mirrors `@ai-sdk/mistral/src/mistral-embedding-model.ts`. One endpoint
//! (`POST /v1/embeddings`) with `encoding_format = "float"` and an optional
//! `output_dimension` / `output_dtype` provider option.
// Rust guideline compliant 2026-05-25

mod model;
mod options;
mod wire;

pub use model::MistralEmbeddingModel;
