//! Cohere Embeddings API implementation.
//!
//! Mirrors `@ai-sdk/cohere/src/cohere-embedding-model.ts` and
//! `cohere-embedding-model-options.ts`.
//!
//! # Endpoint
//!
//! `POST {base_url}/embed` — Cohere v2 embeddings.
// Rust guideline compliant 2026-05-25

mod model;
mod options;
mod wire;

pub use model::CohereEmbeddingModel;
