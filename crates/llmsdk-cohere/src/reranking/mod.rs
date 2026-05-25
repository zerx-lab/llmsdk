//! Cohere v2 Reranking API implementation.
//!
//! Mirrors `@ai-sdk/cohere/src/reranking/cohere-reranking-model.ts` and
//! supporting files. First [`llmsdk_provider::RerankingModel`] implementation
//! in the workspace.
//!
//! # Endpoint
//!
//! `POST {base_url}/rerank` — Cohere v2 rerank.
// Rust guideline compliant 2026-05-25

mod model;
mod options;
mod wire;

pub use model::CohereRerankingModel;
