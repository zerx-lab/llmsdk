//! Bedrock Reranking API implementation (`POST /rerank` on
//! `bedrock-agent-runtime`).
//!
//! Mirrors `@ai-sdk/amazon-bedrock/src/reranking/amazon-bedrock-reranking-model.ts`.
//! Supported models on Bedrock today:
//!
//! - `cohere.rerank-v3-5:0`
//! - `amazon.rerank-v1:0`
//!
//! Documents are wrapped into the `inlineDocumentSource` payload using
//! either `textDocument` (for `RerankingDocuments::Text`) or `jsonDocument`
//! (for `RerankingDocuments::Object`).
// Rust guideline compliant 2026-05-25

mod model;
mod options;
mod wire;

pub use model::AmazonBedrockRerankingModel;
