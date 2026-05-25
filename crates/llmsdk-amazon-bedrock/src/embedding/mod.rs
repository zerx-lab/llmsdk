//! Bedrock embedding models (Titan / Cohere / Nova families).
//!
//! Mirrors `@ai-sdk/amazon-bedrock/src/amazon-bedrock-embedding-model.ts`.
//! The endpoint (`POST /model/{id}/invoke`) is the same for every family;
//! only the request body and the response shape differ. We dispatch on the
//! model id prefix:
//!
//! - `amazon.titan-embed-*` → Titan body / response
//! - `amazon.nova-*` containing `"embed"` → Nova body / response
//! - `cohere.embed-*` → Cohere body / response (v3 array + v4 object)
// Rust guideline compliant 2026-05-25

mod model;
mod options;
mod wire;

pub use model::AmazonBedrockEmbeddingModel;
