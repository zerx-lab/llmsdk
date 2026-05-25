//! Wire types for `POST /v1/responses` (request, response, SSE chunks).
//!
//! Mirrors `@ai-sdk/openai/src/responses/openai-responses-api.ts`.
// Rust guideline compliant 2026-02-21

pub mod chunk;
pub mod request;
pub mod response;
