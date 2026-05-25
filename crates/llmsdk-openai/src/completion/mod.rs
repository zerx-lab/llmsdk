//! `OpenAI` Completions (legacy `POST /v1/completions`) implementation.
//!
//! Mirrors `@ai-sdk/openai/src/completion/*`. Supports `gpt-3.5-turbo-instruct`
//! and other models that still accept the legacy `prompt` (string) endpoint.
//! No tool / `response_format` / multimodal content — those produce warnings.
// Rust guideline compliant 2026-02-21

mod model;

pub use model::OpenAiCompletionLanguageModel;
