//! Cohere Chat API implementation.
//!
//! Mirrors `@ai-sdk/cohere/src/cohere-chat-language-model.ts` and supporting
//! modules (`convert-to-cohere-chat-prompt.ts`, `cohere-prepare-tools.ts`,
//! `convert-cohere-usage.ts`, `map-cohere-finish-reason.ts`,
//! `cohere-chat-language-model-options.ts`, `cohere-error.ts`).
//!
//! # Endpoint
//!
//! `POST {base_url}/chat` — Cohere v2 chat (not OpenAI-compatible).
// Rust guideline compliant 2026-05-25

mod convert_prompt;
mod finish_reason;
mod model;
mod options;
mod parse_response;
mod prepare_tools;
mod stream;
mod usage;
mod wire;

pub use model::CohereChatModel;
