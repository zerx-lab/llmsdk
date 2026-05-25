//! Mistral Chat Completions API implementation.
//!
//! Mirrors `@ai-sdk/mistral/src/mistral-chat-language-model.ts` plus the
//! supporting modules (`convert-to-mistral-chat-messages.ts`,
//! `mistral-prepare-tools.ts`, `convert-mistral-usage.ts`,
//! `map-mistral-finish-reason.ts`, `mistral-chat-language-model-options.ts`,
//! `mistral-error.ts`).
//!
//! # Endpoint
//!
//! `POST {base_url}/chat/completions` — Mistral wire shape with Mistral-specific
//! extensions (`prefix` continuation mode, `safe_prompt`, `document_url`).
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

pub use model::MistralChatModel;
