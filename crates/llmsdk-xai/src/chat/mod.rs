//! xAI Chat Completions API implementation.
//!
//! Mirrors `@ai-sdk/xai/src/xai-chat-language-model.ts` plus the supporting
//! modules (`convert-to-xai-chat-messages.ts`, `xai-prepare-tools.ts`,
//! `convert-xai-chat-usage.ts`, `map-xai-finish-reason.ts`,
//! `xai-chat-language-model-options.ts`, `xai-error.ts`).
//!
//! # Endpoint
//!
//! `POST {base_url}/chat/completions` — `OpenAI`-compatible wire shape with
//! xAI-specific extensions (`reasoning_content`, `citations`, `search_parameters`).
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

pub use model::XaiChatModel;
