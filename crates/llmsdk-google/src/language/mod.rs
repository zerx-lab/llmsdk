//! Gemini Generative Language API (`generateContent` /
//! `streamGenerateContent?alt=sse`).
//!
//! Mirrors `@ai-sdk/google/src/google-language-model.ts` plus the supporting
//! modules. Implements [`llmsdk_provider::LanguageModel`].
// Rust guideline compliant 2026-05-25

mod accumulator;
mod convert_prompt;
mod finish_reason;
mod model;
mod options;
mod parse_response;
mod prepare_tools;
mod stream;
mod usage;
mod wire;

pub use model::GoogleLanguageModel;
