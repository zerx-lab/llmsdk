//! Google Interactions API (`POST /v1beta/interactions`).
//!
//! Mirrors `@ai-sdk/google/src/interactions/*`. The Interactions API is
//! Gemini's agent-runtime / multi-turn orchestration surface — distinct
//! from `:generateContent`. It carries a richer wire format (`steps[]` /
//! `model_output` / `function_call` / `thought` / built-in tool steps),
//! supports background long-running operations, and exposes a separate
//! `agent` mode with managed sandbox + sources / network configuration.
//!
//! # Scope
//!
//! - `do_generate` ✓ (foreground + background poll)
//! - `do_stream` ✓ (SSE event forwarding)
//! - Typed `provider_options.google.*` slot (agent / agentConfig / thinking
//!   / responseFormat / mediaResolution / environment / etc.) ✓
//! - Prompt -> input conversion (text / image / tool-result) ✓
//! - Response -> Content / FinishReason / Usage parsing ✓
//! - Cancel (`POST /interactions/{id}:cancel`) ✓
//!
//! Nested high-order semantics (environment sandbox preload, deep-research
//! agent thinking summaries, network allowlist transform shapes) are
//! passed through as `serde_json::Value` and not re-typed in Rust — the
//! wire shape is intentionally `.loose()` upstream too.
// Rust guideline compliant 2026-05-25

mod extract_sources;
mod model;
mod prepare_tools;
mod stream;
mod synthesize_stream;

pub use model::{
    GoogleInteractionsAgent, GoogleInteractionsLanguageModel, GoogleInteractionsStatus,
    builtin_agent,
};
