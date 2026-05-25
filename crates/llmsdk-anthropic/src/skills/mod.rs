//! `Anthropic` Skills API (`POST /v1/skills`).
//!
//! Mirrors `@ai-sdk/anthropic/src/skills/anthropic-skills.ts`. After uploading
//! a skill bundle the returned skill id can be referenced from a Messages
//! request via `provider_options.anthropic.container.skills`.
// Rust guideline compliant 2026-02-21

mod model;
mod wire;

pub use model::AnthropicSkills;
