//! `OpenAI` Skills API (`POST /v1/skills`).
//!
//! Mirrors `@ai-sdk/openai/src/skills/*`. Bundles a set of files into a
//! reusable "skill" referenced from later API calls.
// Rust guideline compliant 2026-02-21

mod model;
mod wire;

pub use model::OpenAiSkills;
