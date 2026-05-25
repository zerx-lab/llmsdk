//! `OpenAI` Transcription (speech-to-text, `POST /v1/audio/transcriptions`).
//!
//! Mirrors `@ai-sdk/openai/src/transcription/*`.
// Rust guideline compliant 2026-02-21

mod model;

pub use model::OpenAiTranscriptionModel;
