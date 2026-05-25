//! Gemini image generation.
//!
//! Mirrors `@ai-sdk/google/src/google-image-model.ts`. Supports both Imagen
//! models (`imagen-*`, via `:predict`) and Gemini image-output models
//! (`gemini-*-image*`, via the language-model path with
//! `responseModalities: ["IMAGE"]`).
// Rust guideline compliant 2026-05-25

mod model;
mod options;
mod wire;

pub use model::GoogleImageModel;
