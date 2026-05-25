//! Bedrock image generation (Nova Canvas / Titan Image / SDXL).
//!
//! Mirrors `@ai-sdk/amazon-bedrock/src/amazon-bedrock-image-model.ts`.
//! Covers all five Nova task types:
//!
//! - `TEXT_IMAGE` — text-to-image (default when no `files`)
//! - `IMAGE_VARIATION` — `files[]` without mask / mask prompt
//! - `INPAINTING` — `mask` or `maskPrompt` provided
//! - `OUTPAINTING` — `outPaintingMode` provider option set
//! - `BACKGROUND_REMOVAL` — explicit `taskType = "BACKGROUND_REMOVAL"`
// Rust guideline compliant 2026-05-25

mod model;
mod options;
mod wire;

pub use model::AmazonBedrockImageModel;
