//! `openai.image_generation` provider-defined tool.
//!
//! Mirrors `@ai-sdk/openai/src/tool/image-generation.ts`.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

/// Args for `Tool::Provider { id: "openai.image_generation", args, .. }`.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Args {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<Background>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_fidelity: Option<InputFidelity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_image_mask: Option<InputImageMask>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moderation: Option<Moderation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_compression: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<OutputFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_images: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<Quality>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<Size>,
}

/// `background` enum.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Background {
    Auto,
    Opaque,
    Transparent,
}

/// `inputFidelity` enum.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum InputFidelity {
    Low,
    High,
}

/// `moderation` enum (only `auto` currently).
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Moderation {
    Auto,
}

/// `outputFormat` enum.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    Png,
    Jpeg,
    Webp,
}

/// `quality` enum.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Quality {
    Auto,
    Low,
    Medium,
    High,
}

/// `size` enum.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub enum Size {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "1024x1024")]
    Square1024,
    #[serde(rename = "1024x1536")]
    Portrait1024x1536,
    #[serde(rename = "1536x1024")]
    Landscape1536x1024,
}

/// Optional mask used for inpainting.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InputImageMask {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}
