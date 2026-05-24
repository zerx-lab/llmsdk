//! Provider trait abstractions for llmsdk.
//!
//! This crate is the Rust port of [`@ai-sdk/provider`](https://github.com/vercel/ai/tree/main/packages/provider)
//! at specification version `v4`. It defines the contract every model provider
//! (`OpenAI`, `Anthropic`, ...) implements, plus the shared message / content /
//! streaming types and the unified error type.
//!
//! The crate is intentionally minimal: no HTTP, no retry, no middleware.
//! Those live in `llmsdk-provider-utils` and downstream provider crates.
//!
//! # Layout
//!
//! - [`language_model`]: chat / completion models with streaming.
//! - [`embedding_model`]: vector embedding models.
//! - [`image_model`]: image generation models.
//! - [`provider`]: top-level factory returning model instances by id.
//! - [`error`]: unified [`ProviderError`].
//! - [`shared`]: provider options / metadata / warnings reused across models.
//!
//! # Example
//!
//! ```
//! use llmsdk_provider::SPECIFICATION_VERSION;
//! assert_eq!(SPECIFICATION_VERSION, "v4");
//! ```
//!
//! # Stability
//!
//! Until 1.0, expect breaking changes; the spec version pins compatibility
//! with the matching `@ai-sdk/provider` major.
// Rust guideline compliant 2026-02-21

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod embedding_model;
pub mod error;
pub mod image_model;
pub mod json;
pub mod language_model;
pub mod provider;
pub mod shared;

#[doc(inline)]
pub use embedding_model::EmbeddingModel;
#[doc(inline)]
pub use error::{ProviderError, Result};
#[doc(inline)]
pub use image_model::ImageModel;
#[doc(inline)]
pub use language_model::LanguageModel;
#[doc(inline)]
pub use provider::Provider;

/// Specification version this crate implements.
///
/// Matches `@ai-sdk/provider` v4. Providers must be wire-compatible with this
/// spec version.
pub const SPECIFICATION_VERSION: &str = "v4";
