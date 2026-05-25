//! xAI provider for llmsdk.
//!
//! Rust port of [`@ai-sdk/xai`](https://github.com/vercel/ai/tree/main/packages/xai).
//! Implements five model surfaces: Chat Completions ([`XaiChatModel`]),
//! Responses API ([`XaiResponsesLanguageModel`]),
//! Image Generation ([`XaiImageModel`]), Video Generation
//! ([`XaiVideoModel`]), and Files upload ([`XaiFiles`]).
// Rust guideline compliant 2026-05-25

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod chat;
mod config;
mod files;
mod image;
mod responses;
pub mod tools;
mod video;

pub use chat::XaiChatModel;
pub use config::{Xai, XaiBuilder};
pub use files::XaiFiles;
pub use image::XaiImageModel;
pub use responses::XaiResponsesLanguageModel;
pub use video::XaiVideoModel;

/// Default base URL for the xAI HTTP API.
pub const DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";

/// Environment variable consulted when no explicit API key is given.
pub const API_KEY_ENV_VAR: &str = "XAI_API_KEY";

/// Provider id reported via the `LanguageModel::provider` trait method.
pub const PROVIDER_ID: &str = "xai";
