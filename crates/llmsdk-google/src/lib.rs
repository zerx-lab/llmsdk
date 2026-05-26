//! Google Gemini provider for llmsdk.
//!
//! Rust port of [`@ai-sdk/google`](https://github.com/vercel/ai/tree/main/packages/google).
//! Implements five model surfaces against the Gemini Generative Language
//! API: chat ([`GoogleLanguageModel`]), text embeddings
//! ([`GoogleEmbeddingModel`]), image generation
//! ([`GoogleImageModel`]), video generation ([`GoogleVideoModel`]), and
//! file uploads ([`GoogleFiles`]).
// Rust guideline compliant 2026-05-25

#![forbid(unsafe_code)]
#![warn(missing_docs)]
// Crate-wide allow-list. Mirrors the convention used in
// `llmsdk-xai/src/responses/mod.rs` and `llmsdk-anthropic/src/messages/`:
// many wire structs deserialize fields we re-emit verbatim, and the wire
// → trait conversion functions are intrinsically long state machines.
#![allow(
    clippy::module_name_repetitions,
    clippy::needless_pass_by_value,
    clippy::too_many_lines,
    clippy::map_unwrap_or,
    clippy::redundant_else,
    clippy::manual_let_else,
    clippy::semicolon_if_nothing_returned,
    clippy::collapsible_if,
    clippy::collapsible_else_if,
    clippy::single_match_else,
    clippy::match_same_arms,
    clippy::needless_pass_by_ref_mut,
    clippy::unnecessary_wraps,
    clippy::doc_markdown,
    clippy::needless_borrow,
    clippy::needless_borrows_for_generic_args,
    clippy::redundant_closure_for_method_calls,
    clippy::default_trait_access,
    clippy::option_if_let_else,
    clippy::manual_unwrap_or,
    clippy::manual_unwrap_or_default,
    clippy::implicit_clone,
    clippy::redundant_clone,
    clippy::single_char_pattern,
    clippy::if_not_else,
    clippy::case_sensitive_file_extension_comparisons,
    clippy::needless_continue,
    clippy::ref_option,
    clippy::trivially_copy_pass_by_ref,
    clippy::unnested_or_patterns,
    clippy::needless_range_loop,
    clippy::single_match,
    clippy::nonminimal_bool,
    reason = "wire conversion modules and option parsers are deliberately verbose for parity with ai-sdk"
)]

mod base64;
mod config;
mod embedding;
mod error;
mod files;
mod image;
mod interactions;
mod language;
mod schema;
mod video;

pub mod tools;

pub use config::{Google, GoogleBuilder};
pub use embedding::GoogleEmbeddingModel;
pub use files::GoogleFiles;
pub use image::GoogleImageModel;
pub use interactions::{
    GoogleInteractionsAgent, GoogleInteractionsLanguageModel, GoogleInteractionsStatus,
    builtin_agent,
};
pub use language::GoogleLanguageModel;
pub use video::GoogleVideoModel;

/// Internal API surface for cross-crate composition.
///
/// Re-exports the wiring primitives ([`Inner`]) plus model `new()`
/// constructors so wrapping providers (Google Vertex today; potentially
/// other Gemini-compatible deployments tomorrow) can build models without
/// going through the [`Google`] builder.
///
/// **Stability**: this module mirrors `@ai-sdk/google/internal` — it is
/// considered semi-public and may evolve more frequently than the
/// user-facing API. Pin a specific version if you depend on it directly.
///
/// [`Inner`]: crate::config::Inner
pub mod internal {
    pub use crate::config::{Inner, InnerBuilder};
    pub use crate::language::GoogleLanguageModel;
}

/// Default base URL for the Google Generative AI HTTP API.
pub const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Environment variable consulted when no explicit API key is given.
pub const API_KEY_ENV_VAR: &str = "GOOGLE_GENERATIVE_AI_API_KEY";

/// Provider id reported via the `*Model::provider` trait methods.
pub const PROVIDER_ID: &str = "google";
