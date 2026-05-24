//! Middleware layer for the language model surface.
//!
//! Mirrors `@ai-sdk/provider/src/language-model-middleware/v4/*` plus the
//! `wrapLanguageModel` helper from `@ai-sdk/ai/src/middleware/*`. See
//! `architecture/0002-middleware-design.md` for the design rationale.
//!
//! Middleware lets callers stack cross-cutting concerns (retry, logging,
//! caching, ...) on top of any [`LanguageModel`] without modifying the
//! provider implementation:
//!
//! ```ignore
//! use std::sync::Arc;
//! use llmsdk_provider::{wrap_language_model, LanguageModel};
//! // (Built-in middleware implementations land in M9.3.)
//! ```
//!
//! [`LanguageModel`]: crate::LanguageModel
// Rust guideline compliant 2026-02-21

pub mod builtin;
mod cache;
mod context;
mod embedding_model;
mod image_model;
mod language_model;
mod logging;
mod provider;
mod retry;

pub use cache::{
    CacheMiddleware, CacheStore, CachedEntry, MemoryCacheStore, MemoryCacheStoreBuilder,
};
pub use context::{LLMSDK_OPTIONS_KEY, MiddlewareContext};
pub use embedding_model::{EmbeddingModelMiddleware, wrap_embedding_model};
pub use image_model::{ImageModelMiddleware, wrap_image_model};
pub use language_model::{CallKind, LanguageModelMiddleware, wrap_language_model};
pub use logging::{
    LogCallEnd, LogCallError, LogCallStart, LogContext, Logger, LoggingMiddleware, StderrLogger,
};
pub use provider::{ProviderMiddlewareSet, wrap_provider};
pub use retry::{
    DEFAULT_BACKOFF_MULTIPLIER, DEFAULT_INITIAL_BACKOFF, DEFAULT_MAX_ATTEMPTS, DEFAULT_MAX_BACKOFF,
    RetryMiddleware, RetryMiddlewareBuilder,
};
