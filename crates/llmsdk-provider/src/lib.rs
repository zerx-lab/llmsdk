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
//! - [`video_model`]: video generation models.
//! - [`reranking_model`]: document reranking models.
//! - [`files_model`]: file-upload models (e.g. `Anthropic`'s `POST /files`).
//! - [`skills_model`]: skill-upload models (`Anthropic` skills bundles).
//! - [`middleware`]: decorators for stacking cross-cutting concerns
//!   (retry / logging / caching) on top of any [`LanguageModel`].
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
pub mod files_model;
pub mod image_model;
pub mod json;
pub mod language_model;
pub mod middleware;
pub mod provider;
pub mod reranking_model;
pub mod shared;
pub mod skills_model;
pub mod speech_model;
pub mod transcription_model;
pub mod video_model;

// === Top-level re-exports ===
//
// High-frequency types are pulled to the crate root so downstream code can
// write `use llmsdk_provider::CallOptions;` instead of
// `use llmsdk_provider::language_model::CallOptions;`. The original
// `pub mod` paths above remain available for callers that prefer the
// fully qualified form.

// --- Traits ---
#[doc(inline)]
pub use embedding_model::EmbeddingModel;
#[doc(inline)]
pub use files_model::FilesModel;
#[doc(inline)]
pub use image_model::ImageModel;
#[doc(inline)]
pub use language_model::LanguageModel;
#[doc(inline)]
pub use provider::Provider;
#[doc(inline)]
pub use reranking_model::RerankingModel;
#[doc(inline)]
pub use skills_model::SkillsModel;
#[doc(inline)]
pub use speech_model::SpeechModel;
#[doc(inline)]
pub use transcription_model::TranscriptionModel;
#[doc(inline)]
pub use video_model::VideoModel;

// --- Error ---
#[doc(inline)]
pub use error::{ApiCallErrorBuilder, ProviderError, Result};

// --- JSON ---
#[doc(inline)]
pub use json::{JsonObject, JsonSchema, JsonValue};

// --- Shared ---
#[doc(inline)]
pub use shared::{
    FileBytes, FileData, Headers, ProviderMetadata, ProviderOptions, ProviderReference,
    RequestInfo, ResponseInfo, Warning,
};

// --- language_model ---
#[doc(inline)]
pub use language_model::{
    AssistantPart, BoxStream, CallOptions, Content, FilePart, FinishReason, FinishReasonKind,
    FunctionTool, GenerateResponse, GenerateResult, InputTokenUsage, Message, OutputTokenUsage,
    Prompt, ProviderTool, ReasoningEffort, ReasoningPart, ResponseFormat, ResponseMetadata, Source,
    StreamPart, StreamResponse, StreamResult, SupportedUrls, TextPart, Tool, ToolApprovalRequest,
    ToolApprovalResponsePart, ToolCallPart, ToolChoice, ToolInputExample, ToolMessagePart,
    ToolOutputPart, ToolResult, ToolResultOutput, ToolResultPart, UrlPattern, Usage, UserPart,
};

// --- embedding_model ---
#[doc(inline)]
pub use embedding_model::{EmbedOptions, EmbedResult, Embedding, EmbeddingUsage};

// --- image_model ---
#[doc(inline)]
pub use image_model::{
    GeneratedImage, ImageOptions, ImageResult, ImageUsage, ImageUsageInputDetails,
};

// --- video_model ---
#[doc(inline)]
pub use video_model::{VideoData, VideoFile, VideoOptions, VideoResponseInfo, VideoResult};

// --- reranking_model ---
#[doc(inline)]
pub use reranking_model::{RankingEntry, RerankingDocuments, RerankingOptions, RerankingResult};

// --- files_model ---
#[doc(inline)]
pub use files_model::{UploadFileData, UploadFileOptions, UploadFileResult};

// --- skills_model ---
#[doc(inline)]
pub use skills_model::{SkillFile, UploadSkillOptions, UploadSkillResult};

// --- speech_model ---
#[doc(inline)]
pub use speech_model::{SpeechOptions, SpeechResponseInfo, SpeechResult};

// --- transcription_model ---
#[doc(inline)]
pub use transcription_model::{
    TranscriptionOptions, TranscriptionResponseInfo, TranscriptionResult, TranscriptionSegment,
};

// --- provider ---
#[doc(inline)]
pub use provider::{DynEmbeddingModel, DynImageModel, DynLanguageModel};

// --- middleware ---
#[doc(inline)]
pub use middleware::{
    CacheMiddleware, CacheStore, CachedEntry, CallKind, EmbeddingModelMiddleware,
    ImageModelMiddleware, LanguageModelMiddleware, Logger, LoggingMiddleware, MemoryCacheStore,
    MemoryCacheStoreBuilder, MiddlewareContext, ProviderMiddlewareSet, RerankingModelMiddleware,
    RetryMiddleware, RetryMiddlewareBuilder, StderrLogger, VideoModelMiddleware,
    wrap_embedding_model, wrap_image_model, wrap_language_model, wrap_provider,
    wrap_reranking_model, wrap_video_model,
};

/// Specification version this crate implements.
///
/// Matches `@ai-sdk/provider` v4. Providers must be wire-compatible with this
/// spec version.
pub const SPECIFICATION_VERSION: &str = "v4";
