//! Language model trait and supporting types.
//!
//! Maps to `@ai-sdk/provider/src/language-model/v4/*`. The trait describes
//! a *raw* model interface — user-facing prompt formats (chat, instruction,
//! ...) are translated into [`Prompt`] before reaching the trait.
// Rust guideline compliant 2026-02-21

mod call_options;
mod content;
mod finish_reason;
mod prompt;
mod result;
mod stream_part;
mod tool;
mod usage;

pub use call_options::{CallOptions, ReasoningEffort, ResponseFormat};
pub use content::{
    Content, ReasoningPart, Source, ToolApprovalRequest, ToolResult, ToolResultOutput,
};
pub use finish_reason::{FinishReason, FinishReasonKind};
pub use prompt::{
    AssistantPart, FilePart, Message, Prompt, TextPart, ToolApprovalResponsePart, ToolCallPart,
    ToolMessagePart, ToolResultPart, UserPart,
};
pub use result::{
    GenerateResponse, GenerateResult, ResponseMetadata, StreamResponse, StreamResult,
    SupportedUrls, UrlPattern,
};
pub use stream_part::StreamPart;
pub use tool::{FunctionTool, ProviderTool, Tool, ToolChoice, ToolInputExample};
pub use usage::{InputTokenUsage, OutputTokenUsage, Usage};

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::error::Result;

/// Boxed `Send` stream alias used for streaming results.
pub type BoxStream<T> = Pin<Box<dyn Stream<Item = T> + Send>>;

/// Contract every chat / completion model implements.
///
/// Mirrors `LanguageModelV4`. Method names keep the `do_` prefix from ai-sdk
/// to discourage direct end-user usage; downstream `llmsdk` crates wrap
/// these into ergonomic helpers.
#[async_trait]
pub trait LanguageModel: Send + Sync + std::fmt::Debug {
    /// Provider id, e.g. `"openai"`.
    fn provider(&self) -> &str;

    /// Provider-specific model id, e.g. `"gpt-4o-mini"`.
    fn model_id(&self) -> &str;

    /// URL patterns the model can ingest natively, by media type.
    ///
    /// Returning a pattern tells the SDK *not* to download a matching URL
    /// before calling the model. Defaults to an empty map (no native URL
    /// support).
    async fn supported_urls(&self) -> SupportedUrls {
        SupportedUrls::default()
    }

    /// Run a non-streaming generation.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::ProviderError`] when the upstream call fails,
    /// the response is malformed, or the prompt is rejected.
    async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult>;

    /// Run a streaming generation.
    ///
    /// The returned stream yields [`StreamPart`]s. Errors come in two flavors:
    ///
    /// - Outer `Result::Err` — the call itself failed before any data flowed.
    /// - Inner [`StreamPart::Error`] — the stream is alive but the provider
    ///   reported a recoverable issue (content filter, partial failure).
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::do_generate`].
    async fn do_stream(&self, options: CallOptions) -> Result<StreamResult>;
}
