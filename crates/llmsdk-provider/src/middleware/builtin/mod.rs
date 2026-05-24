//! Built-in middleware implementations modelled after `@ai-sdk/ai/src/middleware/*`.
//!
//! Each submodule provides one focused decorator:
//!
//! - [`default_settings`] — fill missing call-option fields with defaults.
//! - [`default_embedding_settings`] — same idea for [`crate::EmbeddingModel`].
//! - [`extract_reasoning`] — pull `<tag>...</tag>` blocks out of text content.
//! - [`simulate_streaming`] — turn a `do_generate` response into a stream.
//! - [`extract_json`] — strip Markdown fences from JSON-format responses.
//! - [`add_tool_input_examples`] — append `inputExamples` to each tool's description.
// Rust guideline compliant 2026-02-21

pub mod add_tool_input_examples;
pub mod default_embedding_settings;
pub mod default_settings;
pub mod extract_json;
pub mod extract_reasoning;
pub mod simulate_streaming;

pub use add_tool_input_examples::AddToolInputExamplesMiddleware;
pub use default_embedding_settings::DefaultEmbeddingSettingsMiddleware;
pub use default_settings::DefaultSettingsMiddleware;
pub use extract_json::ExtractJsonMiddleware;
pub use extract_reasoning::ExtractReasoningMiddleware;
pub use simulate_streaming::SimulateStreamingMiddleware;
