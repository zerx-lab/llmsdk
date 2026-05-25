//! Typed factories for xAI Responses provider-defined tools.
//!
//! Mirrors `@ai-sdk/xai/src/tool/*`. Each helper returns a
//! [`Tool::Provider`](llmsdk_provider::language_model::Tool::Provider) value
//! ready to drop into [`CallOptions::tools`].
//!
//! The factories cover the seven provider-defined tools recognised by
//! [`crate::XaiResponsesLanguageModel`]:
//!
//! - [`code_execution`] — server-side Python interpreter.
//! - [`file_search`] — vector store search.
//! - [`mcp_server`] — Model Context Protocol relay.
//! - [`view_image`] — image understanding.
//! - [`view_x_video`] — X video understanding.
//! - [`web_search`] — web search.
//! - [`x_search`] — X (formerly Twitter) search.
// Rust guideline compliant 2026-05-25

mod code_execution;
mod file_search;
mod mcp_server;
mod view_image;
mod view_x_video;
mod web_search;
mod x_search;

pub use code_execution::code_execution;
pub use file_search::{FileSearchOptions, file_search};
pub use mcp_server::{McpServerOptions, mcp_server};
pub use view_image::view_image;
pub use view_x_video::view_x_video;
pub use web_search::{WebSearchOptions, web_search};
pub use x_search::{XSearchOptions, x_search};
