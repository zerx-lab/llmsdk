//! Typed factories for Google-defined provider tools (Vertex subset).
//!
//! Mirrors `google-vertex-tools.ts`. Re-exports the upstream-recognized
//! subset of `llmsdk_google::tools` for Vertex. The `vertex_rag_store`
//! tool is Vertex-only (already returns `Tool::Provider` with
//! `id="google.vertex_rag_store"`); the rest are shared with Public
//! API Gemini.
// Rust guideline compliant 2026-05-25

pub use llmsdk_google::tools::{
    FileSearchArgs, GoogleSearchArgs, GoogleSearchTimeRange, GoogleSearchTypes, VertexRagStoreArgs,
    code_execution, enterprise_web_search, file_search, google_maps, google_search, url_context,
    vertex_rag_store,
};

/// Anthropic typed tools recognized by Vertex AI.
///
/// Mirrors `googleVertexAnthropicTools` in the upstream JS provider — a
/// curated 10-tool subset of `@ai-sdk/anthropic`'s full server-tool catalog
/// that the Vertex deployment of Claude actually accepts. Use these
/// factories instead of `llmsdk_anthropic::tools::*` when targeting
/// Vertex so the compiler rejects versions the Vertex API would 400 on.
pub mod anthropic_tools {
    #[allow(deprecated, reason = "kept for parity with upstream deprecated alias")]
    pub use llmsdk_anthropic::tools::text_editor_20250429;
    pub use llmsdk_anthropic::tools::{
        ComputerArgs, TextEditor20250728Args, WebSearchArgs, bash_20241022, bash_20250124,
        computer_20241022, text_editor_20241022, text_editor_20250124, text_editor_20250728,
        tool_search_bm25_20251119, tool_search_regex_20251119, web_search_20250305,
    };
}
