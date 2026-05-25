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
