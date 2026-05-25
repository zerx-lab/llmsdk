//! Typed factories for Anthropic provider-defined tools.
//!
//! Mirrors `@ai-sdk/anthropic/src/anthropic-tools.ts` + the per-tool files in
//! `@ai-sdk/anthropic/src/tool/`. Each factory returns a
//! [`llmsdk_provider::language_model::Tool::Provider`] keyed by an
//! `anthropic.*` id; the routing table in
//! `crate::messages::model::resolve_anthropic_server_tool` translates the id
//! to the wire `type` + `name` + beta header for the Messages API.
//!
//! ## Coverage (20 tools)
//!
//! - [`bash_20241022`] / [`bash_20250124`]
//! - [`code_execution_20250522`] / [`code_execution_20250825`] / [`code_execution_20260120`]
//! - [`computer_20241022`] / [`computer_20250124`] / [`computer_20251124`]
//! - [`memory_20250818`]
//! - [`text_editor_20241022`] / [`text_editor_20250124`] / [`text_editor_20250429`] / [`text_editor_20250728`]
//! - [`web_fetch_20250910`] / [`web_fetch_20260209`]
//! - [`web_search_20250305`] / [`web_search_20260209`]
//! - [`tool_search_regex_20251119`] / [`tool_search_bm25_20251119`]
//! - [`advisor_20260301`]
// Rust guideline compliant 2026-02-21

mod advisor;
mod common;
mod computer;
mod no_args;
mod text_editor;
mod web_fetch;
mod web_search;

pub use advisor::{AdvisorArgs, advisor_20260301};
pub use common::{
    CitationsConfig, EphemeralCache, EphemeralCacheKind, EphemeralCacheTtl, UserLocation,
    UserLocationKind,
};
pub use computer::{
    ComputerArgs, ComputerArgsWithZoom, computer_20241022, computer_20250124, computer_20251124,
};
#[allow(deprecated, reason = "intentionally re-exported deprecated factory")]
pub use no_args::text_editor_20250429;
pub use no_args::{
    bash_20241022, bash_20250124, code_execution_20250522, code_execution_20250825,
    code_execution_20260120, memory_20250818, text_editor_20241022, text_editor_20250124,
    tool_search_bm25_20251119, tool_search_regex_20251119,
};
pub use text_editor::{TextEditor20250728Args, text_editor_20250728};
pub use web_fetch::{WebFetchArgs, web_fetch_20250910, web_fetch_20260209};
pub use web_search::{WebSearchArgs, web_search_20250305, web_search_20260209};
