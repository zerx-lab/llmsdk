//! Azure-specific tool re-exports.
//!
//! Mirrors `@ai-sdk/azure/src/azure-openai-tools.ts`. Upstream defines
//! `azureOpenaiTools` as a four-key alias object pointing at the
//! corresponding `@ai-sdk/openai` exports. On the Rust side we just
//! re-export the `OpenAI` tool argument / output modules directly:
//! `Azure-on-OpenAI` uses the **same** `openai.*` provider-defined tool
//! ids as plain `OpenAI`, so callers construct `Tool::Provider` the exact
//! same way.
//!
//! Includes all eleven `OpenAI` Responses-API tool modules (the four
//! mentioned in upstream `azureOpenaiTools` — `code_interpreter` /
//! `file_search` / `image_generation` / `web_search_preview` — plus the
//! other seven: `apply_patch`, `custom`, `local_shell`, `mcp`, `shell`,
//! `tool_search`, `web_search`). `Azure-on-OpenAI` accepts the same set
//! as plain `OpenAI`.
//!
//! ```no_run
//! use llmsdk_azure::azure_openai_tools::web_search_preview;
//!
//! let _args = web_search_preview::Args::default();
//! ```
// Rust guideline compliant 2026-02-21

pub use llmsdk_openai::internal::tools as azure_openai_tools;
