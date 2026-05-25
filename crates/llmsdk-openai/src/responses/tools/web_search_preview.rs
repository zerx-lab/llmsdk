//! `openai.web_search_preview` provider-defined tool.
//!
//! Mirrors `@ai-sdk/openai/src/tool/web-search-preview.ts`.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

use super::UserLocation;
use super::web_search::SearchContextSize;

/// Args for `Tool::Provider { id: "openai.web_search_preview", args, .. }`.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Args {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<SearchContextSize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<UserLocation>,
}
