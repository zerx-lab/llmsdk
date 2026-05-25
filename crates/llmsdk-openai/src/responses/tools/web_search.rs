//! `openai.web_search` provider-defined tool.
//!
//! Mirrors `@ai-sdk/openai/src/tool/web-search.ts`.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

use super::UserLocation;

/// Args for `Tool::Provider { id: "openai.web_search", args, .. }`.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Args {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_web_access: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<Filters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<SearchContextSize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<UserLocation>,
}

/// Filters narrowing the search domain.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Filters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
}

/// `searchContextSize` enum.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SearchContextSize {
    Low,
    Medium,
    High,
}

/// Action description returned in the `web_search_call` output item.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    Search {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        query: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sources: Option<Vec<Source>>,
    },
    OpenPage {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
    },
    FindInPage {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pattern: Option<String>,
    },
}

/// Cited source returned alongside a `search` action.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Source {
    Url { url: String },
    Api { name: String },
}
