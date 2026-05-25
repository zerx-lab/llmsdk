//! `openai.local_shell` provider-defined tool.
//!
//! Mirrors `@ai-sdk/openai/src/tool/local-shell.ts`.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Args struct (no fields; the tool takes no configuration).
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct Args;

/// Action emitted as part of `local_shell_call` output items.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    Exec {
        command: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        timeout_ms: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        user: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        working_directory: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        env: Option<HashMap<String, String>>,
    },
}
