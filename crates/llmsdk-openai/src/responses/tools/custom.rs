//! `openai.custom` provider-defined tool.
//!
//! Mirrors `@ai-sdk/openai/src/tool/custom.ts`.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

/// Args for `Tool::Provider { id: "openai.custom", args, .. }`.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct Args {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<Format>,
}

/// Output format spec; omit for unconstrained text output.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Format {
    Grammar {
        syntax: GrammarSyntax,
        definition: String,
    },
    Text,
}

/// Grammar dialect for `Format::Grammar`.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GrammarSyntax {
    Regex,
    Lark,
}
