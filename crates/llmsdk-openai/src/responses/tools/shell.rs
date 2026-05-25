//! `openai.shell` provider-defined tool.
//!
//! Mirrors `@ai-sdk/openai/src/tool/shell.ts`.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Args for `Tool::Provider { id: "openai.shell", args, .. }`.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Args {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<Environment>,
}

/// Three-way `environment` shape.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Environment {
    ContainerAuto {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_ids: Option<Vec<String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        memory_limit: Option<MemoryLimit>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        network_policy: Option<NetworkPolicy>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        skills: Option<Vec<AutoSkill>>,
    },
    ContainerReference {
        container_id: String,
    },
    Local {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        skills: Option<Vec<LocalSkill>>,
    },
}

/// `memoryLimit` enum.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub enum MemoryLimit {
    #[serde(rename = "1g")]
    Gb1,
    #[serde(rename = "4g")]
    Gb4,
    #[serde(rename = "16g")]
    Gb16,
    #[serde(rename = "64g")]
    Gb64,
}

/// Network policy for `containerAuto`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum NetworkPolicy {
    Disabled,
    Allowlist {
        allowed_domains: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        domain_secrets: Option<Vec<DomainSecret>>,
    },
}

/// Per-domain secret rule.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct DomainSecret {
    pub domain: String,
    pub name: String,
    pub value: String,
}

/// Skill entry for `containerAuto`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AutoSkill {
    SkillReference {
        provider_reference: JsonValue,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        version: Option<String>,
    },
    Inline {
        name: String,
        description: String,
        source: InlineSkillSource,
    },
}

/// Base64-encoded zip source for an inline skill.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InlineSkillSource {
    /// Discriminator (only `"base64"`).
    #[serde(rename = "type")]
    pub kind: String,
    pub media_type: String,
    pub data: String,
}

/// Skill entry for `local` environment.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LocalSkill {
    pub name: String,
    pub description: String,
    pub path: String,
}

/// Output rows for `shell_call_output`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct OutputRow {
    pub stdout: String,
    pub stderr: String,
    pub outcome: Outcome,
}

/// `outcome` discriminator (`exit` carries an `exit_code`).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Outcome {
    Timeout,
    Exit {
        #[serde(rename = "exitCode")]
        exit_code: i32,
    },
}
