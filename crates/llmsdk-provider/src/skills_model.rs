//! Skill upload model trait and supporting types.
//!
//! Mirrors `@ai-sdk/provider/src/skills/v4/*`. A "skill" is a bundle of files
//! grouped under a single identifier; the only current provider is
//! `Anthropic` (`POST /v1/skills`, beta `skills-2025-10-02`), where the
//! returned skill id can be referenced from the Messages API
//! `container.skills[]` field.
//!
//! Kept separate from [`crate::Provider`] for the same reason as
//! [`crate::FilesModel`]: only providers that expose a skills endpoint
//! implement this trait.
// Rust guideline compliant 2026-02-21

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::files_model::UploadFileData;
use crate::shared::{ProviderMetadata, ProviderOptions, ProviderReference, Warning};

/// Contract every skill-upload model implements.
///
/// Mirrors `SkillsV4`.
#[async_trait]
pub trait SkillsModel: Send + Sync + std::fmt::Debug {
    /// Provider id, e.g. `"anthropic.skills"`.
    fn provider(&self) -> &str;

    /// Specification version (currently `"v4"`).
    fn specification_version(&self) -> &'static str {
        "v4"
    }

    /// Upload a skill from the given files.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::ProviderError`] when the upstream call fails or
    /// the response is malformed.
    async fn upload_skill(&self, options: UploadSkillOptions) -> Result<UploadSkillResult>;
}

/// One file within a skill bundle.
///
/// Mirrors `SkillsV4File`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillFile {
    /// Path of the file relative to the skill root (e.g. `"main.py"`).
    pub path: String,
    /// File payload.
    pub data: UploadFileData,
}

/// Options for one [`SkillsModel::upload_skill`] call.
///
/// Mirrors `SkillsV4UploadSkillCallOptions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadSkillOptions {
    /// Files that make up the skill (at least one).
    pub files: Vec<SkillFile>,
    /// Optional human-readable title.
    #[serde(
        default,
        rename = "displayTitle",
        skip_serializing_if = "Option::is_none"
    )]
    pub display_title: Option<String>,
    /// Provider-specific options.
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_options: Option<ProviderOptions>,
}

/// Result of [`SkillsModel::upload_skill`].
///
/// Mirrors `SkillsV4UploadSkillResult`.
#[derive(Debug, Clone)]
pub struct UploadSkillResult {
    /// `{ providerId → skillId }` reference.
    pub provider_reference: ProviderReference,
    /// Display title as stored by the provider.
    pub display_title: Option<String>,
    /// Skill name (often resolved from the latest version manifest).
    pub name: Option<String>,
    /// Skill description.
    pub description: Option<String>,
    /// Latest version identifier reported by the provider.
    pub latest_version: Option<String>,
    /// Provider-specific metadata.
    pub provider_metadata: Option<ProviderMetadata>,
    /// Warnings (e.g. setting coerced away).
    pub warnings: Vec<Warning>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::FileBytes;

    #[test]
    fn options_serde_roundtrip() {
        let opts = UploadSkillOptions {
            files: vec![SkillFile {
                path: "main.py".into(),
                data: UploadFileData::Text {
                    text: "print(1)".into(),
                },
            }],
            display_title: Some("greeter".into()),
            provider_options: None,
        };
        let json = serde_json::to_value(&opts).unwrap();
        assert_eq!(json["displayTitle"], "greeter");
        assert_eq!(json["files"][0]["path"], "main.py");
        assert_eq!(json["files"][0]["data"]["type"], "text");
    }

    #[test]
    fn skill_file_supports_bytes() {
        let f = SkillFile {
            path: "asset.bin".into(),
            data: UploadFileData::Data {
                data: FileBytes::Bytes(vec![1, 2, 3]),
            },
        };
        let v = serde_json::to_value(&f).unwrap();
        assert_eq!(v["data"]["type"], "data");
    }
}
