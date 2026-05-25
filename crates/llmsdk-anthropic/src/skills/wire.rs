//! Wire types for `POST /v1/skills` and `GET /v1/skills/{id}/versions/{v}`.
//!
//! Mirrors `@ai-sdk/anthropic/src/skills/anthropic-skills-api.ts`.
// Rust guideline compliant 2026-02-21

use serde::Deserialize;

/// Successful response body from `POST /v1/skills`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct WireSkillResponse {
    /// Server-assigned skill id.
    pub id: String,
    /// Optional display title echoed back.
    #[serde(default)]
    pub display_title: Option<String>,
    /// Optional skill name.
    #[serde(default)]
    pub name: Option<String>,
    /// Optional description.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional latest version identifier; used to fetch the version metadata.
    #[serde(default)]
    pub latest_version: Option<String>,
    /// Source identifier (e.g. `"user"`).
    pub source: String,
    /// RFC 3339 creation timestamp.
    pub created_at: String,
    /// RFC 3339 update timestamp.
    pub updated_at: String,
}

/// Response from `GET /v1/skills/{id}/versions/{version}`. Used to refine
/// `name` / `description` after upload (the upload response itself only
/// reports them when the server side resolved them eagerly).
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct WireSkillVersionResponse {
    /// Object type (`"skill_version"`).
    #[serde(rename = "type")]
    pub _kind: String,
    /// Skill id (currently only used for parse-correctness tests).
    #[allow(
        dead_code,
        reason = "captured for forward compat; consumed only in tests"
    )]
    pub skill_id: String,
    /// Resolved skill name.
    #[serde(default)]
    pub name: Option<String>,
    /// Resolved skill description.
    #[serde(default)]
    pub description: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_skill_response_minimum() {
        let json = serde_json::json!({
            "id": "skill-abc",
            "source": "user",
            "created_at": "2025-10-02T00:00:00Z",
            "updated_at": "2025-10-02T00:00:00Z"
        });
        let r: WireSkillResponse = serde_json::from_value(json).unwrap();
        assert_eq!(r.id, "skill-abc");
        assert_eq!(r.source, "user");
        assert!(r.display_title.is_none());
        assert!(r.latest_version.is_none());
    }

    #[test]
    fn parses_skill_response_full() {
        let json = serde_json::json!({
            "id": "skill-1",
            "display_title": "Greeter",
            "name": "greeter",
            "description": "Says hi",
            "latest_version": "v1",
            "source": "user",
            "created_at": "2025-10-02T00:00:00Z",
            "updated_at": "2025-10-02T00:00:00Z"
        });
        let r: WireSkillResponse = serde_json::from_value(json).unwrap();
        assert_eq!(r.latest_version.as_deref(), Some("v1"));
        assert_eq!(r.name.as_deref(), Some("greeter"));
    }

    #[test]
    fn parses_version_response() {
        let json = serde_json::json!({
            "type": "skill_version",
            "skill_id": "skill-1",
            "name": "greeter-v1",
            "description": "Says hi (v1)"
        });
        let r: WireSkillVersionResponse = serde_json::from_value(json).unwrap();
        assert_eq!(r.skill_id, "skill-1");
        assert_eq!(r.name.as_deref(), Some("greeter-v1"));
    }
}
