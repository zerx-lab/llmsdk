//! Parse the `xai` slot of [`ProviderOptions`] for the Files API.
//!
//! Mirrors `xaiFilesOptionsSchema` from
//! `@ai-sdk/xai/src/files/xai-files-options.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::ProviderError;
use llmsdk_provider::error::Result;
use llmsdk_provider::shared::ProviderOptions;
use serde::Deserialize;

/// Typed view of `provider_options["xai"]` for `upload_file`.
///
/// All fields optional; the upstream zod schema is `.passthrough()` so
/// unknown keys are accepted silently (mirrored here via `serde(default)` +
/// no `deny_unknown_fields`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct XaiFilesOptions {
    /// `teamId`: when present, forwarded as the `team_id` form field.
    pub team_id: Option<String>,

    /// `filePath`: passthrough metadata (xAI accepts it via
    /// `.passthrough()` on the schema). Currently not used by the wire
    /// envelope; mirrored for API surface parity so unknown keys do not
    /// trip a strict deserializer.
    #[allow(dead_code, reason = "captured for parity with upstream schema")]
    pub file_path: Option<String>,
}

/// Parse `provider_options["xai"]` into [`XaiFilesOptions`].
///
/// Returns `Ok(Default::default())` when the caller did not pass any
/// xAI-specific options. Mirrors ai-sdk's `parseProviderOptions({ provider:
/// 'xai', ... })`.
///
/// # Errors
///
/// Returns [`ProviderError::invalid_argument`] when the JSON shape under
/// `xai` does not deserialize to [`XaiFilesOptions`].
pub(crate) fn parse_xai_files_options(po: Option<&ProviderOptions>) -> Result<XaiFilesOptions> {
    let Some(map) = po else {
        return Ok(XaiFilesOptions::default());
    };
    let Some(slot) = map.get("xai") else {
        return Ok(XaiFilesOptions::default());
    };
    serde_json::from_value::<XaiFilesOptions>(serde_json::Value::Object(slot.clone())).map_err(
        |err| {
            ProviderError::invalid_argument(
                "provider_options.xai",
                format!("invalid xAI files options: {err}"),
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn po(v: &serde_json::Value) -> ProviderOptions {
        let mut m = ProviderOptions::default();
        m.insert("xai".into(), v.as_object().expect("object").clone());
        m
    }

    #[test]
    fn missing_options_yields_default() {
        let r = parse_xai_files_options(None).expect("ok");
        assert!(r.team_id.is_none());
        assert!(r.file_path.is_none());
    }

    #[test]
    fn missing_xai_slot_yields_default() {
        let mut m = ProviderOptions::default();
        m.insert(
            "openai".into(),
            json!({ "reasoningEffort": "high" })
                .as_object()
                .unwrap()
                .clone(),
        );
        let r = parse_xai_files_options(Some(&m)).expect("ok");
        assert!(r.team_id.is_none());
    }

    #[test]
    fn parses_team_id() {
        let m = po(&json!({ "teamId": "team-123" }));
        let r = parse_xai_files_options(Some(&m)).expect("ok");
        assert_eq!(r.team_id.as_deref(), Some("team-123"));
    }

    #[test]
    fn parses_file_path_passthrough() {
        let m = po(&json!({ "filePath": "/tmp/x.bin" }));
        let r = parse_xai_files_options(Some(&m)).expect("ok");
        assert_eq!(r.file_path.as_deref(), Some("/tmp/x.bin"));
    }

    #[test]
    fn unknown_keys_are_ignored_for_passthrough_parity() {
        // Upstream schema uses `.passthrough()` — unknown keys must not error.
        let m = po(&json!({ "teamId": "t1", "futureKnob": 42 }));
        let r = parse_xai_files_options(Some(&m)).expect("ok");
        assert_eq!(r.team_id.as_deref(), Some("t1"));
    }

    #[test]
    fn wrong_type_for_team_id_errors() {
        let m = po(&json!({ "teamId": 42 }));
        let err = parse_xai_files_options(Some(&m)).unwrap_err();
        assert!(format!("{err}").contains("invalid xAI files options"));
    }
}
