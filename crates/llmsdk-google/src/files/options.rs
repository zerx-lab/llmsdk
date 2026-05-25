//! Provider options parser for the Gemini Files API.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::ProviderError;
use llmsdk_provider::shared::ProviderOptions;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct FilesUploadOptions {
    #[serde(default, rename = "displayName")]
    pub display_name: Option<String>,
    #[serde(default, rename = "pollIntervalMs")]
    pub poll_interval_ms: Option<u64>,
    #[serde(default, rename = "pollTimeoutMs")]
    pub poll_timeout_ms: Option<u64>,
    #[serde(flatten)]
    #[allow(dead_code, reason = "passthrough kept for parity with upstream")]
    pub extras: serde_json::Map<String, Value>,
}

pub(crate) fn parse(
    provider_options: Option<&ProviderOptions>,
) -> Result<Option<FilesUploadOptions>, ProviderError> {
    let Some(opts) = provider_options else {
        return Ok(None);
    };
    let Some(payload) = opts.get("google") else {
        return Ok(None);
    };
    let value = Value::Object(payload.clone());
    let parsed: FilesUploadOptions = serde_json::from_value(value.clone()).map_err(|e| {
        ProviderError::type_validation("provider_options.google", value, e.to_string())
    })?;
    Ok(Some(parsed))
}
