//! `provider_options["google"]` parser for Gemini language calls.
//!
//! Mirrors `googleLanguageModelOptions` in
//! `@ai-sdk/google/src/google-language-model-options.ts`. All fields are
//! optional; unknown fields are ignored.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::ProviderError;
use llmsdk_provider::shared::ProviderOptions;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Parsed provider options ready to merge into the wire body.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct GoogleOptions {
    #[serde(default, rename = "responseModalities")]
    pub response_modalities: Option<Vec<String>>,
    #[serde(default, rename = "thinkingConfig")]
    pub thinking_config: Option<ThinkingConfig>,
    #[serde(default, rename = "cachedContent")]
    pub cached_content: Option<String>,
    #[serde(default, rename = "structuredOutputs")]
    pub structured_outputs: Option<bool>,
    #[serde(default, rename = "safetySettings")]
    pub safety_settings: Option<Vec<Value>>,
    #[serde(default)]
    pub threshold: Option<String>,
    #[serde(default, rename = "audioTimestamp")]
    pub audio_timestamp: Option<bool>,
    #[serde(default)]
    pub labels: Option<Value>,
    #[serde(default, rename = "mediaResolution")]
    pub media_resolution: Option<String>,
    #[serde(default, rename = "imageConfig")]
    pub image_config: Option<Value>,
    #[serde(default, rename = "retrievalConfig")]
    pub retrieval_config: Option<Value>,
    #[serde(default, rename = "streamFunctionCallArguments")]
    pub stream_function_call_arguments: Option<bool>,
    #[serde(default, rename = "serviceTier")]
    pub service_tier: Option<String>,
    #[serde(default, rename = "sharedRequestType")]
    pub shared_request_type: Option<String>,
    #[serde(default, rename = "requestType")]
    pub request_type: Option<String>,
}

/// Configuration for the `thinkingConfig` field.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct ThinkingConfig {
    #[serde(default, rename = "thinkingBudget")]
    pub thinking_budget: Option<i64>,
    #[serde(default, rename = "includeThoughts")]
    pub include_thoughts: Option<bool>,
    #[serde(default, rename = "thinkingLevel")]
    pub thinking_level: Option<String>,
}

/// Read `provider_options[name]` (first match wins among the candidate
/// keys) and deserialize into [`GoogleOptions`].
///
/// # Errors
///
/// Returns [`ProviderError::type_validation`] when a field has the wrong
/// type.
pub(crate) fn parse(
    provider_options: Option<&ProviderOptions>,
    candidates: &[&str],
) -> Result<Option<GoogleOptions>, ProviderError> {
    let Some(opts) = provider_options else {
        return Ok(None);
    };
    for name in candidates {
        if let Some(payload) = opts.get(*name) {
            let value = Value::Object(payload.clone());
            let parsed: GoogleOptions = serde_json::from_value(value.clone()).map_err(|e| {
                ProviderError::type_validation(
                    format!("provider_options.{name}"),
                    value,
                    e.to_string(),
                )
            })?;
            return Ok(Some(parsed));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_full_options() {
        let mut po = ProviderOptions::new();
        po.insert(
            "google".into(),
            json!({
                "responseModalities":["TEXT","IMAGE"],
                "thinkingConfig":{"thinkingBudget":1024,"includeThoughts":true},
                "structuredOutputs":false,
                "safetySettings":[{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_LOW_AND_ABOVE"}],
                "audioTimestamp":true,
                "mediaResolution":"MEDIA_RESOLUTION_HIGH",
                "serviceTier":"flex"
            })
            .as_object().unwrap().clone(),
        );
        let r = parse(Some(&po), &["google"]).unwrap().unwrap();
        assert_eq!(r.response_modalities.as_ref().unwrap().len(), 2);
        assert_eq!(
            r.thinking_config.as_ref().unwrap().thinking_budget,
            Some(1024)
        );
        assert_eq!(r.structured_outputs, Some(false));
        assert_eq!(r.audio_timestamp, Some(true));
        assert_eq!(r.service_tier.as_deref(), Some("flex"));
    }

    #[test]
    fn skips_when_no_key() {
        let mut po = ProviderOptions::new();
        po.insert(
            "openai".into(),
            json!({"foo":"bar"}).as_object().unwrap().clone(),
        );
        let r = parse(Some(&po), &["google"]).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn type_validation_error() {
        let mut po = ProviderOptions::new();
        po.insert(
            "google".into(),
            json!({"thinkingConfig":42}).as_object().unwrap().clone(),
        );
        let err = parse(Some(&po), &["google"]).unwrap_err();
        assert!(
            format!("{err}").to_lowercase().contains("validation")
                || format!("{err}").to_lowercase().contains("invalid")
        );
    }
}
