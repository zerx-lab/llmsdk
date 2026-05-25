//! Convert OpenAI Responses `usage` block → llmsdk [`Usage`].
//!
//! Mirrors `@ai-sdk/openai/src/responses/convert-openai-responses-usage.ts`.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::language_model::{InputTokenUsage, OutputTokenUsage, Usage};
use serde::{Deserialize, Serialize};

/// Raw `usage` block returned by the Responses API.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct ResponsesUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

/// `input_tokens_details` block.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct InputTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

/// `output_tokens_details` block.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct OutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
}

/// Convert. Returns a `Usage` with `raw` populated from the original JSON.
#[must_use]
pub fn convert_usage(usage: Option<&ResponsesUsage>) -> Usage {
    let Some(usage) = usage else {
        return Usage::default();
    };
    let cached = usage
        .input_tokens_details
        .as_ref()
        .and_then(|d| d.cached_tokens)
        .unwrap_or(0);
    let reasoning = usage
        .output_tokens_details
        .as_ref()
        .and_then(|d| d.reasoning_tokens)
        .unwrap_or(0);
    let input_total = usage.input_tokens;
    let output_total = usage.output_tokens;

    let raw = serde_json::to_value(usage)
        .ok()
        .and_then(|v| v.as_object().cloned());

    Usage {
        input_tokens: InputTokenUsage {
            total: Some(input_total),
            no_cache: Some(input_total.saturating_sub(cached)),
            cache_read: Some(cached),
            cache_write: None,
        },
        output_tokens: OutputTokenUsage {
            total: Some(output_total),
            text: Some(output_total.saturating_sub(reasoning)),
            reasoning: Some(reasoning),
        },
        raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_usage_is_default() {
        assert_eq!(convert_usage(None), Usage::default());
    }

    #[test]
    fn splits_cached_and_reasoning() {
        let u = ResponsesUsage {
            input_tokens: 100,
            output_tokens: 50,
            input_tokens_details: Some(InputTokensDetails {
                cached_tokens: Some(30),
            }),
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: Some(20),
            }),
        };
        let out = convert_usage(Some(&u));
        assert_eq!(out.input_tokens.total, Some(100));
        assert_eq!(out.input_tokens.cache_read, Some(30));
        assert_eq!(out.input_tokens.no_cache, Some(70));
        assert_eq!(out.output_tokens.total, Some(50));
        assert_eq!(out.output_tokens.reasoning, Some(20));
        assert_eq!(out.output_tokens.text, Some(30));
        assert!(out.raw.is_some());
    }
}
