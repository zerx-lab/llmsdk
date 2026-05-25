//! Convert Bedrock token usage into llmsdk [`Usage`].
//!
//! Mirrors `convert-amazon-bedrock-usage.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{InputTokenUsage, OutputTokenUsage, Usage};

use super::wire::BedrockUsage;

/// Convert a Bedrock usage payload into the unified [`Usage`] type.
///
/// The Bedrock value is preserved verbatim under [`Usage::raw`] when present
/// so callers can inspect provider-specific cache breakdowns.
pub(crate) fn convert_usage(value: Option<BedrockUsage>) -> Usage {
    let Some(u) = value else {
        return Usage::default();
    };
    let input = u.input_tokens.unwrap_or(0);
    let cache_read = u.cache_read_input_tokens.unwrap_or(0);
    let cache_write = u.cache_write_input_tokens.unwrap_or(0);
    let output = u.output_tokens.unwrap_or(0);

    let raw = serde_json::to_value(&u)
        .ok()
        .and_then(|v| v.as_object().cloned());

    Usage {
        input_tokens: InputTokenUsage {
            total: Some(input + cache_read + cache_write),
            no_cache: Some(input),
            cache_read: Some(cache_read),
            cache_write: Some(cache_write),
        },
        output_tokens: OutputTokenUsage {
            total: Some(output),
            text: Some(output),
            reasoning: None,
        },
        raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_usage_yields_default() {
        let usage = convert_usage(None);
        assert!(usage.input_tokens.total.is_none());
    }

    #[test]
    fn input_total_includes_cache_tokens() {
        let u = BedrockUsage {
            input_tokens: Some(100),
            output_tokens: Some(50),
            total_tokens: Some(160),
            cache_read_input_tokens: Some(10),
            cache_write_input_tokens: Some(0),
            cache_details: None,
        };
        let usage = convert_usage(Some(u));
        assert_eq!(usage.input_tokens.total, Some(110));
        assert_eq!(usage.input_tokens.no_cache, Some(100));
        assert_eq!(usage.input_tokens.cache_read, Some(10));
        assert_eq!(usage.output_tokens.total, Some(50));
    }
}
