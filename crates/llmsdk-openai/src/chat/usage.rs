//! `OpenAI` usage object -> normalized [`Usage`].
//!
//! Mirrors `convert-openai-chat-usage.ts`. Differences:
//!
//! - We keep the raw object exactly; ai-sdk only keeps the typed sub-fields.
//! - `no_cache` is `total - cache_read` when both are known, else `None`
//!   (ai-sdk computes a zero, which we treat as "unknown" for clearer
//!   downstream display).
// Rust guideline compliant 2026-02-21

use llmsdk_provider::language_model::{InputTokenUsage, OutputTokenUsage, Usage};

use super::wire::WireUsage;

/// Convert wire usage into the normalized type.
pub(crate) fn convert(usage: Option<&WireUsage>) -> Usage {
    let Some(usage) = usage else {
        return Usage::default();
    };

    let total_in = usage.prompt_tokens;
    let total_out = usage.completion_tokens;
    let cache_read = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|d| d.cached_tokens);
    let reasoning = usage
        .completion_tokens_details
        .as_ref()
        .and_then(|d| d.reasoning_tokens);

    let no_cache = match (total_in, cache_read) {
        (Some(t), Some(c)) => Some(t.saturating_sub(c)),
        _ => None,
    };
    let text = match (total_out, reasoning) {
        (Some(t), Some(r)) => Some(t.saturating_sub(r)),
        (Some(t), None) => Some(t),
        _ => None,
    };

    let raw = serde_json::to_value(usage)
        .ok()
        .and_then(|v| v.as_object().cloned());

    Usage {
        input_tokens: InputTokenUsage {
            total: total_in,
            no_cache,
            cache_read,
            cache_write: None,
        },
        output_tokens: OutputTokenUsage {
            total: total_out,
            text,
            reasoning,
        },
        raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::wire::{WireCompletionTokensDetails, WirePromptTokensDetails};

    #[test]
    fn none_yields_default() {
        let u = convert(None);
        assert!(u.input_tokens.total.is_none());
        assert!(u.output_tokens.total.is_none());
    }

    #[test]
    fn computes_no_cache_and_text() {
        let wire = WireUsage {
            prompt_tokens: Some(100),
            completion_tokens: Some(40),
            total_tokens: Some(140),
            prompt_tokens_details: Some(WirePromptTokensDetails {
                cached_tokens: Some(30),
            }),
            completion_tokens_details: Some(WireCompletionTokensDetails {
                reasoning_tokens: Some(10),
            }),
        };
        let u = convert(Some(&wire));
        assert_eq!(u.input_tokens.total, Some(100));
        assert_eq!(u.input_tokens.cache_read, Some(30));
        assert_eq!(u.input_tokens.no_cache, Some(70));
        assert_eq!(u.output_tokens.total, Some(40));
        assert_eq!(u.output_tokens.reasoning, Some(10));
        assert_eq!(u.output_tokens.text, Some(30));
    }

    #[test]
    fn falls_back_when_subfields_missing() {
        let wire = WireUsage {
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            total_tokens: Some(15),
            prompt_tokens_details: None,
            completion_tokens_details: None,
        };
        let u = convert(Some(&wire));
        assert_eq!(u.input_tokens.total, Some(10));
        assert!(u.input_tokens.cache_read.is_none());
        assert!(u.input_tokens.no_cache.is_none());
        assert_eq!(u.output_tokens.text, Some(5));
        assert!(u.output_tokens.reasoning.is_none());
    }
}
