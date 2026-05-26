//! `OpenAI` usage object -> normalized [`Usage`].
//!
//! Mirrors `convert-openai-chat-usage.ts:18-58`. Upstream defaults each
//! missing wire field to `0` via `?? 0` before subtracting, so `no_cache` and
//! `text` are derived whenever `total_in` / `total_out` are known regardless
//! of whether the cache/reasoning sub-fields were sent. We keep `total_in` /
//! `total_out` themselves as `Option` because they carry "wire present?"
//! information used elsewhere (`raw` keeps the original object intact, which
//! upstream omits in favour of typed-only output).
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

    // Mirrors upstream `?? 0` fallback: when `prompt_tokens` is present but
    // `cached_tokens` is missing, treat cached as zero so `no_cache == total`.
    let no_cache = total_in.map(|t| t.saturating_sub(cache_read.unwrap_or(0)));
    let text = total_out.map(|t| t.saturating_sub(reasoning.unwrap_or(0)));

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
                accepted_prediction_tokens: None,
                rejected_prediction_tokens: None,
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
        // Upstream `?? 0` semantics: missing sub-fields default to 0, so
        // `no_cache == total_in` and `text == total_out` when the sub-field is absent.
        assert_eq!(u.input_tokens.total, Some(10));
        assert!(u.input_tokens.cache_read.is_none());
        assert_eq!(u.input_tokens.no_cache, Some(10));
        assert_eq!(u.output_tokens.text, Some(5));
        assert!(u.output_tokens.reasoning.is_none());
    }
}
