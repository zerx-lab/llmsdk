//! xAI Responses usage object -> normalized [`Usage`].
//!
//! Mirrors `convert-xai-responses-usage.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{InputTokenUsage, OutputTokenUsage, Usage};

use super::wire::WireUsage;

/// Convert a wire usage object into the normalized [`Usage`] shape.
///
/// xAI may report `input_tokens` either inclusive of cached tokens or
/// exclusive of them. We follow the upstream heuristic: when
/// `cached_tokens <= input_tokens`, treat the reported total as inclusive;
/// otherwise add cached on top.
pub(crate) fn convert(usage: &WireUsage) -> Usage {
    let input = usage.input_tokens.unwrap_or(0);
    let output = usage.output_tokens.unwrap_or(0);
    let cache_read = usage
        .input_tokens_details
        .as_ref()
        .and_then(|d| d.cached_tokens)
        .unwrap_or(0);
    let reasoning = usage
        .output_tokens_details
        .as_ref()
        .and_then(|d| d.reasoning_tokens)
        .unwrap_or(0);

    let input_includes_cached = cache_read <= input;
    let total_input = if input_includes_cached {
        input
    } else {
        input.saturating_add(cache_read)
    };
    let no_cache_input = if input_includes_cached {
        input.saturating_sub(cache_read)
    } else {
        input
    };

    let raw = serde_json::to_value(usage)
        .ok()
        .and_then(|v| v.as_object().cloned());

    Usage {
        input_tokens: InputTokenUsage {
            total: Some(total_input),
            no_cache: Some(no_cache_input),
            cache_read: Some(cache_read),
            cache_write: None,
        },
        output_tokens: OutputTokenUsage {
            total: Some(output),
            text: Some(output.saturating_sub(reasoning)),
            reasoning: Some(reasoning),
        },
        raw,
    }
}

/// Zero-usage placeholder used when xAI omits `usage`.
pub(crate) fn zero() -> Usage {
    Usage {
        input_tokens: InputTokenUsage {
            total: Some(0),
            no_cache: Some(0),
            cache_read: Some(0),
            cache_write: None,
        },
        output_tokens: OutputTokenUsage {
            total: Some(0),
            text: Some(0),
            reasoning: Some(0),
        },
        raw: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses::wire::{WireInputTokensDetails, WireOutputTokensDetails};

    #[test]
    fn input_includes_cached_branch() {
        let wire = WireUsage {
            input_tokens: Some(100),
            output_tokens: Some(40),
            input_tokens_details: Some(WireInputTokensDetails {
                cached_tokens: Some(30),
            }),
            output_tokens_details: Some(WireOutputTokensDetails {
                reasoning_tokens: Some(10),
            }),
            ..Default::default()
        };
        let u = convert(&wire);
        assert_eq!(u.input_tokens.total, Some(100));
        assert_eq!(u.input_tokens.no_cache, Some(70));
        assert_eq!(u.input_tokens.cache_read, Some(30));
        assert_eq!(u.output_tokens.total, Some(40));
        assert_eq!(u.output_tokens.text, Some(30));
        assert_eq!(u.output_tokens.reasoning, Some(10));
    }

    #[test]
    fn cached_exceeds_input_adds_branch() {
        let wire = WireUsage {
            input_tokens: Some(10),
            output_tokens: Some(5),
            input_tokens_details: Some(WireInputTokensDetails {
                cached_tokens: Some(20),
            }),
            ..Default::default()
        };
        let u = convert(&wire);
        assert_eq!(u.input_tokens.total, Some(30));
        assert_eq!(u.input_tokens.no_cache, Some(10));
        assert_eq!(u.input_tokens.cache_read, Some(20));
    }

    #[test]
    fn zero_placeholder_has_all_zeros() {
        let u = zero();
        assert_eq!(u.input_tokens.total, Some(0));
        assert_eq!(u.output_tokens.total, Some(0));
    }
}
