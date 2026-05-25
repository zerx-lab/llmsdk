//! Cohere usage tokens -> normalized [`Usage`].
//!
//! Mirrors `convert-cohere-usage.ts`. Cohere reports `usage.tokens` /
//! `usage.billed_units` with `input_tokens` + `output_tokens`. We feed
//! `usage.tokens` (raw) into the unified shape and capture `billed_units`
//! in the `raw` slot.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{InputTokenUsage, OutputTokenUsage, Usage};

use super::wire::WireUsage;

/// Convert a Cohere usage envelope into the normalized [`Usage`] shape.
pub(crate) fn convert(usage: Option<&WireUsage>) -> Usage {
    let Some(envelope) = usage else {
        return empty();
    };

    let tokens = envelope.tokens.unwrap_or_default();
    let raw = serde_json::to_value(envelope)
        .ok()
        .and_then(|v| v.as_object().cloned());

    Usage {
        input_tokens: InputTokenUsage {
            total: tokens.input_tokens,
            no_cache: tokens.input_tokens,
            cache_read: None,
            cache_write: None,
        },
        output_tokens: OutputTokenUsage {
            total: tokens.output_tokens,
            text: tokens.output_tokens,
            reasoning: None,
        },
        raw,
    }
}

/// Empty-usage placeholder used when Cohere returns no `usage` block.
pub(crate) fn empty() -> Usage {
    Usage {
        input_tokens: InputTokenUsage {
            total: None,
            no_cache: None,
            cache_read: None,
            cache_write: None,
        },
        output_tokens: OutputTokenUsage {
            total: None,
            text: None,
            reasoning: None,
        },
        raw: None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::wire::WireUsageTokens;
    use super::*;

    #[test]
    fn empty_when_no_usage() {
        let u = convert(None);
        assert!(u.input_tokens.total.is_none());
        assert!(u.output_tokens.total.is_none());
    }

    #[test]
    fn convert_basic_usage() {
        let wire = WireUsage {
            billed_units: Some(WireUsageTokens {
                input_tokens: Some(8),
                output_tokens: Some(4),
            }),
            tokens: Some(WireUsageTokens {
                input_tokens: Some(8),
                output_tokens: Some(4),
            }),
        };
        let u = convert(Some(&wire));
        assert_eq!(u.input_tokens.total, Some(8));
        assert_eq!(u.input_tokens.no_cache, Some(8));
        assert_eq!(u.output_tokens.total, Some(4));
        assert_eq!(u.output_tokens.text, Some(4));
        assert!(u.raw.is_some());
    }

    #[test]
    fn missing_tokens_yields_nones() {
        let wire = WireUsage {
            billed_units: None,
            tokens: None,
        };
        let u = convert(Some(&wire));
        assert!(u.input_tokens.total.is_none());
        assert!(u.output_tokens.total.is_none());
    }
}
