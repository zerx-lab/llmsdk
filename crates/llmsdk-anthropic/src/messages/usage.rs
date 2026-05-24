//! `Anthropic` usage -> normalized [`Usage`].
//!
//! Mirrors `convert-anthropic-usage.ts` (simplified: no `iterations`,
//! which is for compaction / advisor — deferred to a later milestone).
// Rust guideline compliant 2026-02-21

use llmsdk_provider::language_model::{InputTokenUsage, OutputTokenUsage, Usage};

use super::wire::ResponseUsage;

/// Convert wire usage into the normalized type.
///
/// Computes `total = input + cache_creation + cache_read` like ai-sdk so
/// the top-level `total` reflects what the user is billed for.
pub(crate) fn convert(usage: &ResponseUsage) -> Usage {
    let cache_create = usage.cache_creation_input_tokens.unwrap_or(0);
    let cache_read = usage.cache_read_input_tokens.unwrap_or(0);
    let total_in = usage.input_tokens + cache_create + cache_read;

    let raw = serde_json::to_value(usage)
        .ok()
        .and_then(|v| v.as_object().cloned());

    Usage {
        input_tokens: InputTokenUsage {
            total: Some(total_in),
            no_cache: Some(usage.input_tokens),
            cache_read: Some(cache_read),
            cache_write: Some(cache_create),
        },
        output_tokens: OutputTokenUsage {
            total: Some(usage.output_tokens),
            text: None,
            reasoning: None,
        },
        raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_cache_into_total() {
        let wire = ResponseUsage {
            input_tokens: 100,
            output_tokens: 42,
            cache_creation_input_tokens: Some(10),
            cache_read_input_tokens: Some(20),
        };
        let u = convert(&wire);
        assert_eq!(u.input_tokens.total, Some(130));
        assert_eq!(u.input_tokens.no_cache, Some(100));
        assert_eq!(u.input_tokens.cache_read, Some(20));
        assert_eq!(u.input_tokens.cache_write, Some(10));
        assert_eq!(u.output_tokens.total, Some(42));
    }

    #[test]
    fn missing_cache_treated_as_zero() {
        let wire = ResponseUsage {
            input_tokens: 5,
            output_tokens: 3,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        };
        let u = convert(&wire);
        assert_eq!(u.input_tokens.total, Some(5));
        assert_eq!(u.input_tokens.cache_read, Some(0));
        assert_eq!(u.input_tokens.cache_write, Some(0));
    }
}
