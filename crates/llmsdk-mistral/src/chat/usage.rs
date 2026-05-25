//! Mistral usage object -> normalized [`Usage`].
//!
//! Mirrors `convert-mistral-usage.ts`. Mistral reports cached prompt tokens
//! in three different shapes depending on API version; we read whichever
//! is populated. `prompt_tokens` already includes cached tokens, so
//! `noCache = prompt - cached`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{InputTokenUsage, OutputTokenUsage, Usage};

use super::wire::WireUsage;

/// Convert wire usage into the normalized [`Usage`] shape.
pub(crate) fn convert(usage: &WireUsage) -> Usage {
    let prompt = usage.prompt_tokens.unwrap_or(0);
    let completion = usage.completion_tokens.unwrap_or(0);
    let cache_read = usage
        .num_cached_tokens
        .or_else(|| {
            usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens)
        })
        .or_else(|| {
            usage
                .prompt_token_details
                .as_ref()
                .and_then(|d| d.cached_tokens)
        })
        .unwrap_or(0);

    let raw = serde_json::to_value(usage)
        .ok()
        .and_then(|v| v.as_object().cloned());

    Usage {
        input_tokens: InputTokenUsage {
            total: Some(prompt),
            no_cache: Some(prompt.saturating_sub(cache_read)),
            cache_read: if cache_read > 0 {
                Some(cache_read)
            } else {
                None
            },
            cache_write: None,
        },
        output_tokens: OutputTokenUsage {
            total: Some(completion),
            text: Some(completion),
            reasoning: None,
        },
        raw,
    }
}

/// Zero-usage placeholder used when Mistral returns no `usage` block.
pub(crate) fn zero() -> Usage {
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
    use super::*;
    use crate::chat::wire::WireCachedTokens;

    #[test]
    fn num_cached_tokens_branch() {
        let wire = WireUsage {
            prompt_tokens: Some(100),
            completion_tokens: Some(40),
            total_tokens: Some(140),
            num_cached_tokens: Some(30),
            ..Default::default()
        };
        let u = convert(&wire);
        assert_eq!(u.input_tokens.total, Some(100));
        assert_eq!(u.input_tokens.no_cache, Some(70));
        assert_eq!(u.input_tokens.cache_read, Some(30));
        assert_eq!(u.output_tokens.total, Some(40));
        assert_eq!(u.output_tokens.text, Some(40));
    }

    #[test]
    fn prompt_tokens_details_branch() {
        let wire = WireUsage {
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            total_tokens: Some(15),
            prompt_tokens_details: Some(WireCachedTokens {
                cached_tokens: Some(3),
            }),
            ..Default::default()
        };
        let u = convert(&wire);
        assert_eq!(u.input_tokens.cache_read, Some(3));
        assert_eq!(u.input_tokens.no_cache, Some(7));
    }

    #[test]
    fn prompt_token_details_alias_branch() {
        let wire = WireUsage {
            prompt_tokens: Some(8),
            completion_tokens: Some(2),
            total_tokens: Some(10),
            prompt_token_details: Some(WireCachedTokens {
                cached_tokens: Some(4),
            }),
            ..Default::default()
        };
        let u = convert(&wire);
        assert_eq!(u.input_tokens.cache_read, Some(4));
    }

    #[test]
    fn missing_cache_yields_none() {
        let wire = WireUsage {
            prompt_tokens: Some(5),
            completion_tokens: Some(1),
            total_tokens: Some(6),
            ..Default::default()
        };
        let u = convert(&wire);
        assert_eq!(u.input_tokens.cache_read, None);
    }

    #[test]
    fn zero_placeholder_has_all_none() {
        let u = zero();
        assert_eq!(u.input_tokens.total, None);
        assert_eq!(u.output_tokens.total, None);
    }
}
