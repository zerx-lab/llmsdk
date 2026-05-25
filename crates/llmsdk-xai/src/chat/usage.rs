//! xAI usage object -> normalized [`Usage`].
//!
//! Mirrors `convert-xai-chat-usage.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{InputTokenUsage, OutputTokenUsage, Usage};

use super::wire::WireUsage;

/// Convert wire usage into the normalized [`Usage`] shape.
///
/// xAI returns `prompt_tokens` that may or may not include cached tokens. We
/// follow ai-sdk's heuristic: if `cached_tokens <= prompt_tokens`, treat
/// `prompt_tokens` as the inclusive total; otherwise add cached on top.
pub(crate) fn convert(usage: &WireUsage) -> Usage {
    let prompt = usage.prompt_tokens.unwrap_or(0);
    let completion = usage.completion_tokens.unwrap_or(0);
    let cache_read = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|d| d.cached_tokens)
        .unwrap_or(0);
    let reasoning = usage
        .completion_tokens_details
        .as_ref()
        .and_then(|d| d.reasoning_tokens)
        .unwrap_or(0);

    let prompt_includes_cached = cache_read <= prompt;
    let total_input = if prompt_includes_cached {
        prompt
    } else {
        prompt.saturating_add(cache_read)
    };
    let no_cache_input = if prompt_includes_cached {
        prompt.saturating_sub(cache_read)
    } else {
        prompt
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
            total: Some(completion.saturating_add(reasoning)),
            text: Some(completion),
            reasoning: Some(reasoning),
        },
        raw,
    }
}

/// Zero-usage placeholder used when xAI returns no `usage` block.
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
    use crate::chat::wire::{WireCompletionTokensDetails, WirePromptTokensDetails};

    #[test]
    fn prompt_tokens_includes_cached_branch() {
        let wire = WireUsage {
            prompt_tokens: Some(100),
            completion_tokens: Some(40),
            total_tokens: Some(140),
            prompt_tokens_details: Some(WirePromptTokensDetails {
                cached_tokens: Some(30),
                ..Default::default()
            }),
            completion_tokens_details: Some(WireCompletionTokensDetails {
                reasoning_tokens: Some(10),
                ..Default::default()
            }),
        };
        let u = convert(&wire);
        assert_eq!(u.input_tokens.total, Some(100));
        assert_eq!(u.input_tokens.no_cache, Some(70));
        assert_eq!(u.input_tokens.cache_read, Some(30));
        assert_eq!(u.output_tokens.total, Some(50));
        assert_eq!(u.output_tokens.text, Some(40));
        assert_eq!(u.output_tokens.reasoning, Some(10));
    }

    #[test]
    fn cached_exceeds_prompt_adds_branch() {
        let wire = WireUsage {
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            total_tokens: Some(15),
            prompt_tokens_details: Some(WirePromptTokensDetails {
                cached_tokens: Some(20),
                ..Default::default()
            }),
            completion_tokens_details: None,
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
