//! Gemini `usageMetadata` → unified [`Usage`] mapping.
//!
//! Mirrors `@ai-sdk/google/src/convert-google-usage.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{InputTokenUsage, OutputTokenUsage, Usage};

use super::wire::WireUsage;

/// Convert Gemini `usageMetadata` to the unified [`Usage`] envelope.
///
/// `cachedContentTokenCount` populates `cache_read`; `thoughtsTokenCount`
/// populates `output_tokens.reasoning` and is rolled into `output_tokens.total`.
#[must_use]
pub(crate) fn convert_usage(u: Option<&WireUsage>) -> Usage {
    let Some(u) = u else {
        return Usage::default();
    };

    let prompt = u.prompt_token_count.unwrap_or(0);
    let candidates = u.candidates_token_count.unwrap_or(0);
    let cached = u.cached_content_token_count.unwrap_or(0);
    let thoughts = u.thoughts_token_count.unwrap_or(0);

    // Preserve the provider-native usage payload verbatim
    // (convert-google-usage.ts:57 `raw: usage`). Callers rely on it for
    // trafficType / serviceTier / tokensDetails breakdown that the
    // normalized counters do not surface.
    let raw = serde_json::to_value(u)
        .ok()
        .and_then(|v| v.as_object().cloned());

    Usage {
        input_tokens: InputTokenUsage {
            total: Some(prompt),
            no_cache: Some(prompt.saturating_sub(cached)),
            cache_read: Some(cached),
            cache_write: None,
        },
        output_tokens: OutputTokenUsage {
            total: Some(candidates + thoughts),
            text: Some(candidates),
            reasoning: Some(thoughts),
        },
        raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_usage() {
        let r = convert_usage(None);
        assert_eq!(r.input_tokens.total, None);
        assert_eq!(r.output_tokens.total, None);
    }

    #[test]
    fn populated_usage() {
        let u = WireUsage {
            prompt_token_count: Some(100),
            candidates_token_count: Some(50),
            cached_content_token_count: Some(30),
            thoughts_token_count: Some(20),
            ..Default::default()
        };
        let r = convert_usage(Some(&u));
        assert_eq!(r.input_tokens.total, Some(100));
        assert_eq!(r.input_tokens.no_cache, Some(70));
        assert_eq!(r.input_tokens.cache_read, Some(30));
        assert_eq!(r.output_tokens.total, Some(70));
        assert_eq!(r.output_tokens.text, Some(50));
        assert_eq!(r.output_tokens.reasoning, Some(20));
    }

    #[test]
    fn raw_passthrough_contains_native_counters() {
        let u = WireUsage {
            prompt_token_count: Some(100),
            candidates_token_count: Some(50),
            cached_content_token_count: Some(30),
            thoughts_token_count: Some(20),
            ..Default::default()
        };
        let r = convert_usage(Some(&u));
        let raw = r.raw.expect("raw usage should be populated");
        assert_eq!(
            raw.get("promptTokenCount").and_then(|v| v.as_u64()),
            Some(100)
        );
        assert_eq!(
            raw.get("thoughtsTokenCount").and_then(|v| v.as_u64()),
            Some(20)
        );
    }
}
