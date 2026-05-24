//! `OpenAI` finish-reason string -> normalized [`FinishReason`].
//!
//! Mirrors `map-openai-finish-reason.ts`.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::language_model::{FinishReason, FinishReasonKind};

/// Map an `OpenAI` `finish_reason` string to a unified [`FinishReason`].
///
/// Preserves the raw value for telemetry. `None` becomes
/// [`FinishReasonKind::Other`] with `raw = None`.
pub(crate) fn map(raw: Option<&str>) -> FinishReason {
    let kind = match raw {
        Some("stop") => FinishReasonKind::Stop,
        Some("length") => FinishReasonKind::Length,
        Some("content_filter") => FinishReasonKind::ContentFilter,
        Some("function_call" | "tool_calls") => FinishReasonKind::ToolCalls,
        _ => FinishReasonKind::Other,
    };
    FinishReason {
        unified: kind,
        raw: raw.map(str::to_owned),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_known_values() {
        assert_eq!(map(Some("stop")).unified, FinishReasonKind::Stop);
        assert_eq!(map(Some("length")).unified, FinishReasonKind::Length);
        assert_eq!(
            map(Some("content_filter")).unified,
            FinishReasonKind::ContentFilter
        );
        assert_eq!(map(Some("tool_calls")).unified, FinishReasonKind::ToolCalls);
        assert_eq!(
            map(Some("function_call")).unified,
            FinishReasonKind::ToolCalls
        );
    }

    #[test]
    fn unknown_is_other_preserving_raw() {
        let fr = map(Some("weird"));
        assert_eq!(fr.unified, FinishReasonKind::Other);
        assert_eq!(fr.raw.as_deref(), Some("weird"));
    }

    #[test]
    fn none_is_other_with_no_raw() {
        let fr = map(None);
        assert_eq!(fr.unified, FinishReasonKind::Other);
        assert!(fr.raw.is_none());
    }
}
