//! xAI Responses status string -> normalized [`FinishReason`].
//!
//! Mirrors `map-xai-responses-finish-reason.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{FinishReason, FinishReasonKind};

/// Map an xAI Responses `status` / `incomplete_details.reason` string to a
/// unified [`FinishReason`].
pub(crate) fn map(raw: Option<&str>) -> FinishReason {
    let kind = match raw {
        Some("stop" | "completed") => FinishReasonKind::Stop,
        Some("length" | "max_output_tokens") => FinishReasonKind::Length,
        Some("tool_calls" | "function_call") => FinishReasonKind::ToolCalls,
        Some("content_filter") => FinishReasonKind::ContentFilter,
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
        assert_eq!(map(Some("completed")).unified, FinishReasonKind::Stop);
        assert_eq!(map(Some("length")).unified, FinishReasonKind::Length);
        assert_eq!(
            map(Some("max_output_tokens")).unified,
            FinishReasonKind::Length
        );
        assert_eq!(map(Some("tool_calls")).unified, FinishReasonKind::ToolCalls);
        assert_eq!(
            map(Some("function_call")).unified,
            FinishReasonKind::ToolCalls
        );
        assert_eq!(
            map(Some("content_filter")).unified,
            FinishReasonKind::ContentFilter
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
