//! `Anthropic` `stop_reason` string -> normalized [`FinishReason`].
//!
//! Mirrors `map-anthropic-stop-reason.ts`.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::language_model::{FinishReason, FinishReasonKind};

/// Map an `Anthropic` `stop_reason` string to a unified [`FinishReason`].
pub(crate) fn map(raw: Option<&str>) -> FinishReason {
    let kind = match raw {
        Some("end_turn" | "stop_sequence" | "pause_turn") => FinishReasonKind::Stop,
        Some("max_tokens" | "model_context_window_exceeded") => FinishReasonKind::Length,
        Some("tool_use") => FinishReasonKind::ToolCalls,
        Some("refusal") => FinishReasonKind::ContentFilter,
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
        assert_eq!(map(Some("end_turn")).unified, FinishReasonKind::Stop);
        assert_eq!(map(Some("stop_sequence")).unified, FinishReasonKind::Stop);
        assert_eq!(map(Some("max_tokens")).unified, FinishReasonKind::Length);
        assert_eq!(map(Some("tool_use")).unified, FinishReasonKind::ToolCalls);
        assert_eq!(
            map(Some("refusal")).unified,
            FinishReasonKind::ContentFilter
        );
        assert_eq!(map(Some("anything")).unified, FinishReasonKind::Other);
        assert_eq!(map(None).unified, FinishReasonKind::Other);
    }
}
