//! Cohere finish-reason string -> normalized [`FinishReason`].
//!
//! Mirrors `map-cohere-finish-reason.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{FinishReason, FinishReasonKind};

/// Map a Cohere `finish_reason` to a unified [`FinishReason`].
pub(crate) fn map(raw: Option<&str>) -> FinishReason {
    let kind = match raw {
        Some("COMPLETE" | "STOP_SEQUENCE") => FinishReasonKind::Stop,
        Some("MAX_TOKENS") => FinishReasonKind::Length,
        Some("ERROR") => FinishReasonKind::Error,
        Some("TOOL_CALL") => FinishReasonKind::ToolCalls,
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
        assert_eq!(map(Some("COMPLETE")).unified, FinishReasonKind::Stop);
        assert_eq!(map(Some("STOP_SEQUENCE")).unified, FinishReasonKind::Stop);
        assert_eq!(map(Some("MAX_TOKENS")).unified, FinishReasonKind::Length);
        assert_eq!(map(Some("ERROR")).unified, FinishReasonKind::Error);
        assert_eq!(map(Some("TOOL_CALL")).unified, FinishReasonKind::ToolCalls);
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
