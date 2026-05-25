//! Gemini finish reason → unified finish reason mapping.
//!
//! Mirrors `@ai-sdk/google/src/map-google-finish-reason.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::FinishReasonKind;

/// Map Gemini's raw `finishReason` to a unified [`FinishReasonKind`].
///
/// `STOP` flips to `ToolCalls` when the response also contains
/// client-executed tool calls (matches ai-sdk).
#[must_use]
pub(crate) fn map_finish_reason(raw: Option<&str>, has_tool_calls: bool) -> FinishReasonKind {
    match raw {
        Some("STOP") => {
            if has_tool_calls {
                FinishReasonKind::ToolCalls
            } else {
                FinishReasonKind::Stop
            }
        }
        Some("MAX_TOKENS") => FinishReasonKind::Length,
        Some("IMAGE_SAFETY")
        | Some("RECITATION")
        | Some("SAFETY")
        | Some("BLOCKLIST")
        | Some("PROHIBITED_CONTENT")
        | Some("SPII") => FinishReasonKind::ContentFilter,
        Some("MALFORMED_FUNCTION_CALL") => FinishReasonKind::Error,
        _ => FinishReasonKind::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_no_tools() {
        assert_eq!(
            map_finish_reason(Some("STOP"), false),
            FinishReasonKind::Stop
        );
    }

    #[test]
    fn stop_with_tools_flips() {
        assert_eq!(
            map_finish_reason(Some("STOP"), true),
            FinishReasonKind::ToolCalls
        );
    }

    #[test]
    fn safety_to_content_filter() {
        assert_eq!(
            map_finish_reason(Some("SAFETY"), false),
            FinishReasonKind::ContentFilter
        );
    }

    #[test]
    fn unknown_to_other() {
        assert_eq!(
            map_finish_reason(Some("WAT"), false),
            FinishReasonKind::Other
        );
        assert_eq!(map_finish_reason(None, false), FinishReasonKind::Other);
    }
}
