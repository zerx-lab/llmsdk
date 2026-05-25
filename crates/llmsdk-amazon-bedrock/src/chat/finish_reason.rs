//! Bedrock stopReason → llmsdk [`FinishReason`] mapping.
//!
//! Mirrors `map-amazon-bedrock-finish-reason.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{FinishReason, FinishReasonKind};

/// Map a Bedrock `stopReason` string to a unified [`FinishReason`].
///
/// The `is_json_response_from_tool` flag overrides the `tool_use` case when
/// the model used the synthetic JSON-tool to emit structured output —
/// upstream surfaces that as plain `Stop`.
pub(crate) fn map_finish_reason(
    raw: Option<&str>,
    is_json_response_from_tool: bool,
) -> FinishReason {
    let raw_owned = raw.map(str::to_owned);
    let kind = match raw {
        Some("stop_sequence" | "end_turn" | "stop") => FinishReasonKind::Stop,
        Some("max_tokens" | "length") => FinishReasonKind::Length,
        Some("content_filtered" | "content-filter" | "guardrail_intervened") => {
            FinishReasonKind::ContentFilter
        }
        Some("tool_use" | "tool-calls") => {
            if is_json_response_from_tool {
                FinishReasonKind::Stop
            } else {
                FinishReasonKind::ToolCalls
            }
        }
        _ => FinishReasonKind::Other,
    };
    FinishReason {
        unified: kind,
        raw: raw_owned,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn end_turn_is_stop() {
        let f = map_finish_reason(Some("end_turn"), false);
        assert_eq!(f.unified, FinishReasonKind::Stop);
        assert_eq!(f.raw.as_deref(), Some("end_turn"));
    }

    #[test]
    fn max_tokens_is_length() {
        assert_eq!(
            map_finish_reason(Some("max_tokens"), false).unified,
            FinishReasonKind::Length
        );
    }

    #[test]
    fn guardrail_intervened_is_content_filter() {
        assert_eq!(
            map_finish_reason(Some("guardrail_intervened"), false).unified,
            FinishReasonKind::ContentFilter
        );
    }

    #[test]
    fn tool_use_with_json_overrides_to_stop() {
        assert_eq!(
            map_finish_reason(Some("tool_use"), true).unified,
            FinishReasonKind::Stop
        );
    }

    #[test]
    fn unknown_falls_back_to_other() {
        assert_eq!(
            map_finish_reason(Some("WIBBLE"), false).unified,
            FinishReasonKind::Other
        );
    }
}
