//! `Anthropic` `stop_reason` string -> normalized [`FinishReason`].
//!
//! Mirrors `map-anthropic-stop-reason.ts`.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::language_model::{FinishReason, FinishReasonKind};

/// Map an `Anthropic` `stop_reason` string to a unified [`FinishReason`].
///
/// Defaults to the regular mapping. Call [`map_with_json_tool`] when the
/// jsonResponseTool fallback is active so a synthesized `json` tool call
/// reports `stop` instead of `tool-calls`.
pub(crate) fn map(raw: Option<&str>) -> FinishReason {
    map_with_json_tool(raw, false)
}

/// Variant of [`map`] aware of the jsonResponseTool fallback.
///
/// When `is_json_response_from_tool` is true and the wire reports
/// `tool_use`, the unified reason flips to [`FinishReasonKind::Stop`] —
/// mirrors `map-anthropic-stop-reason.ts:14`'s
/// `isJsonResponseFromTool ? 'stop' : 'tool-calls'` branch.
pub(crate) fn map_with_json_tool(
    raw: Option<&str>,
    is_json_response_from_tool: bool,
) -> FinishReason {
    let kind = match raw {
        Some("end_turn" | "stop_sequence" | "pause_turn") => FinishReasonKind::Stop,
        Some("max_tokens" | "model_context_window_exceeded") => FinishReasonKind::Length,
        Some("tool_use") => {
            if is_json_response_from_tool {
                FinishReasonKind::Stop
            } else {
                FinishReasonKind::ToolCalls
            }
        }
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
