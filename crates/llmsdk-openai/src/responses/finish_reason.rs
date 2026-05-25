//! Maps OpenAI Responses `incomplete_details.reason` → `FinishReason`.
//!
//! Mirrors `@ai-sdk/openai/src/responses/map-openai-responses-finish-reason.ts`.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::language_model::{FinishReason, FinishReasonKind};

/// Apply the upstream finish-reason mapping table.
///
/// `reason` is `incomplete_details.reason` from the Responses API
/// (may be `None`); `has_function_call` flags whether any client-side
/// function call appeared in the output.
#[must_use]
pub fn map_finish_reason(reason: Option<&str>, has_function_call: bool) -> FinishReason {
    let kind = match reason {
        None => {
            if has_function_call {
                FinishReasonKind::ToolCalls
            } else {
                FinishReasonKind::Stop
            }
        }
        Some("max_output_tokens") => FinishReasonKind::Length,
        Some("content_filter") => FinishReasonKind::ContentFilter,
        Some(_) => {
            if has_function_call {
                FinishReasonKind::ToolCalls
            } else {
                FinishReasonKind::Other
            }
        }
    };
    match reason {
        Some(raw) => FinishReason::with_raw(kind, raw),
        None => FinishReason::new(kind),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_when_no_reason_and_no_fn_call() {
        assert_eq!(
            map_finish_reason(None, false).unified,
            FinishReasonKind::Stop
        );
    }

    #[test]
    fn tool_calls_when_no_reason_but_fn_call_present() {
        assert_eq!(
            map_finish_reason(None, true).unified,
            FinishReasonKind::ToolCalls
        );
    }

    #[test]
    fn max_output_tokens_maps_to_length() {
        let r = map_finish_reason(Some("max_output_tokens"), false);
        assert_eq!(r.unified, FinishReasonKind::Length);
        assert_eq!(r.raw.as_deref(), Some("max_output_tokens"));
    }

    #[test]
    fn content_filter_maps_through() {
        let r = map_finish_reason(Some("content_filter"), true);
        assert_eq!(r.unified, FinishReasonKind::ContentFilter);
    }

    #[test]
    fn unknown_reason_with_fn_call_is_tool_calls() {
        assert_eq!(
            map_finish_reason(Some("weird"), true).unified,
            FinishReasonKind::ToolCalls
        );
    }

    #[test]
    fn unknown_reason_without_fn_call_is_other() {
        assert_eq!(
            map_finish_reason(Some("weird"), false).unified,
            FinishReasonKind::Other
        );
    }
}
