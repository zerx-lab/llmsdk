//! Tool-call id normalization for Mistral models on Bedrock.
//!
//! Mirrors `normalize-tool-call-id.ts`. Mistral models require tool-call ids
//! matching `^[a-zA-Z0-9]{9}$`; Bedrock's native ids do not, so we strip
//! non-alphanumerics and take the first 9 characters.
// Rust guideline compliant 2026-05-25

/// `true` when the model id belongs to the Mistral family on Bedrock.
///
/// Mirrors `isMistralModel` upstream: matches both `mistral.*` and the
/// region-prefixed `us.mistral.*` ids.
#[must_use]
pub(crate) fn is_mistral_model(model_id: &str) -> bool {
    model_id.contains("mistral.")
}

/// Normalize a tool-call id for the Mistral family.
///
/// For non-Mistral models the id is returned unchanged.
#[must_use]
pub(crate) fn normalize_tool_call_id(tool_call_id: &str, is_mistral: bool) -> String {
    if !is_mistral {
        return tool_call_id.to_owned();
    }
    tool_call_id
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .take(9)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_mistral_matches_dotted_prefix() {
        assert!(is_mistral_model("mistral.mistral-7b-instruct-v0:2"));
        assert!(is_mistral_model("us.mistral.pixtral-large-2502-v1:0"));
        assert!(!is_mistral_model(
            "anthropic.claude-3-5-haiku-20241022-v1:0"
        ));
        assert!(!is_mistral_model("amazon.nova-pro-v1:0"));
    }

    #[test]
    fn normalize_strips_and_truncates_for_mistral() {
        let id = normalize_tool_call_id("tooluse_bpe71yCfRu2b5i-nKGDr5g", true);
        assert_eq!(id.len(), 9);
        assert!(id.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn normalize_returns_input_for_non_mistral() {
        let original = "tooluse_abcdef_12345";
        assert_eq!(normalize_tool_call_id(original, false), original);
    }

    #[test]
    fn normalize_handles_short_input() {
        // shorter than 9 alphanumeric chars after filtering → returns what it
        // could collect (Bedrock still accepts the short id; defensive only)
        let id = normalize_tool_call_id("abc-def", true);
        assert_eq!(id, "abcdef");
    }
}
