//! Per-model capability flags driving request shape.
//!
//! Mirrors `openai-language-model-capabilities.ts`. Used to detect
//! reasoning / search-preview models and to drive parameter stripping.
//!
//! # Allow-list policy
//!
//! Identifying reasoning models from the model id is intentionally an
//! allow-list (matching ai-sdk). Custom fine-tunes and third-party
//! deployments with non-matching ids will be treated as plain chat models.
// Rust guideline compliant 2026-02-21

/// Capability flags derived from a model id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Capabilities {
    /// True for `o1*` / `o3*` / `o4-mini*` / `gpt-5*` (except `gpt-5-chat*`).
    pub is_reasoning_model: bool,
    /// True for the `gpt-4o(-mini)?-search-preview*` family.
    pub is_search_preview_model: bool,
    /// True for the `gpt-5.1+` reasoning families that permit
    /// `temperature` / `top_p` / `logprobs` when `reasoning_effort == none`.
    pub supports_non_reasoning_parameters: bool,
}

impl Capabilities {
    /// Compute capability flags from a model id.
    pub(crate) fn detect(model_id: &str) -> Self {
        let is_reasoning_model = model_id.starts_with("o1")
            || model_id.starts_with("o3")
            || model_id.starts_with("o4-mini")
            || (model_id.starts_with("gpt-5") && !model_id.starts_with("gpt-5-chat"));

        let is_search_preview_model = model_id.starts_with("gpt-4o-search-preview")
            || model_id.starts_with("gpt-4o-mini-search-preview");

        let supports_non_reasoning_parameters = model_id.starts_with("gpt-5.1")
            || model_id.starts_with("gpt-5.2")
            || model_id.starts_with("gpt-5.3")
            || model_id.starts_with("gpt-5.4")
            || model_id.starts_with("gpt-5.5");

        Self {
            is_reasoning_model,
            is_search_preview_model,
            supports_non_reasoning_parameters,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_models_detected() {
        assert!(Capabilities::detect("o1").is_reasoning_model);
        assert!(Capabilities::detect("o1-preview").is_reasoning_model);
        assert!(Capabilities::detect("o3-mini").is_reasoning_model);
        assert!(Capabilities::detect("o4-mini-2025-04-16").is_reasoning_model);
        assert!(Capabilities::detect("gpt-5").is_reasoning_model);
        assert!(Capabilities::detect("gpt-5-nano").is_reasoning_model);
        assert!(Capabilities::detect("gpt-5.1").is_reasoning_model);
    }

    #[test]
    fn gpt_5_chat_is_not_reasoning() {
        assert!(!Capabilities::detect("gpt-5-chat-latest").is_reasoning_model);
    }

    #[test]
    fn plain_chat_models_are_not_reasoning() {
        assert!(!Capabilities::detect("gpt-4o-mini").is_reasoning_model);
        assert!(!Capabilities::detect("gpt-4.1-nano").is_reasoning_model);
        assert!(!Capabilities::detect("gpt-3.5-turbo").is_reasoning_model);
    }

    #[test]
    fn search_preview_models_detected() {
        assert!(Capabilities::detect("gpt-4o-search-preview").is_search_preview_model);
        assert!(
            Capabilities::detect("gpt-4o-mini-search-preview-2025-01-01").is_search_preview_model
        );
        assert!(!Capabilities::detect("gpt-4o-mini").is_search_preview_model);
    }

    #[test]
    fn gpt_5_1_supports_non_reasoning_parameters_when_none() {
        assert!(Capabilities::detect("gpt-5.1").supports_non_reasoning_parameters);
        assert!(Capabilities::detect("gpt-5.4-nano").supports_non_reasoning_parameters);
        assert!(!Capabilities::detect("gpt-5").supports_non_reasoning_parameters);
        assert!(!Capabilities::detect("o3").supports_non_reasoning_parameters);
    }
}
