//! Model-capability lookup table.
//!
//! Mirrors `getModelCapabilities()` in `anthropic-language-model.ts`. Exposes
//! per-model defaults and feature flags so callers can branch without
//! duplicating the table.
// Rust guideline compliant 2026-02-21

/// Capabilities a Claude model exposes via the Messages API.
///
/// Mirrors the upstream `getModelCapabilities()` return shape.
#[allow(
    clippy::struct_excessive_bools,
    reason = "mirrors ai-sdk getModelCapabilities() return shape verbatim"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCapabilities {
    /// Cap for `max_output_tokens`.
    pub max_output_tokens: u32,
    /// Whether structured output (`output_config.format`) is honored.
    pub supports_structured_output: bool,
    /// Whether adaptive thinking (`thinking.type = "adaptive"`) is honored.
    pub supports_adaptive_thinking: bool,
    /// Whether the model ignores `temperature` / `top_p` / `top_k`
    /// (Anthropic strips them for these models).
    pub rejects_sampling_parameters: bool,
    /// Whether `reasoning.effort = "xhigh"` is supported.
    pub supports_xhigh_effort: bool,
    /// Whether the model id matches any of the known families.
    pub is_known_model: bool,
}

/// Return capabilities for `model_id`, falling back to a conservative
/// unknown-model profile.
#[must_use]
pub fn model_capabilities(model_id: &str) -> ModelCapabilities {
    if model_id.contains("claude-opus-4-7") {
        ModelCapabilities {
            max_output_tokens: 128_000,
            supports_structured_output: true,
            supports_adaptive_thinking: true,
            rejects_sampling_parameters: true,
            supports_xhigh_effort: true,
            is_known_model: true,
        }
    } else if model_id.contains("claude-sonnet-4-6") || model_id.contains("claude-opus-4-6") {
        ModelCapabilities {
            max_output_tokens: 128_000,
            supports_structured_output: true,
            supports_adaptive_thinking: true,
            rejects_sampling_parameters: false,
            supports_xhigh_effort: false,
            is_known_model: true,
        }
    } else if model_id.contains("claude-sonnet-4-5")
        || model_id.contains("claude-opus-4-5")
        || model_id.contains("claude-haiku-4-5")
    {
        ModelCapabilities {
            max_output_tokens: 64_000,
            supports_structured_output: true,
            supports_adaptive_thinking: false,
            rejects_sampling_parameters: false,
            supports_xhigh_effort: false,
            is_known_model: true,
        }
    } else if model_id.contains("claude-opus-4-1") {
        ModelCapabilities {
            max_output_tokens: 32_000,
            supports_structured_output: true,
            supports_adaptive_thinking: false,
            rejects_sampling_parameters: false,
            supports_xhigh_effort: false,
            is_known_model: true,
        }
    } else if model_id.contains("claude-sonnet-4-") {
        ModelCapabilities {
            max_output_tokens: 64_000,
            supports_structured_output: false,
            supports_adaptive_thinking: false,
            rejects_sampling_parameters: false,
            supports_xhigh_effort: false,
            is_known_model: true,
        }
    } else if model_id.contains("claude-opus-4-") {
        ModelCapabilities {
            max_output_tokens: 32_000,
            supports_structured_output: false,
            supports_adaptive_thinking: false,
            rejects_sampling_parameters: false,
            supports_xhigh_effort: false,
            is_known_model: true,
        }
    } else if model_id.contains("claude-3-haiku") {
        ModelCapabilities {
            max_output_tokens: 4_096,
            supports_structured_output: false,
            supports_adaptive_thinking: false,
            rejects_sampling_parameters: false,
            supports_xhigh_effort: false,
            is_known_model: true,
        }
    } else {
        ModelCapabilities {
            max_output_tokens: 4_096,
            supports_structured_output: false,
            supports_adaptive_thinking: false,
            rejects_sampling_parameters: false,
            supports_xhigh_effort: false,
            is_known_model: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opus_4_7_caps() {
        let c = model_capabilities("claude-opus-4-7-20251015");
        assert_eq!(c.max_output_tokens, 128_000);
        assert!(c.supports_adaptive_thinking);
        assert!(c.rejects_sampling_parameters);
        assert!(c.supports_xhigh_effort);
    }

    #[test]
    fn sonnet_4_6_caps() {
        let c = model_capabilities("claude-sonnet-4-6-20251014");
        assert_eq!(c.max_output_tokens, 128_000);
        assert!(c.supports_adaptive_thinking);
        assert!(!c.rejects_sampling_parameters);
    }

    #[test]
    fn unknown_falls_back_to_conservative() {
        let c = model_capabilities("some-future-model");
        assert_eq!(c.max_output_tokens, 4_096);
        assert!(!c.is_known_model);
    }
}
