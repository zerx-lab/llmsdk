//! Map the top-level `CallOptions.reasoning` enum onto Bedrock's
//! `reasoningConfig` block, mirroring upstream
//! `resolveAmazonBedrockReasoningConfig`
//! (`amazon-bedrock-chat-language-model.ts:1210-1294`).
//!
//! Anthropic models take three shapes (disabled / adaptive / enabled), and
//! all other models receive a top-level `maxReasoningEffort`. Effort levels
//! that the wire does not support are mapped through
//! `amazonBedrockReasoningEffortMap` (`minimal` / `xhigh` collapse onto
//! adjacent levels with a `compatibility` warning).
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::ReasoningEffort;
use llmsdk_provider::shared::Warning;

use super::options::ReasoningConfig;

/// Translate a top-level `ReasoningEffort` to the Bedrock effort string.
///
/// Mirrors `amazonBedrockReasoningEffortMap`
/// (`amazon-bedrock-chat-language-model.ts:1210-1218`): `minimal` and
/// `low` both project onto `"low"`, and `xhigh` projects onto `"max"` (the
/// Bedrock wire literal). Returns `None` for `ProviderDefault` / `None` —
/// callers above this layer have already routed those through the wider
/// reasoning-config decision tree.
fn map_effort_to_provider(effort: ReasoningEffort) -> Option<&'static str> {
    match effort {
        ReasoningEffort::Minimal | ReasoningEffort::Low => Some("low"),
        ReasoningEffort::Medium => Some("medium"),
        ReasoningEffort::High => Some("high"),
        ReasoningEffort::Xhigh => Some("max"),
        ReasoningEffort::None | ReasoningEffort::ProviderDefault => None,
    }
}

/// Emit a `compatibility` warning when an effort level is silently
/// collapsed (e.g. `minimal` → `low`, `xhigh` → `max`). Mirrors the
/// `mapReasoningToProviderEffort`
/// (`packages/provider-utils/src/map-reasoning-to-provider.ts:30-58`)
/// warning shape.
fn note_effort_compatibility(effort: ReasoningEffort, mapped: &str, warnings: &mut Vec<Warning>) {
    let raw = effort_to_str(effort);
    if raw == mapped {
        return;
    }
    warnings.push(Warning::Compatibility {
        feature: "reasoning".into(),
        details: Some(format!(
            "reasoning \"{raw}\" is not directly supported by this model. mapped to effort \"{mapped}\"."
        )),
    });
}

fn effort_to_str(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
        ReasoningEffort::None => "none",
        ReasoningEffort::ProviderDefault => "provider-default",
    }
}

/// Map a reasoning level to an absolute token budget by multiplying the
/// model's max output tokens by a per-level percentage, clamped to
/// `[1024, max_reasoning_budget]`.
///
/// Mirrors `mapReasoningToProviderBudget`
/// (`packages/provider-utils/src/map-reasoning-to-provider.ts:60-110`)
/// including the `DEFAULT_REASONING_BUDGET_PERCENTAGES` table
/// (minimal=2% / low=10% / medium=30% / high=60% / xhigh=90%) and the
/// 1024-token floor.
fn map_reasoning_to_provider_budget(
    effort: ReasoningEffort,
    max_output_tokens: u32,
    max_reasoning_budget: u32,
) -> Option<u32> {
    const MIN_REASONING_BUDGET: u32 = 1024;
    let pct = match effort {
        ReasoningEffort::Minimal => 0.02,
        ReasoningEffort::Low => 0.10,
        ReasoningEffort::Medium => 0.30,
        ReasoningEffort::High => 0.60,
        ReasoningEffort::Xhigh => 0.90,
        ReasoningEffort::None | ReasoningEffort::ProviderDefault => return None,
    };
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "max_output_tokens fits u32; pct in [0,1]; result rounded to nearest int"
    )]
    let raw = (f64::from(max_output_tokens) * pct).round() as u32;
    let clamped = raw.clamp(MIN_REASONING_BUDGET, max_reasoning_budget);
    Some(clamped)
}

/// True when the Bedrock model id corresponds to an Anthropic-hosted model
/// (`*claude-*` plus the regional variants like `us.anthropic.claude-opus-4-7`).
///
/// Used to gate the three-way Anthropic reasoning routing
/// (disabled / adaptive / enabled). Mirrors upstream's `isAnthropicModel`
/// branch in `resolveAmazonBedrockReasoningConfig`.
fn is_anthropic_model(model_id: &str) -> bool {
    model_id.contains("claude") || model_id.contains("anthropic.")
}

/// Resolve the final `reasoning_config` block.
///
/// `bedrock_rc` is the user-supplied `provider_options.amazonBedrock.reasoningConfig`
/// (may already specify overrides); `reasoning` is the top-level effort.
/// Returns the merged config; emits warnings for compatibility downgrades
/// and unsupported scenarios.
///
/// Mirrors `resolveAmazonBedrockReasoningConfig`
/// (`amazon-bedrock-chat-language-model.ts:1220-1294`).
pub(crate) fn resolve_reasoning_config(
    reasoning: Option<ReasoningEffort>,
    bedrock_rc: Option<ReasoningConfig>,
    model_id: &str,
    warnings: &mut Vec<Warning>,
) -> Option<ReasoningConfig> {
    // `None` / `ProviderDefault` defer entirely to the user-supplied config
    // (matches upstream `isCustomReasoning` short-circuit).
    let effort = match reasoning {
        Some(e) if e != ReasoningEffort::ProviderDefault => e,
        _ => return bedrock_rc,
    };

    let mut result = bedrock_rc.unwrap_or_default();

    if is_anthropic_model(model_id) {
        let caps = llmsdk_anthropic::model_capabilities(model_id);

        if matches!(effort, ReasoningEffort::None) {
            result.kind = Some("disabled".into());
        } else if caps.supports_adaptive_thinking {
            if let Some(mapped) = map_effort_to_provider(effort) {
                note_effort_compatibility(effort, mapped, warnings);
                result.kind.get_or_insert_with(|| "adaptive".into());
                if result.max_reasoning_effort.is_none() {
                    result.max_reasoning_effort = Some(mapped.to_owned());
                }
            }
        } else if let Some(budget) =
            map_reasoning_to_provider_budget(effort, caps.max_output_tokens, caps.max_output_tokens)
        {
            result.kind.get_or_insert_with(|| "enabled".into());
            if result.budget_tokens.is_none() {
                result.budget_tokens = Some(budget);
            }
        }
    } else if !matches!(effort, ReasoningEffort::None)
        && let Some(mapped) = map_effort_to_provider(effort)
    {
        note_effort_compatibility(effort, mapped, warnings);
        if result.max_reasoning_effort.is_none() {
            result.max_reasoning_effort = Some(mapped.to_owned());
        }
    }

    // Mirror upstream :1288-1291: when the merged type ends up `disabled`,
    // strip derived effort / budget so downstream does not emit
    // `output_config.effort` alongside `disabled` thinking.
    if result.kind.as_deref() == Some("disabled") {
        result.max_reasoning_effort = None;
        result.budget_tokens = None;
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xhigh_collapses_to_max_with_compatibility_warning() {
        let mut warnings = Vec::new();
        let rc = resolve_reasoning_config(
            Some(ReasoningEffort::Xhigh),
            None,
            "us.anthropic.claude-opus-4-7-20260201-v1:0",
            &mut warnings,
        )
        .expect("anthropic + adaptive should produce a config");
        // claude-opus-4-7 supports adaptive thinking (mirrors anthropic capabilities).
        assert_eq!(rc.kind.as_deref(), Some("adaptive"));
        assert_eq!(rc.max_reasoning_effort.as_deref(), Some("max"));
        assert!(
            warnings.iter().any(
                |w| matches!(w, Warning::Compatibility { feature, .. } if feature == "reasoning")
            ),
            "xhigh → max should produce a compatibility warning"
        );
    }

    #[test]
    fn none_disables_thinking_and_strips_derived_fields() {
        let mut warnings = Vec::new();
        let pre = ReasoningConfig {
            kind: None,
            budget_tokens: Some(2048),
            max_reasoning_effort: Some("high".into()),
            display: None,
        };
        let rc = resolve_reasoning_config(
            Some(ReasoningEffort::None),
            Some(pre),
            "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            &mut warnings,
        )
        .expect("disabled config still emitted");
        assert_eq!(rc.kind.as_deref(), Some("disabled"));
        assert!(rc.budget_tokens.is_none(), "budget stripped under disabled");
        assert!(
            rc.max_reasoning_effort.is_none(),
            "effort stripped under disabled"
        );
    }

    #[test]
    fn non_anthropic_routes_to_max_reasoning_effort_only() {
        let mut warnings = Vec::new();
        let rc = resolve_reasoning_config(
            Some(ReasoningEffort::Medium),
            None,
            "amazon.nova-pro-v1:0",
            &mut warnings,
        )
        .expect("non-anthropic routes to maxReasoningEffort");
        assert!(rc.kind.is_none(), "non-anthropic does not set type");
        assert_eq!(rc.max_reasoning_effort.as_deref(), Some("medium"));
    }

    #[test]
    fn provider_default_passes_through_user_config() {
        let mut warnings = Vec::new();
        let pre = ReasoningConfig {
            kind: Some("enabled".into()),
            budget_tokens: Some(1500),
            max_reasoning_effort: None,
            display: None,
        };
        let rc = resolve_reasoning_config(
            Some(ReasoningEffort::ProviderDefault),
            Some(pre.clone()),
            "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            &mut warnings,
        )
        .expect("provider-default returns user config");
        assert_eq!(rc.kind.as_deref(), Some("enabled"));
        assert_eq!(rc.budget_tokens, Some(1500));
        assert!(warnings.is_empty(), "no warning for provider-default");
    }
}
