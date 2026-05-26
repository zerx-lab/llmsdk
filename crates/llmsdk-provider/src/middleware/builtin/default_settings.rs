//! Fill missing [`CallOptions`] fields with provider-level defaults.
//!
//! Mirrors `@ai-sdk/ai/src/middleware/default-settings-middleware.ts`. Caller
//! values always win — defaults only apply to fields the caller left `None` /
//! unspecified. `prompt`, `tools` (when present), `tool_choice`, `headers` and
//! `provider_options` are *merged*: caller wins per-key, defaults supply
//! the rest.
// Rust guideline compliant 2026-02-21

use async_trait::async_trait;

use crate::error::Result;
use crate::language_model::{CallOptions, LanguageModel};
use crate::middleware::language_model::{CallKind, LanguageModelMiddleware};
use crate::shared::{Headers, ProviderOptions};

/// Middleware applying a baseline [`CallOptions`] to every call.
///
/// Construct once with the defaults you want and attach via
/// [`crate::wrap_language_model`].
#[derive(Debug, Clone)]
pub struct DefaultSettingsMiddleware {
    defaults: CallOptions,
}

impl DefaultSettingsMiddleware {
    /// Build with the given default options.
    #[must_use]
    pub fn new(defaults: CallOptions) -> Self {
        Self { defaults }
    }
}

#[async_trait]
impl LanguageModelMiddleware for DefaultSettingsMiddleware {
    async fn transform_params(
        &self,
        _kind: CallKind,
        params: CallOptions,
        _inner: &dyn LanguageModel,
    ) -> Result<CallOptions> {
        Ok(merge_call_options(self.defaults.clone(), params))
    }
}

fn merge_call_options(default: CallOptions, caller: CallOptions) -> CallOptions {
    CallOptions {
        prompt: if caller.prompt.is_empty() {
            default.prompt
        } else {
            caller.prompt
        },
        max_output_tokens: caller.max_output_tokens.or(default.max_output_tokens),
        temperature: caller.temperature.or(default.temperature),
        stop_sequences: caller.stop_sequences.or(default.stop_sequences),
        top_p: caller.top_p.or(default.top_p),
        top_k: caller.top_k.or(default.top_k),
        presence_penalty: caller.presence_penalty.or(default.presence_penalty),
        frequency_penalty: caller.frequency_penalty.or(default.frequency_penalty),
        response_format: caller.response_format.or(default.response_format),
        seed: caller.seed.or(default.seed),
        tools: caller.tools.or(default.tools),
        tool_choice: caller.tool_choice.or(default.tool_choice),
        include_raw_chunks: caller.include_raw_chunks.or(default.include_raw_chunks),
        headers: merge_headers(default.headers, caller.headers),
        reasoning: caller.reasoning.or(default.reasoning),
        provider_options: merge_provider_options(default.provider_options, caller.provider_options),
    }
}

fn merge_headers(default: Option<Headers>, caller: Option<Headers>) -> Option<Headers> {
    match (default, caller) {
        (None, c) => c,
        (Some(d), None) => Some(d),
        (Some(mut d), Some(c)) => {
            d.extend(c);
            Some(d)
        }
    }
}

fn merge_provider_options(
    default: Option<ProviderOptions>,
    caller: Option<ProviderOptions>,
) -> Option<ProviderOptions> {
    match (default, caller) {
        (None, c) => c,
        (Some(d), None) => Some(d),
        (Some(mut d), Some(c)) => {
            for (provider, caller_inner) in c {
                let entry = d.entry(provider).or_default();
                for (k, v) in caller_inner {
                    match entry.remove(&k) {
                        Some(base) => {
                            // Mirror upstream `mergeObjects` deep recursion
                            // (`packages/ai/src/util/merge-objects.ts:14-84`):
                            // when both sides are JSON objects (not arrays,
                            // not dates), merge recursively so per-feature
                            // overrides do not clobber sibling keys the
                            // caller did not mention.
                            entry.insert(k, deep_merge_value(base, v));
                        }
                        None => {
                            entry.insert(k, v);
                        }
                    }
                }
            }
            Some(d)
        }
    }
}

/// Deep merge two JSON values mirroring upstream `mergeObjects`.
///
/// When both sides are JSON objects, recurse per-key. Otherwise the
/// `overrides` value wins. Arrays / scalars / nulls do not merge.
fn deep_merge_value(base: serde_json::Value, overrides: serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match (base, overrides) {
        (Value::Object(mut b), Value::Object(o)) => {
            for (k, v) in o {
                match b.remove(&k) {
                    Some(base_v) => {
                        b.insert(k, deep_merge_value(base_v, v));
                    }
                    None => {
                        b.insert(k, v);
                    }
                }
            }
            Value::Object(b)
        }
        (_, overrides) => overrides,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::language_model::{GenerateResult, Message, Prompt, StreamResult};
    use crate::middleware::wrap_language_model;

    #[derive(Debug, Default)]
    struct Recorder(Mutex<Option<CallOptions>>);

    #[async_trait]
    impl LanguageModel for Recorder {
        fn provider(&self) -> &'static str {
            "rec"
        }
        fn model_id(&self) -> &'static str {
            "rec"
        }
        async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult> {
            *self.0.lock().expect("mutex") = Some(options);
            Ok(GenerateResult {
                content: vec![],
                finish_reason: crate::language_model::FinishReason::new(
                    crate::language_model::FinishReasonKind::Stop,
                ),
                usage: crate::language_model::Usage::default(),
                provider_metadata: None,
                request: None,
                response: None,
                warnings: vec![],
            })
        }
        async fn do_stream(&self, _opts: CallOptions) -> Result<StreamResult> {
            unimplemented!()
        }
    }

    fn user_prompt() -> Prompt {
        vec![Message::System {
            content: "sys".into(),
            provider_options: None,
        }]
    }

    #[tokio::test]
    async fn caller_fills_missing_fields_from_defaults() {
        let rec = Arc::new(Recorder::default());
        let defaults = CallOptions {
            temperature: Some(0.7),
            max_output_tokens: Some(1024),
            ..Default::default()
        };
        let wrapped = wrap_language_model(
            Arc::clone(&rec) as Arc<dyn LanguageModel>,
            [Arc::new(DefaultSettingsMiddleware::new(defaults))
                as Arc<dyn LanguageModelMiddleware>],
        );

        wrapped
            .do_generate(CallOptions {
                prompt: user_prompt(),
                temperature: Some(0.1),
                ..Default::default()
            })
            .await
            .expect("generate");

        let captured = rec.0.lock().expect("mutex").clone().expect("params");
        assert_eq!(captured.temperature, Some(0.1), "caller wins");
        assert_eq!(captured.max_output_tokens, Some(1024), "default filled");
    }

    #[tokio::test]
    async fn provider_options_merge_is_deep_recursive() {
        // Mirrors upstream `mergeObjects` semantics tested in
        // `packages/ai/src/util/merge-objects.test.ts`: when the caller
        // overrides a nested key, sibling keys at the *same nested level*
        // must survive from the defaults — a shallow per-key insert would
        // wipe them out. Catches the prior bug where Rust did per-key
        // insert on the inner Map but did not recurse into the JSON
        // value payload.
        let rec = Arc::new(Recorder::default());

        let mut defaults_inner = serde_json::Map::new();
        defaults_inner.insert(
            "feature".into(),
            serde_json::json!({ "enabled": true, "cache": true }),
        );
        let mut defaults_po = ProviderOptions::new();
        defaults_po.insert("anthropic".into(), defaults_inner);

        let defaults = CallOptions {
            provider_options: Some(defaults_po),
            ..Default::default()
        };
        let wrapped = wrap_language_model(
            Arc::clone(&rec) as Arc<dyn LanguageModel>,
            [Arc::new(DefaultSettingsMiddleware::new(defaults))
                as Arc<dyn LanguageModelMiddleware>],
        );

        let mut caller_inner = serde_json::Map::new();
        caller_inner.insert("feature".into(), serde_json::json!({ "enabled": false }));
        let mut caller_po = ProviderOptions::new();
        caller_po.insert("anthropic".into(), caller_inner);

        wrapped
            .do_generate(CallOptions {
                prompt: user_prompt(),
                provider_options: Some(caller_po),
                ..Default::default()
            })
            .await
            .expect("generate");

        let captured = rec.0.lock().expect("mutex").clone().expect("params");
        let merged = captured.provider_options.expect("provider_options merged");
        let anthropic = merged.get("anthropic").expect("anthropic key present");
        let feature = anthropic.get("feature").expect("feature key present");
        assert_eq!(feature["enabled"], false, "caller override survives");
        assert_eq!(
            feature["cache"], true,
            "sibling key from defaults must survive deep merge"
        );
    }
}
