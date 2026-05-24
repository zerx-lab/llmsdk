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
                    entry.insert(k, v);
                }
            }
            Some(d)
        }
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
}
