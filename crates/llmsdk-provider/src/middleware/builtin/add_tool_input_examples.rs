//! Append `input_examples` to each tool's description so non-tool-using
//! models can still see the examples.
//!
//! Mirrors `@ai-sdk/ai/src/middleware/add-tool-input-examples-middleware.ts`.
// Rust guideline compliant 2026-02-21

use async_trait::async_trait;

use crate::error::Result;
use crate::language_model::{CallOptions, FunctionTool, LanguageModel, Tool};
use crate::middleware::language_model::{CallKind, LanguageModelMiddleware};

/// Middleware that serializes `tool.input_examples` (if any) and appends them
/// to the tool's `description` field.
///
/// Default layout mirrors `@ai-sdk/ai/src/middleware/add-tool-input-examples-middleware.ts`:
/// `"{description}\n\n{prefix}\n{example_1}\n{example_2}..."` where `prefix`
/// defaults to `"Input Examples:"` and each example is `JSON.stringify(example.input)`
/// (no enumeration prefix). Override with [`Self::with_prefix`] to customise
/// the header line or [`Self::with_formatter`] to take full control.
pub struct AddToolInputExamplesMiddleware {
    prefix: String,
    formatter: ExampleFormatter,
}

/// Boxed formatter that renders a list of [`crate::language_model::ToolInputExample`]
/// into a string appended to a tool's description. The middleware passes the
/// configured `prefix` alongside the examples so custom formatters can keep
/// the header line in sync.
type ExampleFormatter =
    Box<dyn Fn(&str, &[crate::language_model::ToolInputExample]) -> String + Send + Sync>;

impl std::fmt::Debug for AddToolInputExamplesMiddleware {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `formatter` is a boxed closure with no useful Debug representation;
        // mark non-exhaustive instead of dumping a function pointer address.
        f.debug_struct("AddToolInputExamplesMiddleware")
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl Default for AddToolInputExamplesMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl AddToolInputExamplesMiddleware {
    /// Build with the upstream-aligned default prefix `"Input Examples:"`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prefix: "Input Examples:".to_owned(),
            formatter: Box::new(default_formatter),
        }
    }

    /// Override the header line prepended before the serialized examples.
    /// Mirrors upstream `prefix` option (default `"Input Examples:"`).
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Override how examples are formatted into the description. The
    /// formatter receives the configured `prefix` plus the examples and
    /// returns the full block that will be appended verbatim.
    #[must_use]
    pub fn with_formatter<F>(mut self, formatter: F) -> Self
    where
        F: Fn(&str, &[crate::language_model::ToolInputExample]) -> String + Send + Sync + 'static,
    {
        self.formatter = Box::new(formatter);
        self
    }
}

fn default_formatter(prefix: &str, examples: &[crate::language_model::ToolInputExample]) -> String {
    use std::fmt::Write as _;
    // Upstream produces `"\n\n{prefix}\n{ex1}\n{ex2}..."`. The leading two
    // newlines separate the examples block from any pre-existing description;
    // they are stripped by the caller when `description` was empty.
    let mut buf = String::from("\n\n");
    buf.push_str(prefix);
    for ex in examples {
        let json =
            serde_json::to_string(&ex.input).unwrap_or_else(|_| "<unserializable>".to_owned());
        let _ = write!(buf, "\n{json}");
    }
    buf
}

#[async_trait]
impl LanguageModelMiddleware for AddToolInputExamplesMiddleware {
    async fn transform_params(
        &self,
        _kind: CallKind,
        mut params: CallOptions,
        _inner: &dyn LanguageModel,
    ) -> Result<CallOptions> {
        let Some(tools) = params.tools.as_mut() else {
            return Ok(params);
        };
        for tool in tools.iter_mut() {
            if let Tool::Function(FunctionTool {
                description,
                input_examples: Some(examples),
                ..
            }) = tool
            {
                if examples.is_empty() {
                    continue;
                }
                let suffix = (self.formatter)(&self.prefix, examples);
                *description = Some(match description.take() {
                    Some(existing) => format!("{existing}{suffix}"),
                    None => suffix.trim_start().to_owned(),
                });
            }
        }
        Ok(params)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::language_model::{GenerateResult, Prompt, StreamResult, ToolInputExample};
    use crate::middleware::wrap_language_model;
    use async_trait::async_trait;

    #[derive(Debug, Default)]
    struct LastParams(std::sync::Mutex<Option<CallOptions>>);

    #[derive(Debug)]
    struct Recorder(Arc<LastParams>);

    #[async_trait]
    impl LanguageModel for Recorder {
        fn provider(&self) -> &'static str {
            "rec"
        }
        fn model_id(&self) -> &'static str {
            "rec"
        }
        async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult> {
            *self.0.0.lock().expect("mutex") = Some(options);
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
        async fn do_stream(&self, _options: CallOptions) -> Result<StreamResult> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn appends_examples_to_description() {
        let last = Arc::new(LastParams::default());
        let inner: Arc<dyn LanguageModel> = Arc::new(Recorder(Arc::clone(&last)));
        let wrapped = wrap_language_model(
            inner,
            [Arc::new(AddToolInputExamplesMiddleware::new()) as Arc<dyn LanguageModelMiddleware>],
        );

        wrapped
            .do_generate(CallOptions {
                prompt: Prompt::default(),
                tools: Some(vec![Tool::Function(FunctionTool {
                    name: "get_weather".into(),
                    description: Some("Get weather".into()),
                    input_schema: serde_json::from_value(serde_json::json!({"type": "object"}))
                        .unwrap(),
                    input_examples: Some(vec![ToolInputExample {
                        input: serde_json::json!({"city": "Tokyo"})
                            .as_object()
                            .cloned()
                            .unwrap(),
                    }]),
                    strict: None,
                    provider_options: None,
                })]),
                ..Default::default()
            })
            .await
            .expect("generate");

        let captured = last.0.lock().expect("mutex").clone().expect("params");
        let tools = captured.tools.unwrap();
        let Tool::Function(f) = &tools[0] else {
            panic!("expected function tool");
        };
        let desc = f.description.as_ref().unwrap();
        assert!(desc.contains("Get weather"), "preserves original desc");
        assert!(desc.contains("Examples:"), "appends examples header");
        assert!(desc.contains("Tokyo"), "renders example body");
    }
}
