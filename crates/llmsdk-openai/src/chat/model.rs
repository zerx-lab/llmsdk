//! [`LanguageModel`] implementation for `OpenAI` Chat Completions.
//!
//! Top-level entry: [`OpenAiChatModel`]. Construct via [`crate::OpenAi::chat`].
// Rust guideline compliant 2026-02-21

use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    CallOptions, GenerateResult, LanguageModel, ResponseFormat, StreamResult, Tool, ToolChoice,
};
use llmsdk_provider::shared::Warning;
use llmsdk_provider_utils::http::{JsonRequest, post_json};

use crate::PROVIDER_ID;
use crate::config::Inner;

use super::wire::{
    ChatRequest, ChatResponse, ResponseFormat as WireResponseFormat, WireFunctionDef,
    WireJsonSchema, WireTool, WireToolCallKind, WireToolChoice, WireToolChoiceFunction,
    WireToolChoiceSimple,
};
use super::{convert_prompt, parse_response};
use crate::error::extract_error_message;

/// `OpenAI` Chat Completions model handle.
///
/// Cheap to clone. Multiple clones share the underlying HTTP client and
/// authentication state via [`OpenAi`]'s `Arc`.
#[derive(Debug, Clone)]
pub struct OpenAiChatModel {
    inner: Arc<Inner>,
    model_id: String,
}

impl OpenAiChatModel {
    pub(crate) fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.inner.base_url)
    }
}

#[async_trait]
impl LanguageModel for OpenAiChatModel {
    fn provider(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult, ProviderError> {
        let (request, warnings) = build_request(&self.model_id, &options);

        // Merge per-provider headers with per-call headers (call-site wins).
        let mut request_headers = self.inner.headers.clone();
        if let Some(headers) = &options.headers {
            for (name, value) in headers {
                request_headers.insert(name.clone(), value.clone());
            }
        }

        let request_body_value = serde_json::to_value(&request).ok();

        let mut http_request = JsonRequest::new(self.endpoint(), request);
        http_request.headers = request_headers;

        let response = match post_json::<_, ChatResponse>(&self.inner.http, http_request).await {
            Ok(r) => r,
            Err(err) => return Err(rewrite_openai_error(err)),
        };

        parse_response(
            response.value,
            response.headers,
            request_body_value,
            warnings,
        )
    }

    async fn do_stream(&self, _options: CallOptions) -> Result<StreamResult, ProviderError> {
        Err(ProviderError::unsupported(
            "OpenAiChatModel::do_stream (arriving in M4)",
        ))
    }
}

/// Build the wire request and collect warnings about dropped settings.
fn build_request(model_id: &str, options: &CallOptions) -> (ChatRequest, Vec<Warning>) {
    let (messages, mut warnings) = convert_prompt(&options.prompt);

    if options.top_k.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "topK".to_owned(),
            details: Some("OpenAI Chat Completions does not accept topK".to_owned()),
        });
    }

    if options.include_raw_chunks.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "includeRawChunks".to_owned(),
            details: Some("only meaningful for do_stream (M4)".to_owned()),
        });
    }

    let response_format = options
        .response_format
        .as_ref()
        .map(convert_response_format);
    let (tools, tool_choice) = convert_tools(
        options.tools.as_deref(),
        options.tool_choice.as_ref(),
        &mut warnings,
    );

    let request = ChatRequest {
        model: model_id.to_owned(),
        messages,
        max_tokens: options.max_output_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        frequency_penalty: options.frequency_penalty,
        presence_penalty: options.presence_penalty,
        seed: options.seed,
        stop: options.stop_sequences.clone(),
        response_format,
        tools,
        tool_choice,
    };

    (request, warnings)
}

/// Rewrite the [`ProviderError`] message to include the `OpenAI`-reported
/// error text, when present.
///
/// The transport layer in `provider-utils` produces messages like
/// `"HTTP 429 Too Many Requests"`. For `OpenAI` we want
/// `"OpenAI API error: rate limited (HTTP 429)"`. Non-`ApiCall` errors and
/// errors without a parseable body pass through unchanged.
fn rewrite_openai_error(err: ProviderError) -> ProviderError {
    if !err.is_api_call() {
        return err;
    }
    let Some(body) = err.response_body() else {
        return err;
    };
    let detail = extract_error_message(body);
    if detail.is_empty() {
        return err;
    }
    let status = err.status_code();
    let url = err.url().unwrap_or("").to_owned();
    let mut builder = ProviderError::api_call_builder(
        url,
        match status {
            Some(s) => format!("OpenAI API error: {detail} (HTTP {s})"),
            None => format!("OpenAI API error: {detail}"),
        },
    )
    .response_body(body.to_owned())
    .retryable(err.is_retryable());
    if let Some(s) = status {
        builder = builder.status_code(s);
    }
    builder.build()
}

fn convert_response_format(fmt: &ResponseFormat) -> WireResponseFormat {
    match fmt {
        ResponseFormat::Text => WireResponseFormat::JsonObject, // unreachable in practice; only set when caller asks for JSON
        ResponseFormat::Json {
            schema,
            name,
            description,
        } => match schema {
            Some(schema) => WireResponseFormat::JsonSchema {
                json_schema: WireJsonSchema {
                    name: name.clone().unwrap_or_else(|| "response".to_owned()),
                    description: description.clone(),
                    schema: schema.clone(),
                    strict: true,
                },
            },
            None => WireResponseFormat::JsonObject,
        },
    }
}

fn convert_tools(
    tools: Option<&[Tool]>,
    choice: Option<&ToolChoice>,
    warnings: &mut Vec<Warning>,
) -> (Option<Vec<WireTool>>, Option<WireToolChoice>) {
    let Some(tools) = tools else {
        return (None, None);
    };
    let tools = tools.iter().filter_map(|t| match t {
        Tool::Function(f) => Some(WireTool::Function {
            function: WireFunctionDef {
                name: f.name.clone(),
                description: f.description.clone(),
                parameters: f.input_schema.clone(),
                strict: f.strict,
            },
        }),
        Tool::Provider(p) => {
            warnings.push(Warning::UnsupportedTool {
                tool: p.name.clone(),
                details: Some("M3 does not relay provider-defined tools".to_owned()),
            });
            None
        }
    });
    let tools: Vec<_> = tools.collect();
    if tools.is_empty() {
        return (None, None);
    }

    let tool_choice = choice.map(|c| match c {
        ToolChoice::Auto => WireToolChoice::Simple(WireToolChoiceSimple::Auto),
        ToolChoice::None => WireToolChoice::Simple(WireToolChoiceSimple::None),
        ToolChoice::Required => WireToolChoice::Simple(WireToolChoiceSimple::Required),
        ToolChoice::Tool { tool_name } => WireToolChoice::Tool {
            kind: WireToolCallKind::Function,
            function: WireToolChoiceFunction {
                name: tool_name.clone(),
            },
        },
    });

    (Some(tools), tool_choice)
}
