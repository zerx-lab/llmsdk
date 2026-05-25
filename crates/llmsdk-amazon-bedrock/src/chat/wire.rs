//! On-wire types for Bedrock Converse + ConverseStream.
//!
//! Mirrors `amazon-bedrock-api-types.ts` and the inline schemas in
//! `amazon-bedrock-chat-language-model.ts`. Field names use Bedrock's
//! camelCase convention; sparse fields are skipped on serialize.
// Rust guideline compliant 2026-05-25

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// ============ request ============

/// Top-level Converse / ConverseStream request body.
#[derive(Debug, Clone, Default, Serialize)]
pub(crate) struct ConverseRequest {
    /// System messages (text blocks + optional cache points).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub system: Vec<SystemBlock>,
    /// User / assistant turn list.
    pub messages: Vec<WireMessage>,
    /// Tool configuration. Skipped when no tools are configured.
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    /// Inference parameters (max tokens / temperature / top-p / top-k / stop).
    #[serde(rename = "inferenceConfig", skip_serializing_if = "Option::is_none")]
    pub inference_config: Option<InferenceConfig>,
    /// Pass-through `additionalModelRequestFields` for model-specific knobs.
    #[serde(
        rename = "additionalModelRequestFields",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_model_request_fields: Option<Value>,
    /// JSON-pointer paths to copy out of `additionalModelResponseFields`.
    #[serde(
        rename = "additionalModelResponseFieldPaths",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_model_response_field_paths: Option<Vec<String>>,
    /// Service tier descriptor.
    #[serde(rename = "serviceTier", skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    /// Guardrail configuration (forwarded verbatim).
    #[serde(rename = "guardrailConfig", skip_serializing_if = "Option::is_none")]
    pub guardrail_config: Option<Value>,
    /// Performance configuration (`{ "latency": "standard"|"optimized" }`).
    #[serde(rename = "performanceConfig", skip_serializing_if = "Option::is_none")]
    pub performance_config: Option<Value>,
    /// `requestMetadata` opaque map forwarded to the upstream model.
    #[serde(rename = "requestMetadata", skip_serializing_if = "Option::is_none")]
    pub request_metadata: Option<Value>,
    /// Prompt variables (Bedrock Prompt Management).
    #[serde(rename = "promptVariables", skip_serializing_if = "Option::is_none")]
    pub prompt_variables: Option<Value>,
}

/// System-block — either text or a cache point.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum SystemBlock {
    /// Plain text system entry.
    Text {
        /// Text payload.
        text: String,
    },
    /// Cache-point marker.
    CachePoint {
        /// Cache-point payload.
        #[serde(rename = "cachePoint")]
        cache_point: CachePointValue,
    },
}

/// `cachePoint` payload (Bedrock prompt caching).
#[derive(Debug, Clone, Serialize)]
pub(crate) struct CachePointValue {
    /// Cache type — always `"default"` upstream.
    #[serde(rename = "type")]
    pub kind: String,
    /// Optional TTL (`"5m"` / `"1h"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

/// One conversation turn.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireMessage {
    /// Either `"user"` or `"assistant"`.
    pub role: String,
    /// Ordered content blocks for this turn.
    pub content: Vec<ContentBlock>,
}

/// A single content block on a [`WireMessage`].
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum ContentBlock {
    /// Plain text.
    Text {
        /// Text payload.
        text: String,
    },
    /// Inline image bytes.
    Image {
        /// `{ format, source: { bytes } }`.
        image: ImageBlock,
    },
    /// Document / file.
    Document {
        /// `{ format, name, source: { bytes }, citations? }`.
        document: DocumentBlock,
    },
    /// Tool call requested by the model.
    ToolUse {
        /// `{ toolUseId, name, input }`.
        #[serde(rename = "toolUse")]
        tool_use: ToolUseBlock,
    },
    /// Tool result returned by the caller.
    ToolResult {
        /// `{ toolUseId, content[] }`.
        #[serde(rename = "toolResult")]
        tool_result: ToolResultBlock,
    },
    /// Reasoning content (visible or redacted).
    ReasoningContent {
        /// Either `reasoningText` or `redactedReasoning` payload.
        #[serde(rename = "reasoningContent")]
        reasoning_content: ReasoningContentBlock,
    },
    /// Cache-point marker between blocks.
    CachePoint {
        /// Cache-point payload.
        #[serde(rename = "cachePoint")]
        cache_point: CachePointValue,
    },
    /// Guard-content marker (forwarded verbatim).
    #[allow(
        dead_code,
        reason = "wire variant kept for forward-compat with Bedrock guardrails"
    )]
    GuardContent {
        /// Guard-content payload.
        #[serde(rename = "guardContent")]
        guard_content: Value,
    },
}

/// `image` content block.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ImageBlock {
    /// Image format (`"jpeg"`, `"png"`, `"gif"`, `"webp"`).
    pub format: String,
    /// `{ bytes: base64 }` source.
    pub source: BytesSource,
}

/// `document` content block.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct DocumentBlock {
    /// Document format (`"pdf"`, `"txt"`, `"md"`, ...).
    pub format: String,
    /// Document name (filename with extension stripped).
    pub name: String,
    /// `{ bytes: base64 }` source.
    pub source: BytesSource,
    /// Optional citations configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<CitationsConfig>,
}

/// `{ enabled: bool }` citations toggle.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct CitationsConfig {
    /// Whether to enable citations for this document.
    pub enabled: bool,
}

/// Generic `{ bytes: base64 }` source.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct BytesSource {
    /// Base64-encoded payload.
    pub bytes: String,
}

/// `toolUse` content block.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ToolUseBlock {
    /// Caller-assigned tool-call id.
    #[serde(rename = "toolUseId")]
    pub tool_use_id: String,
    /// Tool name.
    pub name: String,
    /// Tool input arguments (JSON object).
    pub input: Value,
}

/// `toolResult` content block.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ToolResultBlock {
    /// Originating tool-call id.
    #[serde(rename = "toolUseId")]
    pub tool_use_id: String,
    /// Result payload (text / image blocks).
    pub content: Vec<ToolResultPart>,
}

/// One entry inside a `toolResult.content[]`.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum ToolResultPart {
    /// Text fragment.
    Text {
        /// Text payload.
        text: String,
    },
    /// Image fragment.
    #[allow(dead_code, reason = "wire variant kept for image-bearing tool results")]
    Image {
        /// `{ format, source: { bytes } }`.
        image: ImageBlock,
    },
}

/// `reasoningContent` content block — two variants.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum ReasoningContentBlock {
    /// Visible reasoning text + optional signature.
    Text {
        /// `{ text, signature? }`.
        #[serde(rename = "reasoningText")]
        reasoning_text: ReasoningText,
    },
    /// Redacted reasoning (opaque blob).
    Redacted {
        /// `{ data: base64 }`.
        #[serde(rename = "redactedReasoning")]
        redacted_reasoning: RedactedReasoning,
    },
}

/// `reasoningText` payload.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ReasoningText {
    /// Reasoning text.
    pub text: String,
    /// Optional signature for round-trip verification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// `redactedReasoning` payload.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct RedactedReasoning {
    /// Base64-encoded opaque blob.
    pub data: String,
}

/// Inference configuration (camelCase Converse spec).
#[derive(Debug, Clone, Default, Serialize)]
pub(crate) struct InferenceConfig {
    /// Hard cap on generated tokens.
    #[serde(rename = "maxTokens", skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling.
    #[serde(rename = "topP", skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-K sampling.
    #[serde(rename = "topK", skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Stop sequences.
    #[serde(rename = "stopSequences", skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

impl InferenceConfig {
    /// Returns `true` when every optional field is `None` — used to decide
    /// whether to emit the surrounding object at all.
    pub(crate) fn is_empty(&self) -> bool {
        self.max_tokens.is_none()
            && self.temperature.is_none()
            && self.top_p.is_none()
            && self.top_k.is_none()
            && self.stop_sequences.is_none()
    }
}

/// Tool configuration (toolConfig).
#[derive(Debug, Clone, Default, Serialize)]
pub(crate) struct ToolConfig {
    /// Tool definitions (function tools wrapped in `toolSpec`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolConfigEntry>>,
    /// Tool-choice policy.
    #[serde(rename = "toolChoice", skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoiceWire>,
}

/// Either a function-tool spec or a cache point marker.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum ToolConfigEntry {
    /// `{ toolSpec: { name, description?, inputSchema: { json } } }`.
    Spec {
        /// Wrapper required by Converse.
        #[serde(rename = "toolSpec")]
        tool_spec: ToolSpec,
    },
    /// Cache-point between tool specs.
    #[allow(
        dead_code,
        reason = "wire variant kept for forward-compat with prompt caching"
    )]
    CachePoint {
        /// Cache-point payload.
        #[serde(rename = "cachePoint")]
        cache_point: CachePointValue,
    },
}

/// `toolSpec` payload.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ToolSpec {
    /// Tool name.
    pub name: String,
    /// Optional description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Optional strict-mode flag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    /// `{ json: <schema> }` wrapper.
    #[serde(rename = "inputSchema")]
    pub input_schema: InputSchema,
}

/// `{ json: <JSON Schema> }` wrapper.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct InputSchema {
    /// JSON schema as a raw object.
    pub json: Value,
}

/// `toolChoice` discriminated union.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum ToolChoiceWire {
    /// `{ "auto": {} }`.
    Auto {
        /// Empty marker.
        auto: Map<String, Value>,
    },
    /// `{ "any": {} }`.
    Any {
        /// Empty marker.
        any: Map<String, Value>,
    },
    /// `{ "tool": { "name": "x" } }`.
    Tool {
        /// `{ name }` payload.
        tool: ToolChoiceTool,
    },
}

/// `tool` payload within [`ToolChoiceWire::Tool`].
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ToolChoiceTool {
    /// Forced tool name.
    pub name: String,
}

/// Service tier descriptor `{ "type": "flex" | ... }`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ServiceTier {
    /// Tier name.
    #[serde(rename = "type")]
    pub kind: String,
}

// ============ response ============

/// Successful Converse response.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ConverseResponse {
    /// `{ message: { content: [...], role } }`.
    pub output: ConverseOutput,
    /// Why the model stopped.
    #[serde(rename = "stopReason")]
    pub stop_reason: Option<String>,
    /// `additionalModelResponseFields` (provider-specific JSON).
    #[serde(default, rename = "additionalModelResponseFields")]
    pub additional_model_response_fields: Option<Value>,
    /// Trace payload (verbatim).
    #[serde(default)]
    pub trace: Option<Value>,
    /// Performance configuration the call ran under.
    #[serde(default, rename = "performanceConfig")]
    pub performance_config: Option<Value>,
    /// Service tier the call ran under.
    #[serde(default, rename = "serviceTier")]
    pub service_tier: Option<Value>,
    /// Token usage.
    pub usage: Option<BedrockUsage>,
}

/// `output` payload.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ConverseOutput {
    /// Assistant message.
    pub message: ConverseOutputMessage,
}

/// Assistant message payload.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ConverseOutputMessage {
    /// Ordered content blocks.
    pub content: Vec<ConverseOutputContent>,
}

/// One response content block. Field-discriminated to match Bedrock's wire.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ConverseOutputContent {
    /// Text payload.
    #[serde(default)]
    pub text: Option<String>,
    /// Tool-use payload.
    #[serde(default, rename = "toolUse")]
    pub tool_use: Option<ResponseToolUse>,
    /// Reasoning payload.
    #[serde(default, rename = "reasoningContent")]
    pub reasoning_content: Option<ResponseReasoningContent>,
}

/// Tool-use response payload.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ResponseToolUse {
    /// Tool-call id assigned by Bedrock.
    #[serde(default, rename = "toolUseId")]
    pub tool_use_id: Option<String>,
    /// Tool name.
    #[serde(default)]
    pub name: Option<String>,
    /// Input arguments. Bedrock always returns an object; we keep `Value`
    /// for robustness against future schema drift.
    #[serde(default)]
    pub input: Option<Value>,
}

/// Reasoning response payload — either `reasoningText` or `redactedReasoning`.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ResponseReasoningContent {
    /// Visible reasoning text payload.
    #[serde(default, rename = "reasoningText")]
    pub reasoning_text: Option<ResponseReasoningText>,
    /// Redacted reasoning payload.
    #[serde(default, rename = "redactedReasoning")]
    pub redacted_reasoning: Option<ResponseRedactedReasoning>,
}

/// `reasoningText` deserialization shape.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ResponseReasoningText {
    /// Reasoning text.
    pub text: String,
    /// Optional signature.
    #[serde(default)]
    pub signature: Option<String>,
}

/// `redactedReasoning` deserialization shape.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ResponseRedactedReasoning {
    /// Base64 opaque blob.
    #[serde(default)]
    pub data: Option<String>,
}

/// Usage payload returned by Converse + ConverseStream.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct BedrockUsage {
    /// Input tokens.
    #[serde(
        default,
        rename = "inputTokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub input_tokens: Option<u64>,
    /// Output tokens.
    #[serde(
        default,
        rename = "outputTokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub output_tokens: Option<u64>,
    /// Combined total (input + output).
    #[serde(
        default,
        rename = "totalTokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub total_tokens: Option<u64>,
    /// Cache-read input tokens (Anthropic prompt caching on Bedrock).
    #[serde(
        default,
        rename = "cacheReadInputTokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_read_input_tokens: Option<u64>,
    /// Cache-write input tokens.
    #[serde(
        default,
        rename = "cacheWriteInputTokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_write_input_tokens: Option<u64>,
    /// Per-TTL cache breakdown.
    #[serde(
        default,
        rename = "cacheDetails",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_details: Option<Value>,
}

// ============ stream chunk schema ============

/// One decoded Converse-stream event JSON object.
///
/// Bedrock wraps event JSON inside a binary EventStream frame; we strip the
/// frame and parse the payload into this struct. The various `*_*` fields are
/// the discriminated event types upstream defines; field-discriminated
/// matches Bedrock's wire shape.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct StreamChunk {
    /// `messageStart` event (carries the role).
    #[allow(
        dead_code,
        reason = "captured for future telemetry; not surfaced today"
    )]
    #[serde(default, rename = "messageStart")]
    pub message_start: Option<MessageStart>,
    /// `messageStop` event.
    #[serde(default, rename = "messageStop")]
    pub message_stop: Option<MessageStop>,
    /// `contentBlockStart` event.
    #[serde(default, rename = "contentBlockStart")]
    pub content_block_start: Option<ContentBlockStart>,
    /// `contentBlockDelta` event.
    #[serde(default, rename = "contentBlockDelta")]
    pub content_block_delta: Option<ContentBlockDelta>,
    /// `contentBlockStop` event.
    #[serde(default, rename = "contentBlockStop")]
    pub content_block_stop: Option<ContentBlockStop>,
    /// `metadata` event (usage + perf + trace + tier).
    #[serde(default)]
    pub metadata: Option<MetadataEvent>,
    /// Exception variants.
    #[serde(default, rename = "internalServerException")]
    pub internal_server_exception: Option<Value>,
    /// Stream error variant.
    #[serde(default, rename = "modelStreamErrorException")]
    pub model_stream_error_exception: Option<Value>,
    /// Throttle variant.
    #[serde(default, rename = "throttlingException")]
    pub throttling_exception: Option<Value>,
    /// Validation variant.
    #[serde(default, rename = "validationException")]
    pub validation_exception: Option<Value>,
}

/// `messageStart` payload.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code, reason = "wire field captured for forward-compat")]
pub(crate) struct MessageStart {
    /// Role of the emerging message (`"assistant"`).
    #[serde(default)]
    pub role: Option<String>,
}

/// `messageStop` payload.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MessageStop {
    /// Why the model stopped.
    #[serde(default, rename = "stopReason")]
    pub stop_reason: Option<String>,
    /// `additionalModelResponseFields` from the terminal event.
    #[serde(default, rename = "additionalModelResponseFields")]
    pub additional_model_response_fields: Option<Value>,
}

/// `contentBlockStart` payload.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ContentBlockStart {
    /// Block index inside the assistant message.
    #[serde(default, rename = "contentBlockIndex")]
    pub content_block_index: Option<u32>,
    /// Optional start payload (carries `toolUse` for tool blocks).
    #[serde(default)]
    pub start: Option<ContentBlockStartPayload>,
}

/// `start.*` payload.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ContentBlockStartPayload {
    /// Tool-use start (`{ toolUseId, name }`).
    #[serde(default, rename = "toolUse")]
    pub tool_use: Option<StreamToolUse>,
}

/// `toolUse` payload in `contentBlockStart`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StreamToolUse {
    /// Tool-call id.
    #[serde(default, rename = "toolUseId")]
    pub tool_use_id: Option<String>,
    /// Tool name.
    #[serde(default)]
    pub name: Option<String>,
}

/// `contentBlockDelta` payload.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ContentBlockDelta {
    /// Block index.
    #[serde(default, rename = "contentBlockIndex")]
    pub content_block_index: Option<u32>,
    /// Delta payload.
    #[serde(default)]
    pub delta: Option<DeltaPayload>,
}

/// `delta.*` payload.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct DeltaPayload {
    /// Text delta.
    #[serde(default)]
    pub text: Option<String>,
    /// Tool-use input delta (`{ input: "..." }` — JSON fragment).
    #[serde(default, rename = "toolUse")]
    pub tool_use: Option<DeltaToolUse>,
    /// Reasoning content delta (text / signature / data).
    #[serde(default, rename = "reasoningContent")]
    pub reasoning_content: Option<DeltaReasoning>,
}

/// Tool-use delta payload.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DeltaToolUse {
    /// JSON fragment of the cumulative input.
    #[serde(default)]
    pub input: Option<String>,
}

/// Reasoning delta payload — exactly one of `text` / `signature` / `data`
/// is populated per event.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct DeltaReasoning {
    /// Visible reasoning text fragment.
    #[serde(default)]
    pub text: Option<String>,
    /// Reasoning signature emitted at the end of the block.
    #[serde(default)]
    pub signature: Option<String>,
    /// Redacted opaque blob (one-shot).
    #[serde(default)]
    pub data: Option<String>,
}

/// `contentBlockStop` payload.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ContentBlockStop {
    /// Block index that just ended.
    #[serde(default, rename = "contentBlockIndex")]
    pub content_block_index: Option<u32>,
}

/// `metadata` event payload.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct MetadataEvent {
    /// Token usage.
    #[serde(default)]
    pub usage: Option<BedrockUsage>,
    /// Trace payload (verbatim).
    #[serde(default)]
    pub trace: Option<Value>,
    /// Performance config the call ran under.
    #[serde(default, rename = "performanceConfig")]
    pub performance_config: Option<Value>,
    /// Service tier descriptor.
    #[serde(default, rename = "serviceTier")]
    pub service_tier: Option<Value>,
}
