//! Wire-format types for Gemini `generateContent` and
//! `streamGenerateContent`.
//!
//! Mirrors the schemas at the bottom of
//! `@ai-sdk/google/src/google-language-model.ts`. We keep types lenient
//! (most fields are `Option`) so partial / future responses don't fail
//! parsing.
// Rust guideline compliant 2026-05-25

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Top-level non-streaming response.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireResponse {
    #[serde(default)]
    pub candidates: Vec<WireCandidate>,
    #[serde(default, rename = "usageMetadata")]
    pub usage_metadata: Option<WireUsage>,
    #[serde(default, rename = "promptFeedback")]
    pub prompt_feedback: Option<WirePromptFeedback>,
}

/// One streaming chunk (same shape as a partial response).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireChunk {
    #[serde(default)]
    pub candidates: Option<Vec<WireCandidate>>,
    #[serde(default, rename = "usageMetadata")]
    pub usage_metadata: Option<WireUsage>,
    #[serde(default, rename = "promptFeedback")]
    pub prompt_feedback: Option<WirePromptFeedback>,
}

/// One candidate generation.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireCandidate {
    #[serde(default)]
    pub content: Option<WireContent>,
    #[serde(default, rename = "finishReason")]
    pub finish_reason: Option<String>,
    #[serde(default, rename = "finishMessage")]
    pub finish_message: Option<String>,
    #[serde(default, rename = "safetyRatings")]
    pub safety_ratings: Option<Vec<Value>>,
    #[serde(default, rename = "groundingMetadata")]
    pub grounding_metadata: Option<WireGroundingMetadata>,
    #[serde(default, rename = "urlContextMetadata")]
    pub url_context_metadata: Option<Value>,
}

/// Candidate content envelope (`{ role, parts }`).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireContent {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub parts: Option<Vec<WirePart>>,
}

/// One part inside a content envelope.
///
/// Gemini's wire shape is "any of these fields may appear together"; we
/// model it as a free-form object with typed convenience accessors.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WirePart {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub thought: Option<bool>,
    #[serde(default, rename = "thoughtSignature")]
    pub thought_signature: Option<String>,
    #[serde(default, rename = "inlineData")]
    pub inline_data: Option<WireInlineData>,
    #[serde(default, rename = "fileData")]
    pub file_data: Option<WireFileData>,
    #[serde(default, rename = "functionCall")]
    pub function_call: Option<WireFunctionCall>,
    #[serde(default, rename = "functionResponse")]
    pub function_response: Option<WireFunctionResponse>,
    #[serde(default, rename = "executableCode")]
    pub executable_code: Option<WireExecutableCode>,
    #[serde(default, rename = "codeExecutionResult")]
    pub code_execution_result: Option<WireCodeExecutionResult>,
    /// Server-tool call ({ toolType, args, id }).
    #[serde(default, rename = "toolCall")]
    pub tool_call: Option<WireServerToolCall>,
    /// Server-tool response ({ toolType, response, id }).
    #[serde(default, rename = "toolResponse")]
    pub tool_response: Option<WireServerToolResponse>,
}

/// `{ mimeType, data }` (base64).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireInlineData {
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    pub data: String,
}

/// `{ mimeType, fileUri }`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireFileData {
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    #[serde(rename = "fileUri")]
    pub file_uri: String,
}

/// `{ id?, name, args, partialArgs?, willContinue? }`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireFunctionCall {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub args: Option<Value>,
    #[serde(default, rename = "partialArgs")]
    pub partial_args: Option<Vec<WirePartialArg>>,
    #[serde(default, rename = "willContinue")]
    pub will_continue: Option<bool>,
}

/// Streaming partial-arg fragment used by Gemini 3 / Vertex streaming.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WirePartialArg {
    #[serde(rename = "jsonPath")]
    pub json_path: String,
    #[serde(default, rename = "stringValue")]
    pub string_value: Option<String>,
    #[serde(default, rename = "numberValue")]
    pub number_value: Option<f64>,
    #[serde(default, rename = "boolValue")]
    pub bool_value: Option<bool>,
    #[serde(default, rename = "nullValue")]
    pub null_value: Option<Value>,
    #[serde(default, rename = "willContinue")]
    pub will_continue: Option<bool>,
}

/// `{ id?, name, response, parts? }`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireFunctionResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub response: Option<Value>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireExecutableCode {
    #[serde(default)]
    pub language: Option<String>,
    pub code: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireCodeExecutionResult {
    pub outcome: String,
    #[serde(default)]
    pub output: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireServerToolCall {
    #[serde(rename = "toolType")]
    pub tool_type: String,
    #[serde(default)]
    pub args: Option<Value>,
    pub id: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireServerToolResponse {
    #[serde(rename = "toolType")]
    pub tool_type: String,
    #[serde(default)]
    pub response: Option<Value>,
    pub id: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireUsage {
    #[serde(default, rename = "cachedContentTokenCount")]
    pub cached_content_token_count: Option<u64>,
    #[serde(default, rename = "thoughtsTokenCount")]
    pub thoughts_token_count: Option<u64>,
    #[serde(default, rename = "promptTokenCount")]
    pub prompt_token_count: Option<u64>,
    #[serde(default, rename = "candidatesTokenCount")]
    pub candidates_token_count: Option<u64>,
    #[serde(default, rename = "totalTokenCount")]
    pub total_token_count: Option<u64>,
    #[serde(default, rename = "trafficType")]
    pub traffic_type: Option<String>,
    #[serde(default, rename = "serviceTier")]
    pub service_tier: Option<String>,
    #[serde(default, rename = "promptTokensDetails")]
    pub prompt_tokens_details: Option<Vec<WireTokenDetail>>,
    #[serde(default, rename = "candidatesTokensDetails")]
    pub candidates_tokens_details: Option<Vec<WireTokenDetail>>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireTokenDetail {
    #[serde(default)]
    pub modality: Option<String>,
    #[serde(default, rename = "tokenCount")]
    pub token_count: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WirePromptFeedback {
    #[serde(default, rename = "blockReason")]
    pub block_reason: Option<String>,
    #[serde(default, rename = "safetyRatings")]
    pub safety_ratings: Option<Vec<Value>>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireGroundingMetadata {
    #[serde(default, rename = "groundingChunks")]
    pub grounding_chunks: Option<Vec<WireGroundingChunk>>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireGroundingChunk {
    #[serde(default)]
    pub web: Option<WireGroundingWeb>,
    #[serde(default)]
    pub image: Option<WireGroundingImage>,
    #[serde(default, rename = "retrievedContext")]
    pub retrieved_context: Option<WireGroundingRetrieved>,
    #[serde(default)]
    pub maps: Option<WireGroundingMaps>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireGroundingWeb {
    pub uri: String,
    #[serde(default)]
    pub title: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireGroundingImage {
    #[serde(rename = "sourceUri")]
    pub source_uri: String,
    #[serde(default)]
    pub title: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireGroundingRetrieved {
    #[serde(default)]
    pub uri: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default, rename = "fileSearchStore")]
    pub file_search_store: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireGroundingMaps {
    #[serde(default)]
    pub uri: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
}

/// Builder helper: convert a [`WireUsage`] back into a JSON map for the
/// `provider_metadata.google.usageMetadata` slot, omitting `None` fields.
pub(crate) fn usage_to_json_map(u: &WireUsage) -> Map<String, Value> {
    let mut m = Map::new();
    if let Some(v) = u.cached_content_token_count {
        m.insert("cachedContentTokenCount".into(), Value::from(v));
    }
    if let Some(v) = u.thoughts_token_count {
        m.insert("thoughtsTokenCount".into(), Value::from(v));
    }
    if let Some(v) = u.prompt_token_count {
        m.insert("promptTokenCount".into(), Value::from(v));
    }
    if let Some(v) = u.candidates_token_count {
        m.insert("candidatesTokenCount".into(), Value::from(v));
    }
    if let Some(v) = u.total_token_count {
        m.insert("totalTokenCount".into(), Value::from(v));
    }
    if let Some(ref s) = u.traffic_type {
        m.insert("trafficType".into(), Value::String(s.clone()));
    }
    if let Some(ref s) = u.service_tier {
        m.insert("serviceTier".into(), Value::String(s.clone()));
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_response() {
        let raw = r#"{
            "candidates":[{
                "content":{"role":"model","parts":[{"text":"hi"}]},
                "finishReason":"STOP"
            }],
            "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}
        }"#;
        let r: WireResponse = serde_json::from_str(raw).expect("ok");
        assert_eq!(r.candidates.len(), 1);
        assert_eq!(
            r.candidates[0]
                .content
                .as_ref()
                .and_then(|c| c.parts.as_ref())
                .map(Vec::len),
            Some(1)
        );
        let u = r.usage_metadata.unwrap();
        assert_eq!(u.total_token_count, Some(8));
    }

    #[test]
    fn parse_function_call() {
        let raw = r#"{"candidates":[{"content":{"role":"model","parts":[
            {"functionCall":{"name":"getWeather","args":{"city":"Tokyo"}}}
        ]},"finishReason":"STOP"}]}"#;
        let r: WireResponse = serde_json::from_str(raw).expect("ok");
        let p = &r.candidates[0]
            .content
            .as_ref()
            .unwrap()
            .parts
            .as_ref()
            .unwrap()[0];
        assert_eq!(
            p.function_call.as_ref().and_then(|f| f.name.as_deref()),
            Some("getWeather")
        );
    }
}
