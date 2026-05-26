//! Bedrock Converse API implementation (chat models).
//!
//! Mirrors `@ai-sdk/amazon-bedrock/src/amazon-bedrock-chat-language-model.ts`
//! and supporting helpers. The Converse API is the unified Bedrock chat
//! surface (`POST /model/{id}/converse` for `do_generate`,
//! `POST /model/{id}/converse-stream` for `do_stream`) and works across
//! every chat-capable family on Bedrock (Anthropic, Nova, Llama, Mistral,
//! Cohere Command, ...).
//!
//! # Endpoints
//!
//! - Non-streaming JSON: `POST {base}/model/{id}/converse`
//! - Streaming (AWS EventStream binary frames):
//!   `POST {base}/model/{id}/converse-stream`
// Rust guideline compliant 2026-05-25

mod convert_prompt;
mod finish_reason;
mod model;
mod normalize_tool_call_id;
mod options;
mod parse_response;
mod prepare_tools;
mod reasoning_mapper;
mod reasoning_metadata;
mod stream;
mod usage;
mod wire;

pub(crate) use convert_prompt::base64_encode as base64_encode_public;
pub use model::AmazonBedrockChatModel;
pub(crate) use model::encode_path_segment;
