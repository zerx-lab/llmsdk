//! Anthropic-on-Bedrock language model (`POST /model/{id}/invoke`).
//!
//! Mirrors `@ai-sdk/amazon-bedrock/src/anthropic/amazon-bedrock-anthropic-provider.ts`.
//!
//! Implementation note: instead of re-implementing the Anthropic Messages
//! wire format, this module composes [`llmsdk_anthropic::internal::Inner`]
//! with three Bedrock-specific hooks:
//!
//! - `endpoint` override → `{base}/model/{id}/{invoke|invoke-with-response-stream}`
//! - `body_transform` → strip `model`, inject `anthropic_version =
//!   "bedrock-2023-05-31"`
//! - `request_auth` → `SigV4` / bearer-token signature per request (via
//!   [`crate::sigv4_auth::AnthropicAuthAdapter`])
//!
//! The result is wrapped by [`AmazonBedrockAnthropicModel`] (an alias for
//! [`AnthropicMessagesModel`]); callers get the full Anthropic feature set
//! (tools / thinking / cache-control / citations / ...) over Bedrock's
//! `invoke` endpoint.
// Rust guideline compliant 2026-05-25

mod model;

pub use model::AmazonBedrockAnthropicModel;
pub(crate) use model::AmazonBedrockAnthropicModelExt;
