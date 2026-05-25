//! Bedrock Mantle: OpenAI-compatible runtime hosted by AWS.
//!
//! Mirrors `@ai-sdk/amazon-bedrock/src/mantle/*`. Mantle serves a subset of
//! `OpenAI`-compatible Chat Completions + Responses endpoints at
//! `https://bedrock-mantle.{region}.api.aws/v1/...`. We reuse
//! [`llmsdk_openai::internal`] for the wire / streaming logic and plug
//! [`BedrockAuth`] in via the [`RequestSigner`] hook.
//!
//! [`BedrockAuth`]: crate::sigv4_auth::BedrockAuth
//! [`RequestSigner`]: llmsdk_openai::internal::RequestSigner
// Rust guideline compliant 2026-05-25

mod provider;

pub use provider::{BedrockMantle, BedrockMantleBuilder};
