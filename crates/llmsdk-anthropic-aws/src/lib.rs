//! Claude Platform on AWS (a.k.a. "Anthropic on AWS") provider for llmsdk.
//!
//! Rust port of
//! [`@ai-sdk/anthropic-aws`](https://github.com/vercel/ai/tree/main/packages/anthropic-aws).
//! Reuses the entire [`llmsdk_anthropic`] request pipeline (Messages, Files,
//! Skills, typed server tools) and swaps in a per-request authentication
//! hook that signs each outbound POST with AWS Signature Version 4 or,
//! alternatively, attaches an `x-api-key` header for AWS-provisioned keys.
//!
//! # Authentication precedence
//!
//! Matches the upstream package:
//!
//! 1. If `apiKey` is provided (option or `ANTHROPIC_AWS_API_KEY` env var)
//!    — `x-api-key` is sent and `SigV4` is **not** attempted.
//! 2. Otherwise `SigV4` signs every POST using AWS credentials resolved from
//!    `accessKeyId` / `secretAccessKey` / `sessionToken` options or
//!    `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`
//!    env vars.
//!
//! # Required configuration
//!
//! - `region` (or `AWS_REGION`): used both to template the default base
//!   URL `https://aws-external-anthropic.{region}.api.aws/v1` and as the
//!   `SigV4` signing region.
//! - `workspace_id` (or `ANTHROPIC_AWS_WORKSPACE_ID`): sent on every request
//!   as `anthropic-workspace-id`.
//!
//! # Quick start
//!
//! ```no_run
//! # async fn doc() -> Result<(), llmsdk_provider::ProviderError> {
//! use llmsdk_anthropic_aws::AnthropicAws;
//! use llmsdk_provider::language_model::{CallOptions, Message, TextPart, UserPart};
//! use llmsdk_provider::LanguageModel;
//!
//! let provider = AnthropicAws::builder()
//!     .region("us-west-2")
//!     .workspace_id("wrkspc_123")
//!     .api_key("sk-aws-platform-key")
//!     .build()?;
//!
//! let result = provider
//!     .language_model("claude-sonnet-4-6")
//!     .do_generate(CallOptions {
//!         prompt: vec![Message::User {
//!             content: vec![UserPart::Text(TextPart {
//!                 text: "Hi".into(),
//!                 provider_options: None,
//!             })],
//!             provider_options: None,
//!         }],
//!         max_output_tokens: Some(64),
//!         ..Default::default()
//!     })
//!     .await?;
//! println!("{result:?}");
//! # Ok(())
//! # }
//! ```
// Rust guideline compliant 2026-02-21

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod auth;
mod config;

pub use auth::{ApiKeyAuth, SigV4Auth};
pub use config::{AnthropicAws, AnthropicAwsBuilder};

/// Re-export of [`llmsdk_anthropic::tools`] for convenience.
///
/// Mirrors `anthropicAws.tools = anthropicTools` from the upstream
/// JS package — the typed Anthropic server-tool factories work identically
/// whether the request is authenticated by direct API key, OAuth token,
/// or AWS `SigV4`.
pub use llmsdk_anthropic::tools;

/// Reported provider name for Messages / Language-model handles.
pub const PROVIDER_NAME_MESSAGES: &str = "anthropic-aws.messages";

/// Reported provider name for the Files handle.
pub const PROVIDER_NAME_FILES: &str = "anthropic-aws.files";

/// Reported provider name for the Skills handle.
pub const PROVIDER_NAME_SKILLS: &str = "anthropic-aws.skills";

/// AWS service id used when generating `SigV4` signatures.
///
/// Lifted verbatim from the upstream `createSigV4FetchFunction` and matches
/// the service expected by the Claude Platform on AWS gateway.
pub const SIGV4_SERVICE: &str = "aws-external-anthropic";

/// Environment variable consulted for the AWS region.
pub const AWS_REGION_ENV_VAR: &str = "AWS_REGION";

/// Environment variable consulted for the Anthropic workspace id.
pub const WORKSPACE_ID_ENV_VAR: &str = "ANTHROPIC_AWS_WORKSPACE_ID";

/// Environment variable consulted for the AWS-provisioned API key.
pub const API_KEY_ENV_VAR: &str = "ANTHROPIC_AWS_API_KEY";

/// Base-URL template applied when no explicit `base_url` is set.
///
/// `{region}` is substituted at build time.
pub const DEFAULT_BASE_URL_TEMPLATE: &str = "https://aws-external-anthropic.{region}.api.aws/v1";

fn render_default_base_url(region: &str) -> String {
    DEFAULT_BASE_URL_TEMPLATE.replace("{region}", region)
}
