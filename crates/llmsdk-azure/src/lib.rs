//! Azure `OpenAI` provider for llmsdk.
//!
//! Rust port of [`@ai-sdk/azure`](https://github.com/vercel/ai/tree/main/packages/azure).
//!
//! Azure `OpenAI` speaks the same wire format as `OpenAI` itself, so this
//! crate is a thin wrapper around [`llmsdk_openai`]: it constructs an
//! [`internal::Inner`](llmsdk_openai::internal::Inner) with an Azure-flavoured
//! [`UrlStrategy`](llmsdk_openai::internal::UrlStrategy) and Azure auth
//! headers, then hands that `Inner` to the existing `OpenAI` model types.
//!
//! Two URL layouts are supported, mirroring upstream:
//!
//! - **v1 mode (default)**: `https://{resource}.openai.azure.com/openai/v1{path}?api-version={apiVersion}`
//! - **Legacy deployment mode**: `https://{resource}.openai.azure.com/openai/deployments/{deploymentId}{path}?api-version={apiVersion}`
//!
//! Toggle via [`AzureOpenAiBuilder::use_deployment_based_urls`].
//!
//! # Quick start
//!
//! ```no_run
//! # async fn doc() -> Result<(), llmsdk_provider::ProviderError> {
//! use llmsdk_azure::AzureOpenAi;
//! use llmsdk_provider::language_model::{CallOptions, Message};
//! use llmsdk_provider::LanguageModel;
//!
//! let provider = AzureOpenAi::builder()
//!     .api_key("...")
//!     .resource_name("my-resource")
//!     .build()?;
//!
//! let model = provider.chat("gpt-4o-mini-deployment");
//! let result = model
//!     .do_generate(CallOptions {
//!         prompt: vec![Message::System {
//!             content: "Be concise.".into(),
//!             provider_options: None,
//!         }],
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

mod config;
mod tools;

pub use config::{AzureOpenAi, AzureOpenAiBuilder};

/// Azure-flavoured re-export of all `OpenAI` Responses-API provider-defined
/// tool argument / output modules.
///
/// Mirrors `@ai-sdk/azure/src/azure-openai-tools.ts`. Upstream JS exposes
/// only four (`codeInterpreter`, `fileSearch`, `imageGeneration`,
/// `webSearchPreview`); Rust forwards the full set of eleven since the
/// `OpenAI` / `Azure-OpenAI` wire surfaces share the same `openai.*` tool
/// ids either way.
pub use tools::azure_openai_tools;

/// Environment variable consulted when no explicit API key is given.
pub const API_KEY_ENV_VAR: &str = "AZURE_API_KEY";

/// Environment variable consulted when no explicit resource name is given.
pub const RESOURCE_NAME_ENV_VAR: &str = "AZURE_RESOURCE_NAME";

/// Default `apiVersion` query parameter when none is configured.
///
/// Mirrors `@ai-sdk/azure` upstream: defaults to `"v1"` (the GA "`OpenAI`
/// v1 compatible" surface), **not** the legacy `"preview"` value used by
/// older SDK versions.
pub const DEFAULT_API_VERSION: &str = "v1";

/// Provider id reported by [`llmsdk_provider::LanguageModel::provider`] for
/// the chat (Chat Completions) surface.
pub const PROVIDER_ID_CHAT: &str = "azure.chat";

/// Provider id reported for the Responses API surface.
pub const PROVIDER_ID_RESPONSES: &str = "azure.responses";

/// Provider id reported for the embeddings surface.
pub const PROVIDER_ID_EMBEDDINGS: &str = "azure.embeddings";

/// Provider id reported for the image generation surface.
pub const PROVIDER_ID_IMAGE: &str = "azure.image";
