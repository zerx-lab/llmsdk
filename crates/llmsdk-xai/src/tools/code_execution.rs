//! `xai.code_execution` typed factory.
//!
//! Mirrors `@ai-sdk/xai/src/tool/code-execution.ts`. The wire payload has
//! no arguments; xAI's responses endpoint maps it onto the
//! `code_interpreter` tool type.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{ProviderTool, Tool};

/// Build a `xai.code_execution` provider tool with no extra arguments.
///
/// # Examples
///
/// ```
/// use llmsdk_xai::tools::code_execution;
/// let tool = code_execution();
/// // Pass `tool` into `CallOptions::tools` alongside any other tools.
/// let _ = tool;
/// ```
#[must_use]
pub fn code_execution() -> Tool {
    Tool::Provider(ProviderTool {
        id: "xai.code_execution".into(),
        name: "code_execution".into(),
        args: None,
        provider_options: None,
    })
}
