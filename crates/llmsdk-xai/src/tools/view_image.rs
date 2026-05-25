//! `xai.view_image` typed factory.
//!
//! Mirrors `@ai-sdk/xai/src/tool/view-image.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{ProviderTool, Tool};

/// Build a `xai.view_image` provider tool with no extra arguments.
///
/// # Examples
///
/// ```
/// use llmsdk_xai::tools::view_image;
/// let tool = view_image();
/// let _ = tool;
/// ```
#[must_use]
pub fn view_image() -> Tool {
    Tool::Provider(ProviderTool {
        id: "xai.view_image".into(),
        name: "view_image".into(),
        args: None,
        provider_options: None,
    })
}
