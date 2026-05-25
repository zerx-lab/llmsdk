//! `xai.view_x_video` typed factory.
//!
//! Mirrors `@ai-sdk/xai/src/tool/view-x-video.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{ProviderTool, Tool};

/// Build a `xai.view_x_video` provider tool with no extra arguments.
///
/// # Examples
///
/// ```
/// use llmsdk_xai::tools::view_x_video;
/// let tool = view_x_video();
/// let _ = tool;
/// ```
#[must_use]
pub fn view_x_video() -> Tool {
    Tool::Provider(ProviderTool {
        id: "xai.view_x_video".into(),
        name: "view_x_video".into(),
        args: None,
        provider_options: None,
    })
}
