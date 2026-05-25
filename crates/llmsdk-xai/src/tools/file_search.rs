//! `xai.file_search` typed factory.
//!
//! Mirrors `@ai-sdk/xai/src/tool/file-search.ts`. `vector_store_ids` is
//! the only required field; xAI rejects calls without it.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{ProviderTool, Tool};
use serde::Serialize;

/// Required + optional knobs for [`file_search`].
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileSearchOptions {
    /// IDs of the vector stores (collections) to search across.
    pub vector_store_ids: Vec<String>,
    /// Maximum number of results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_num_results: Option<u32>,
}

/// Build a `xai.file_search` provider tool.
///
/// # Examples
///
/// ```
/// use llmsdk_xai::tools::{file_search, FileSearchOptions};
/// let tool = file_search(&FileSearchOptions {
///     vector_store_ids: vec!["vs_123".into()],
///     max_num_results: Some(10),
/// });
/// let _ = tool;
/// ```
#[must_use]
pub fn file_search(opts: &FileSearchOptions) -> Tool {
    let args = serde_json::to_value(opts)
        .ok()
        .and_then(|v| v.as_object().cloned());
    Tool::Provider(ProviderTool {
        id: "xai.file_search".into(),
        name: "file_search".into(),
        args,
        provider_options: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_store_ids_required_field_emitted() {
        let Tool::Provider(p) = file_search(&FileSearchOptions {
            vector_store_ids: vec!["vs_1".into()],
            max_num_results: Some(7),
        }) else {
            panic!("expected provider tool");
        };
        let args = p.args.unwrap();
        assert_eq!(args["vectorStoreIds"][0], "vs_1");
        assert_eq!(args["maxNumResults"], 7);
    }
}
