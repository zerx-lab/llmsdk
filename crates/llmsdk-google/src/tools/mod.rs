//! Typed factories for Google-defined provider tools.
//!
//! Mirrors `@ai-sdk/google/src/google-tools.ts` + the per-tool files under
//! `@ai-sdk/google/src/tool/`. Each factory returns a
//! [`llmsdk_provider::language_model::Tool::Provider`] keyed by a
//! `google.*` id; the wiring is handled in
//! `crate::language::prepare_tools::prepare_tools`.
//!
//! # Coverage (8 tools)
//!
//! - [`google_search`] / [`google_search_retrieval`]
//! - [`enterprise_web_search`]
//! - [`code_execution`]
//! - [`url_context`]
//! - [`file_search`]
//! - [`google_maps`]
//! - [`vertex_rag_store`]
// Rust guideline compliant 2026-05-25

use llmsdk_provider::json::JsonObject;
use llmsdk_provider::language_model::{ProviderTool, Tool};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

fn provider_tool(id: &str, name: &str, args: Option<JsonObject>) -> Tool {
    Tool::Provider(ProviderTool {
        id: id.into(),
        name: name.into(),
        args,
        provider_options: None,
    })
}

fn obj_from<T: Serialize>(v: &T) -> Option<JsonObject> {
    serde_json::to_value(v).ok().and_then(|x| match x {
        Value::Object(m) => Some(m),
        _ => None,
    })
}

/// Arguments for [`google_search`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GoogleSearchArgs {
    /// Optional search-type modifiers.
    #[serde(
        default,
        rename = "searchTypes",
        skip_serializing_if = "Option::is_none"
    )]
    pub search_types: Option<GoogleSearchTypes>,
    /// Optional ISO-8601 time range filter.
    #[serde(
        default,
        rename = "timeRangeFilter",
        skip_serializing_if = "Option::is_none"
    )]
    pub time_range_filter: Option<GoogleSearchTimeRange>,
}

/// Search-type modifiers for [`GoogleSearchArgs`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GoogleSearchTypes {
    /// Enable web search.
    #[serde(default, rename = "webSearch", skip_serializing_if = "Option::is_none")]
    pub web_search: Option<Map<String, Value>>,
    /// Enable image search.
    #[serde(
        default,
        rename = "imageSearch",
        skip_serializing_if = "Option::is_none"
    )]
    pub image_search: Option<Map<String, Value>>,
}

/// Time-range filter for [`GoogleSearchArgs`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GoogleSearchTimeRange {
    /// ISO-8601 start time.
    #[serde(rename = "startTime")]
    pub start_time: String,
    /// ISO-8601 end time.
    #[serde(rename = "endTime")]
    pub end_time: String,
}

/// Google Search grounding tool.
#[must_use]
pub fn google_search(args: GoogleSearchArgs) -> Tool {
    provider_tool("google.google_search", "google_search", obj_from(&args))
}

/// Legacy `googleSearchRetrieval` tool (pre-Gemini 2 grounding).
#[must_use]
pub fn google_search_retrieval(args: serde_json::Value) -> Tool {
    let args_obj = match args {
        Value::Object(m) => Some(m),
        _ => None,
    };
    provider_tool(
        "google.google_search_retrieval",
        "google_search_retrieval",
        args_obj,
    )
}

/// Enterprise web search (Vertex-only grounding).
#[must_use]
pub fn enterprise_web_search() -> Tool {
    provider_tool(
        "google.enterprise_web_search",
        "enterprise_web_search",
        None,
    )
}

/// Code-execution tool.
#[must_use]
pub fn code_execution() -> Tool {
    provider_tool("google.code_execution", "code_execution", None)
}

/// URL-context tool (grounding from URLs already in the prompt).
#[must_use]
pub fn url_context() -> Tool {
    provider_tool("google.url_context", "url_context", None)
}

/// Arguments for [`file_search`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSearchArgs {
    /// File-search store resource names.
    #[serde(rename = "fileSearchStoreNames")]
    pub file_search_store_names: Vec<String>,
    /// Top-K chunks to retrieve.
    #[serde(default, rename = "topK", skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// AIP-160 metadata filter.
    #[serde(
        default,
        rename = "metadataFilter",
        skip_serializing_if = "Option::is_none"
    )]
    pub metadata_filter: Option<String>,
}

/// File Search (RAG) tool.
#[must_use]
pub fn file_search(args: FileSearchArgs) -> Tool {
    provider_tool("google.file_search", "file_search", obj_from(&args))
}

/// Google Maps grounding tool.
#[must_use]
pub fn google_maps() -> Tool {
    provider_tool("google.google_maps", "google_maps", None)
}

/// Arguments for [`vertex_rag_store`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexRagStoreArgs {
    /// Fully-qualified RagCorpus resource name.
    #[serde(rename = "ragCorpus")]
    pub rag_corpus: String,
    /// Top-K contexts to retrieve.
    #[serde(default, rename = "topK", skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
}

/// Vertex RAG Store tool (Vertex AI only).
#[must_use]
pub fn vertex_rag_store(args: VertexRagStoreArgs) -> Tool {
    provider_tool(
        "google.vertex_rag_store",
        "vertex_rag_store",
        obj_from(&args),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn google_search_factory() {
        let t = google_search(GoogleSearchArgs {
            search_types: Some(GoogleSearchTypes {
                web_search: Some(Map::new()),
                ..Default::default()
            }),
            time_range_filter: None,
        });
        let Tool::Provider(p) = t else { panic!() };
        assert_eq!(p.id, "google.google_search");
        assert_eq!(p.name, "google_search");
        assert!(p.args.unwrap().contains_key("searchTypes"));
    }

    #[test]
    fn file_search_factory() {
        let t = file_search(FileSearchArgs {
            file_search_store_names: vec!["stores/x".into()],
            top_k: Some(3),
            metadata_filter: None,
        });
        let Tool::Provider(p) = t else { panic!() };
        assert_eq!(p.id, "google.file_search");
        assert_eq!(
            p.args.unwrap().get("fileSearchStoreNames").unwrap()[0],
            "stores/x"
        );
    }

    #[test]
    fn vertex_rag_store_factory() {
        let t = vertex_rag_store(VertexRagStoreArgs {
            rag_corpus: "projects/p/locations/l/ragCorpora/x".into(),
            top_k: Some(5),
        });
        let Tool::Provider(p) = t else { panic!() };
        assert_eq!(p.id, "google.vertex_rag_store");
        let args = p.args.unwrap();
        assert_eq!(args["ragCorpus"], "projects/p/locations/l/ragCorpora/x");
        assert_eq!(args["topK"], 5);
    }

    #[test]
    fn no_args_factories() {
        for (factory, expected_id, expected_name) in [
            (
                enterprise_web_search as fn() -> Tool,
                "google.enterprise_web_search",
                "enterprise_web_search",
            ),
            (code_execution, "google.code_execution", "code_execution"),
            (url_context, "google.url_context", "url_context"),
            (google_maps, "google.google_maps", "google_maps"),
        ] {
            let t = factory();
            let Tool::Provider(p) = t else { panic!() };
            assert_eq!(p.id, expected_id);
            assert_eq!(p.name, expected_name);
            assert!(p.args.is_none());
        }
    }
}
