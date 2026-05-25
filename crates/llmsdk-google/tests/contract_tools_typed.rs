//! Sanity tests for the typed tool factory module.
// Rust guideline compliant 2026-05-25

use llmsdk_google::tools;
use llmsdk_provider::language_model::Tool;

fn assert_id_name(t: Tool, id: &str, name: &str) {
    let Tool::Provider(p) = t else {
        panic!("expected provider tool");
    };
    assert_eq!(p.id, id);
    assert_eq!(p.name, name);
}

#[test]
fn all_no_args_factories() {
    assert_id_name(
        tools::enterprise_web_search(),
        "google.enterprise_web_search",
        "enterprise_web_search",
    );
    assert_id_name(
        tools::code_execution(),
        "google.code_execution",
        "code_execution",
    );
    assert_id_name(tools::url_context(), "google.url_context", "url_context");
    assert_id_name(tools::google_maps(), "google.google_maps", "google_maps");
}

#[test]
fn google_search_serializes() {
    let t = tools::google_search(tools::GoogleSearchArgs {
        time_range_filter: Some(tools::GoogleSearchTimeRange {
            start_time: "2024-01-01T00:00:00Z".into(),
            end_time: "2024-12-31T00:00:00Z".into(),
        }),
        ..Default::default()
    });
    let Tool::Provider(p) = t else { panic!() };
    let args = p.args.unwrap();
    let trf = args.get("timeRangeFilter").unwrap();
    assert_eq!(trf["startTime"], "2024-01-01T00:00:00Z");
}

#[test]
fn vertex_rag_args() {
    let t = tools::vertex_rag_store(tools::VertexRagStoreArgs {
        rag_corpus: "corpora/abc".into(),
        top_k: Some(7),
    });
    let Tool::Provider(p) = t else { panic!() };
    let args = p.args.unwrap();
    assert_eq!(args["ragCorpus"], "corpora/abc");
    assert_eq!(args["topK"], 7);
}

#[test]
fn file_search_args() {
    let t = tools::file_search(tools::FileSearchArgs {
        file_search_store_names: vec!["fs/1".into(), "fs/2".into()],
        top_k: Some(3),
        metadata_filter: Some("year > 2020".into()),
    });
    let Tool::Provider(p) = t else { panic!() };
    let args = p.args.unwrap();
    assert_eq!(args["fileSearchStoreNames"].as_array().unwrap().len(), 2);
    assert_eq!(args["topK"], 3);
}
