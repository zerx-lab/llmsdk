//! Extract `Source` parts from Interactions response data.
//!
//! Mirrors `@ai-sdk/google/src/interactions/extract-google-interactions-sources.ts`.
//! Two entry points:
//!
//! - [`annotation_to_source`]: maps a single text-block annotation
//!   (`url_citation` / `file_citation` / `place_citation`).
//! - [`builtin_tool_result_to_sources`]: maps the `result` payload of a
//!   built-in tool result step (`url_context_result` / `google_search_result`
//!   / `google_maps_result` / `file_search_result`) into zero or more sources.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::Source;
use serde_json::Value as JsonValue;

/// Returns a `mime_type` guess for a document URI / filename based on the
/// extension. Mirrors upstream `inferDocMediaType`.
fn infer_doc_media_type(uri_or_name: &str) -> &'static str {
    let lower = uri_or_name.to_ascii_lowercase();
    if lower.ends_with(".pdf") {
        "application/pdf"
    } else if lower.ends_with(".txt") {
        "text/plain"
    } else if lower.ends_with(".md") || lower.ends_with(".markdown") {
        "text/markdown"
    } else if lower.ends_with(".doc") {
        "application/msword"
    } else if lower.ends_with(".docx") {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    } else {
        "application/octet-stream"
    }
}

fn basename(uri_or_name: &str) -> Option<String> {
    let last = uri_or_name.rsplit('/').next()?;
    if last.is_empty() {
        None
    } else {
        Some(last.to_owned())
    }
}

fn next_id<F: FnMut() -> String>(gen_id: &mut F) -> String {
    gen_id()
}

/// Map a single text-block annotation to a [`Source`]. Returns `None` when
/// the annotation lacks the minimum payload to form a source (e.g. a URL
/// citation without a `url`).
///
/// Recognized annotation types:
/// - `url_citation` (web sources)
/// - `file_citation` (file references, falls back to document when not http)
/// - `place_citation` (Maps grounding)
pub(crate) fn annotation_to_source<F: FnMut() -> String>(
    annotation: &JsonValue,
    gen_id: &mut F,
) -> Option<Source> {
    let kind = annotation.get("type").and_then(JsonValue::as_str)?;
    match kind {
        "url_citation" => {
            let url = annotation
                .get("url")
                .and_then(JsonValue::as_str)
                .filter(|s| !s.is_empty())?
                .to_owned();
            let title = annotation
                .get("title")
                .and_then(JsonValue::as_str)
                .map(str::to_owned);
            Some(Source::Url {
                id: next_id(gen_id),
                url,
                title,
                provider_metadata: None,
            })
        }
        "file_citation" => {
            let uri = annotation
                .get("url")
                .or_else(|| annotation.get("document_uri"))
                .or_else(|| annotation.get("file_name"))
                .and_then(JsonValue::as_str)
                .filter(|s| !s.is_empty())?
                .to_owned();
            let original_name = annotation
                .get("file_name")
                .and_then(JsonValue::as_str)
                .map(str::to_owned);
            if uri.starts_with("http://") || uri.starts_with("https://") {
                return Some(Source::Url {
                    id: next_id(gen_id),
                    url: uri,
                    title: original_name,
                    provider_metadata: None,
                });
            }
            let resolved_filename = original_name.clone().or_else(|| basename(&uri));
            let media_type = infer_doc_media_type(&uri).to_owned();
            let title = original_name
                .clone()
                .or_else(|| resolved_filename.clone())
                .unwrap_or_else(|| uri.clone());
            Some(Source::Document {
                id: next_id(gen_id),
                media_type,
                title,
                filename: resolved_filename,
                provider_metadata: None,
            })
        }
        "place_citation" => {
            let url = annotation
                .get("url")
                .and_then(JsonValue::as_str)
                .filter(|s| !s.is_empty())?
                .to_owned();
            let title = annotation
                .get("name")
                .and_then(JsonValue::as_str)
                .map(str::to_owned);
            Some(Source::Url {
                id: next_id(gen_id),
                url,
                title,
                provider_metadata: None,
            })
        }
        _ => None,
    }
}

/// Maps a list of annotations attached to one text block into sources,
/// de-duplicated by URL/filename so the same citation reappearing across
/// deltas only surfaces once.
pub(crate) fn annotations_to_sources<F: FnMut() -> String>(
    annotations: Option<&JsonValue>,
    gen_id: &mut F,
    seen: &mut std::collections::HashSet<String>,
) -> Vec<Source> {
    let Some(list) = annotations.and_then(JsonValue::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for ann in list {
        let Some(src) = annotation_to_source(ann, gen_id) else {
            continue;
        };
        let key = source_key(&src);
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);
        out.push(src);
    }
    out
}

/// Maps a built-in tool *result* step's payload to zero or more sources.
/// Supported result kinds:
///
/// - `url_context_result` → URL sources for each entry whose `status` is
///   `success` (or unset) and `url` is non-empty.
/// - `google_search_result` → URL sources for each entry with a `url` (the
///   `search_suggestions` shape is skipped — it carries HTML widgets, not
///   citations).
/// - `google_maps_result` → URL sources for each place with a `url`.
/// - `file_search_result` → URL or document sources (entries with http(s)
///   URIs become URL sources; everything else becomes a document source with
///   inferred media type).
pub(crate) fn builtin_tool_result_to_sources<F: FnMut() -> String>(
    step_type: &str,
    result: Option<&JsonValue>,
    gen_id: &mut F,
) -> Vec<Source> {
    let Some(arr) = result.and_then(JsonValue::as_array) else {
        return Vec::new();
    };
    let mut sources = Vec::new();
    match step_type {
        "url_context_result" => {
            for entry in arr {
                let Some(url) = entry
                    .get("url")
                    .and_then(JsonValue::as_str)
                    .filter(|s| !s.is_empty())
                else {
                    continue;
                };
                let status = entry.get("status").and_then(JsonValue::as_str);
                if status.is_some_and(|s| s != "success") {
                    continue;
                }
                sources.push(Source::Url {
                    id: next_id(gen_id),
                    url: url.to_owned(),
                    title: None,
                    provider_metadata: None,
                });
            }
        }
        "google_search_result" => {
            for entry in arr {
                let Some(url) = entry
                    .get("url")
                    .and_then(JsonValue::as_str)
                    .filter(|s| !s.is_empty())
                else {
                    continue;
                };
                let title = entry
                    .get("title")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned);
                sources.push(Source::Url {
                    id: next_id(gen_id),
                    url: url.to_owned(),
                    title,
                    provider_metadata: None,
                });
            }
        }
        "google_maps_result" => {
            for entry in arr {
                let Some(places) = entry.get("places").and_then(JsonValue::as_array) else {
                    continue;
                };
                for place in places {
                    let Some(url) = place
                        .get("url")
                        .and_then(JsonValue::as_str)
                        .filter(|s| !s.is_empty())
                    else {
                        continue;
                    };
                    let title = place
                        .get("name")
                        .and_then(JsonValue::as_str)
                        .map(str::to_owned);
                    sources.push(Source::Url {
                        id: next_id(gen_id),
                        url: url.to_owned(),
                        title,
                        provider_metadata: None,
                    });
                }
            }
        }
        "file_search_result" => {
            for entry in arr {
                if !entry.is_object() {
                    continue;
                }
                let uri = entry
                    .get("url")
                    .or_else(|| entry.get("document_uri"))
                    .or_else(|| entry.get("file_name"))
                    .and_then(JsonValue::as_str)
                    .filter(|s| !s.is_empty());
                let Some(uri) = uri else {
                    continue;
                };
                let title = entry
                    .get("title")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned);
                if uri.starts_with("http://") || uri.starts_with("https://") {
                    sources.push(Source::Url {
                        id: next_id(gen_id),
                        url: uri.to_owned(),
                        title,
                        provider_metadata: None,
                    });
                    continue;
                }
                let original_name = entry
                    .get("file_name")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned);
                let resolved_filename = original_name.clone().or_else(|| basename(uri));
                let final_title = title
                    .or(original_name)
                    .or_else(|| resolved_filename.clone())
                    .unwrap_or_else(|| uri.to_owned());
                sources.push(Source::Document {
                    id: next_id(gen_id),
                    media_type: infer_doc_media_type(uri).to_owned(),
                    title: final_title,
                    filename: resolved_filename,
                    provider_metadata: None,
                });
            }
        }
        _ => {}
    }
    sources
}

pub(crate) fn source_key(source: &Source) -> String {
    match source {
        Source::Url { url, .. } => format!("url:{url}"),
        Source::Document {
            filename, title, ..
        } => format!("doc:{}", filename.clone().unwrap_or_else(|| title.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn id_gen() -> impl FnMut() -> String {
        let mut n = 0;
        move || {
            n += 1;
            format!("src-{n}")
        }
    }

    #[test]
    fn url_citation_to_url_source() {
        let mut gen_id = id_gen();
        let src = annotation_to_source(
            &json!({"type": "url_citation", "url": "x", "title": "X"}),
            &mut gen_id,
        )
        .unwrap();
        match src {
            Source::Url { url, title, .. } => {
                assert_eq!(url, "x");
                assert_eq!(title.as_deref(), Some("X"));
            }
            Source::Document { .. } => panic!("expected url source"),
        }
    }

    #[test]
    fn url_citation_drops_empty_url() {
        let mut gen_id = id_gen();
        let src = annotation_to_source(&json!({"type": "url_citation", "url": ""}), &mut gen_id);
        assert!(src.is_none());
    }

    #[test]
    fn file_citation_http_routes_to_url() {
        let mut gen_id = id_gen();
        let src = annotation_to_source(
            &json!({"type": "file_citation", "url": "https://x", "file_name": "a"}),
            &mut gen_id,
        )
        .unwrap();
        match src {
            Source::Url { url, title, .. } => {
                assert_eq!(url, "https://x");
                assert_eq!(title.as_deref(), Some("a"));
            }
            Source::Document { .. } => panic!("expected url source"),
        }
    }

    #[test]
    fn file_citation_local_routes_to_document() {
        let mut gen_id = id_gen();
        let src = annotation_to_source(
            &json!({"type": "file_citation", "file_name": "report.pdf"}),
            &mut gen_id,
        )
        .unwrap();
        match src {
            Source::Document {
                media_type,
                filename,
                ..
            } => {
                assert_eq!(media_type, "application/pdf");
                assert_eq!(filename.as_deref(), Some("report.pdf"));
            }
            Source::Url { .. } => panic!("expected document source"),
        }
    }

    #[test]
    fn place_citation_to_url_source() {
        let mut gen_id = id_gen();
        let src = annotation_to_source(
            &json!({"type": "place_citation", "url": "https://maps", "name": "Place"}),
            &mut gen_id,
        )
        .unwrap();
        match src {
            Source::Url { title, .. } => assert_eq!(title.as_deref(), Some("Place")),
            Source::Document { .. } => panic!("expected url source"),
        }
    }

    #[test]
    fn google_search_result_extracts_urls() {
        let mut gen_id = id_gen();
        let result = json!([
            {"url": "https://a", "title": "A"},
            {"url": "https://b"},
            {"search_suggestions": "<html/>"}
        ]);
        let sources =
            builtin_tool_result_to_sources("google_search_result", Some(&result), &mut gen_id);
        assert_eq!(sources.len(), 2);
    }

    #[test]
    fn url_context_result_filters_non_success() {
        let mut gen_id = id_gen();
        let result = json!([
            {"url": "https://ok", "status": "success"},
            {"url": "https://err", "status": "error"},
            {"url": "https://default"}
        ]);
        let sources =
            builtin_tool_result_to_sources("url_context_result", Some(&result), &mut gen_id);
        assert_eq!(sources.len(), 2); // success + unset (default to success)
    }

    #[test]
    fn google_maps_result_extracts_places() {
        let mut gen_id = id_gen();
        let result = json!([
            {"places": [{"name": "P1", "url": "https://p1"}, {"name": "no_url"}]},
            {"places": [{"name": "P2", "url": "https://p2"}]},
        ]);
        let sources =
            builtin_tool_result_to_sources("google_maps_result", Some(&result), &mut gen_id);
        assert_eq!(sources.len(), 2);
    }

    #[test]
    fn file_search_result_routes_http_vs_local() {
        let mut gen_id = id_gen();
        let result = json!([
            {"url": "https://web", "title": "T"},
            {"file_name": "doc.docx"}
        ]);
        let sources =
            builtin_tool_result_to_sources("file_search_result", Some(&result), &mut gen_id);
        assert_eq!(sources.len(), 2);
        assert!(matches!(sources[0], Source::Url { .. }));
        assert!(matches!(sources[1], Source::Document { .. }));
    }
}
