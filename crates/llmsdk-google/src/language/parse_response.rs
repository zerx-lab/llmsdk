//! Parse Gemini `generateContent` response → llmsdk `GenerateResult`.
//!
//! Mirrors the `doGenerate` branch of
//! `@ai-sdk/google/src/google-language-model.ts`. Walks `parts[]` in order
//! and emits text / reasoning / file / reasoning-file / tool-call /
//! tool-result / source content units; collects sources from
//! `groundingMetadata.groundingChunks`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{
    Content, FilePart, ReasoningPart, Source, TextPart, ToolCallPart, ToolResult, ToolResultOutput,
};
use llmsdk_provider::shared::{FileBytes, FileData, ProviderMetadata};
use serde_json::{Map, Value};

use super::wire::{WireGroundingMetadata, WirePart, WireResponse};

/// Build the [`Vec<Content>`] for the parts array.
pub(crate) fn build_content(
    parts: &[WirePart],
    grounding: Option<&WireGroundingMetadata>,
    provider_keys: &[&str],
    mut next_id: impl FnMut() -> String,
) -> (Vec<Content>, bool) {
    let mut content: Vec<Content> = Vec::new();
    let mut last_code_id: Option<String> = None;
    let mut last_server_id: Option<String> = None;
    let mut has_client_tool = false;

    for part in parts {
        if let Some(exe) = &part.executable_code {
            if !exe.code.is_empty() {
                let id = next_id();
                last_code_id = Some(id.clone());
                let input = serde_json::to_string(exe).unwrap_or_else(|_| "{}".into());
                content.push(Content::ToolCall(ToolCallPart {
                    tool_call_id: id,
                    tool_name: "code_execution".into(),
                    input: serde_json::from_str(&input).unwrap_or(Value::Null),
                    provider_executed: Some(true),
                    dynamic: None,
                    provider_options: None,
                }));
                continue;
            }
        }
        if let Some(res) = &part.code_execution_result {
            if let Some(id) = last_code_id.take() {
                let mut out = Map::new();
                out.insert("outcome".into(), Value::String(res.outcome.clone()));
                out.insert(
                    "output".into(),
                    Value::String(res.output.clone().unwrap_or_default()),
                );
                content.push(Content::ToolResult(ToolResult {
                    tool_call_id: id,
                    tool_name: "code_execution".into(),
                    output: ToolResultOutput::Json {
                        value: Value::Object(out),
                        provider_options: None,
                    },
                    provider_metadata: None,
                }));
            }
            continue;
        }
        if let Some(text) = &part.text {
            let pm = thought_sig_metadata(part.thought_signature.as_deref(), provider_keys);
            if text.is_empty() {
                // Attach thought signature to last content if any.
                if let Some(pm) = pm {
                    if !content.is_empty() {
                        let last_idx = content.len() - 1;
                        attach_metadata_to_content(&mut content[last_idx], pm);
                    }
                }
            } else if part.thought == Some(true) {
                content.push(Content::Reasoning(ReasoningPart {
                    text: text.clone(),
                    provider_options: None,
                }));
                if let Some(pm) = pm {
                    let last_idx = content.len() - 1;
                    attach_metadata_to_content(&mut content[last_idx], pm);
                }
            } else {
                content.push(Content::Text(TextPart {
                    text: text.clone(),
                    provider_options: None,
                }));
                if let Some(pm) = pm {
                    let last_idx = content.len() - 1;
                    attach_metadata_to_content(&mut content[last_idx], pm);
                }
            }
            continue;
        }
        if let Some(fc) = &part.function_call {
            if let Some(name) = &fc.name {
                has_client_tool = true;
                let id = fc.id.clone().unwrap_or_else(&mut next_id);
                content.push(Content::ToolCall(ToolCallPart {
                    tool_call_id: id,
                    tool_name: name.clone(),
                    input: fc.args.clone().unwrap_or_else(|| Value::Object(Map::new())),
                    provider_executed: None,
                    dynamic: None,
                    provider_options: thought_sig_metadata(
                        part.thought_signature.as_deref(),
                        provider_keys,
                    )
                    .map(metadata_to_options),
                }));
            }
            continue;
        }
        if let Some(inline) = &part.inline_data {
            let is_thought = part.thought == Some(true);
            let media = inline.mime_type.clone();
            let data = FileData::Data {
                data: FileBytes::Base64(inline.data.clone()),
            };
            if is_thought {
                content.push(Content::ReasoningFile {
                    data,
                    media_type: media,
                    provider_options: None,
                });
            } else {
                content.push(Content::File(FilePart {
                    filename: None,
                    data,
                    media_type: media,
                    provider_options: None,
                }));
            }
            continue;
        }
        if let Some(tc) = &part.tool_call {
            let id = if tc.id.is_empty() {
                next_id()
            } else {
                tc.id.clone()
            };
            last_server_id = Some(id.clone());
            let mut server_meta = Map::new();
            server_meta.insert("serverToolCallId".into(), Value::String(id.clone()));
            server_meta.insert("serverToolType".into(), Value::String(tc.tool_type.clone()));
            if let Some(sig) = &part.thought_signature {
                server_meta.insert("thoughtSignature".into(), Value::String(sig.clone()));
            }
            content.push(Content::ToolCall(ToolCallPart {
                tool_call_id: id,
                tool_name: format!("server:{}", tc.tool_type),
                input: tc.args.clone().unwrap_or_else(|| Value::Object(Map::new())),
                provider_executed: Some(true),
                dynamic: Some(true),
                provider_options: Some(wrap_provider_options(provider_keys, server_meta)),
            }));
            continue;
        }
        if let Some(tr) = &part.tool_response {
            let id = last_server_id.take().unwrap_or_else(|| {
                if tr.id.is_empty() {
                    next_id()
                } else {
                    tr.id.clone()
                }
            });
            let mut server_meta = Map::new();
            server_meta.insert("serverToolCallId".into(), Value::String(id.clone()));
            server_meta.insert("serverToolType".into(), Value::String(tr.tool_type.clone()));
            if let Some(sig) = &part.thought_signature {
                server_meta.insert("thoughtSignature".into(), Value::String(sig.clone()));
            }
            content.push(Content::ToolResult(ToolResult {
                tool_call_id: id,
                tool_name: format!("server:{}", tr.tool_type),
                output: ToolResultOutput::Json {
                    value: tr.response.clone().unwrap_or(Value::Object(Map::new())),
                    provider_options: None,
                },
                provider_metadata: Some(wrap_provider_metadata(provider_keys, server_meta)),
            }));
            continue;
        }
        // fileData (input-side) is not emitted as output content.
    }

    // Append sources from grounding metadata.
    if let Some(g) = grounding {
        let sources = extract_sources(g, &mut next_id);
        for s in sources {
            content.push(Content::Source(s));
        }
    }

    (content, has_client_tool)
}

/// Build the `provider_metadata.<keys[]>` payload from the structured
/// Google response.
pub(crate) fn build_provider_metadata(
    response: &WireResponse,
    provider_keys: &[&str],
) -> ProviderMetadata {
    let candidate = response.candidates.first();
    let mut payload = Map::new();
    payload.insert(
        "promptFeedback".into(),
        response
            .prompt_feedback
            .as_ref()
            .map(|p| serde_json::to_value(p).unwrap_or(Value::Null))
            .unwrap_or(Value::Null),
    );
    payload.insert(
        "groundingMetadata".into(),
        candidate
            .and_then(|c| c.grounding_metadata.as_ref())
            .map(|g| serde_json::to_value(g).unwrap_or(Value::Null))
            .unwrap_or(Value::Null),
    );
    payload.insert(
        "urlContextMetadata".into(),
        candidate
            .and_then(|c| c.url_context_metadata.clone())
            .unwrap_or(Value::Null),
    );
    payload.insert(
        "safetyRatings".into(),
        candidate
            .and_then(|c| c.safety_ratings.clone())
            .map(Value::Array)
            .unwrap_or(Value::Null),
    );
    payload.insert(
        "usageMetadata".into(),
        response
            .usage_metadata
            .as_ref()
            .map(|u| serde_json::to_value(u).unwrap_or(Value::Null))
            .unwrap_or(Value::Null),
    );
    payload.insert(
        "finishMessage".into(),
        candidate
            .and_then(|c| c.finish_message.clone())
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    payload.insert(
        "serviceTier".into(),
        response
            .usage_metadata
            .as_ref()
            .and_then(|u| u.service_tier.clone())
            .map(Value::String)
            .unwrap_or(Value::Null),
    );

    wrap_provider_metadata(provider_keys, payload)
}

fn wrap_provider_options(
    keys: &[&str],
    payload: Map<String, Value>,
) -> llmsdk_provider::shared::ProviderOptions {
    let mut out = llmsdk_provider::shared::ProviderOptions::new();
    for k in keys {
        out.insert((*k).to_owned(), payload.clone());
    }
    out
}

fn wrap_provider_metadata(keys: &[&str], payload: Map<String, Value>) -> ProviderMetadata {
    let mut out = ProviderMetadata::new();
    for k in keys {
        out.insert((*k).to_owned(), payload.clone());
    }
    out
}

fn thought_sig_metadata(sig: Option<&str>, keys: &[&str]) -> Option<ProviderMetadata> {
    let sig = sig?;
    let mut payload = Map::new();
    payload.insert("thoughtSignature".into(), Value::String(sig.to_owned()));
    Some(wrap_provider_metadata(keys, payload))
}

fn metadata_to_options(m: ProviderMetadata) -> llmsdk_provider::shared::ProviderOptions {
    let mut out = llmsdk_provider::shared::ProviderOptions::new();
    for (k, v) in m {
        out.insert(k, v);
    }
    out
}

fn attach_metadata_to_content(c: &mut Content, m: ProviderMetadata) {
    match c {
        Content::Text(t) => {
            t.provider_options = Some(metadata_to_options(m));
        }
        Content::Reasoning(r) => {
            r.provider_options = Some(metadata_to_options(m));
        }
        Content::File(f) => {
            f.provider_options = Some(metadata_to_options(m));
        }
        Content::ReasoningFile {
            provider_options, ..
        } => {
            *provider_options = Some(metadata_to_options(m));
        }
        Content::ToolCall(tc) => {
            tc.provider_options = Some(metadata_to_options(m));
        }
        Content::ToolResult(tr) => {
            tr.provider_metadata = Some(m);
        }
        _ => {}
    }
}

fn extract_sources(
    grounding: &WireGroundingMetadata,
    next_id: &mut impl FnMut() -> String,
) -> Vec<Source> {
    let mut out = Vec::new();
    let Some(chunks) = grounding.grounding_chunks.as_ref() else {
        return out;
    };
    for chunk in chunks {
        if let Some(w) = &chunk.web {
            out.push(Source::Url {
                id: next_id(),
                url: w.uri.clone(),
                title: w.title.clone(),
                provider_metadata: None,
            });
        } else if let Some(img) = &chunk.image {
            out.push(Source::Url {
                id: next_id(),
                url: img.source_uri.clone(),
                title: img.title.clone(),
                provider_metadata: None,
            });
        } else if let Some(r) = &chunk.retrieved_context {
            let uri_opt = r.uri.as_deref();
            if let Some(uri) = uri_opt {
                if uri.starts_with("http://") || uri.starts_with("https://") {
                    out.push(Source::Url {
                        id: next_id(),
                        url: uri.into(),
                        title: r.title.clone(),
                        provider_metadata: None,
                    });
                } else {
                    let (media, filename) = guess_media_type(uri);
                    out.push(Source::Document {
                        id: next_id(),
                        media_type: media,
                        title: r.title.clone().unwrap_or_else(|| "Unknown Document".into()),
                        filename,
                        provider_metadata: None,
                    });
                }
            } else if let Some(fs) = &r.file_search_store {
                out.push(Source::Document {
                    id: next_id(),
                    media_type: "application/octet-stream".into(),
                    title: r.title.clone().unwrap_or_else(|| "Unknown Document".into()),
                    filename: fs.rsplit('/').next().map(str::to_owned),
                    provider_metadata: None,
                });
            }
        } else if let Some(m) = &chunk.maps {
            if let Some(u) = &m.uri {
                out.push(Source::Url {
                    id: next_id(),
                    url: u.clone(),
                    title: m.title.clone(),
                    provider_metadata: None,
                });
            }
        }
    }
    out
}

fn guess_media_type(uri: &str) -> (String, Option<String>) {
    let filename = uri.rsplit('/').next().map(str::to_owned);
    if uri.ends_with(".pdf") {
        ("application/pdf".into(), filename)
    } else if uri.ends_with(".txt") {
        ("text/plain".into(), filename)
    } else if uri.ends_with(".docx") {
        (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document".into(),
            filename,
        )
    } else if uri.ends_with(".doc") {
        ("application/msword".into(), filename)
    } else if uri.ends_with(".md") || uri.ends_with(".markdown") {
        ("text/markdown".into(), filename)
    } else {
        ("application/octet-stream".into(), filename)
    }
}
