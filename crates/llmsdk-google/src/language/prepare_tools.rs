//! Convert llmsdk `[Tool]` + `ToolChoice` into Gemini wire `tools[]`
//! and `toolConfig`.
//!
//! Mirrors `@ai-sdk/google/src/google-prepare-tools.ts`. Function tools
//! and provider-defined tools (google_search / google_search_retrieval /
//! code_execution / url_context / enterprise_web_search / file_search /
//! vertex_rag_store / google_maps / ...) are translated separately.
//!
//! Returns:
//! - `tools`: wire `tools[]` array (mixed function declarations + server
//!   tools).
//! - `tool_config`: wire `toolConfig` envelope when a `toolChoice` or
//!   strict mode is set.
//! - `warnings`: capability mismatches (e.g. server tool requires Gemini 2+).
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{Tool, ToolChoice};
use llmsdk_provider::shared::Warning;
use serde_json::{Map, Value};

use crate::schema::convert_json_schema_to_openapi_nested;

/// Result of [`prepare_tools`].
#[derive(Debug, Default)]
pub(crate) struct PreparedTools {
    pub tools: Option<Value>,
    pub tool_config: Option<Value>,
    pub warnings: Vec<Warning>,
}

/// Prepare wire tools + toolConfig for a Gemini call.
pub(crate) fn prepare_tools(
    tools: Option<&[Tool]>,
    tool_choice: Option<&ToolChoice>,
    model_id: &str,
    is_vertex: bool,
) -> PreparedTools {
    let mut warnings: Vec<Warning> = Vec::new();
    let tools = match tools {
        Some(t) if !t.is_empty() => t,
        _ => {
            return PreparedTools::default();
        }
    };

    let is_latest = matches!(
        model_id,
        "gemini-flash-latest" | "gemini-flash-lite-latest" | "gemini-pro-latest"
    );
    let is_gemini2_or_newer = model_id.contains("gemini-2")
        || model_id.contains("gemini-3")
        || model_id.contains("nano-banana")
        || is_latest;
    let is_gemini3_or_newer = model_id.contains("gemini-3");
    let supports_file_search = model_id.contains("gemini-2.5") || model_id.contains("gemini-3");

    let has_function = tools.iter().any(|t| matches!(t, Tool::Function(_)));
    let has_provider = tools.iter().any(|t| matches!(t, Tool::Provider(_)));

    if has_function && has_provider && !is_gemini3_or_newer {
        warnings.push(Warning::Other {
            message: "combination of function and provider-defined tools".into(),
        });
    }

    if has_provider {
        let mut google_tools: Vec<Value> = Vec::new();
        for tool in tools {
            let Tool::Provider(p) = tool else {
                continue;
            };
            match p.id.as_str() {
                "google.google_search" => {
                    if is_gemini2_or_newer {
                        let mut o = Map::new();
                        let args_val = p
                            .args
                            .as_ref()
                            .map(|m| Value::Object(m.clone()))
                            .unwrap_or_else(|| Value::Object(Map::new()));
                        o.insert("googleSearch".into(), args_val);
                        google_tools.push(Value::Object(o));
                    } else {
                        warnings.push(Warning::UnsupportedTool {
                            tool: p.id.clone(),
                            details: Some("Google Search requires Gemini 2.0 or newer.".into()),
                        });
                    }
                }
                "google.google_search_retrieval" => {
                    let args_val = p
                        .args
                        .as_ref()
                        .map(|m| Value::Object(m.clone()))
                        .unwrap_or_else(|| Value::Object(Map::new()));
                    let mut o = Map::new();
                    o.insert("googleSearchRetrieval".into(), args_val);
                    google_tools.push(Value::Object(o));
                }
                "google.enterprise_web_search" => {
                    if is_gemini2_or_newer {
                        let mut o = Map::new();
                        o.insert("enterpriseWebSearch".into(), Value::Object(Map::new()));
                        google_tools.push(Value::Object(o));
                    } else {
                        warnings.push(Warning::UnsupportedTool {
                            tool: p.id.clone(),
                            details: Some(
                                "Enterprise Web Search requires Gemini 2.0 or newer.".into(),
                            ),
                        });
                    }
                }
                "google.url_context" => {
                    if is_gemini2_or_newer {
                        let mut o = Map::new();
                        o.insert("urlContext".into(), Value::Object(Map::new()));
                        google_tools.push(Value::Object(o));
                    } else {
                        warnings.push(Warning::UnsupportedTool {
                            tool: p.id.clone(),
                            details: Some("URL context requires Gemini 2.0 or newer.".into()),
                        });
                    }
                }
                "google.code_execution" => {
                    if is_gemini2_or_newer {
                        let mut o = Map::new();
                        o.insert("codeExecution".into(), Value::Object(Map::new()));
                        google_tools.push(Value::Object(o));
                    } else {
                        warnings.push(Warning::UnsupportedTool {
                            tool: p.id.clone(),
                            details: Some("code_execution requires Gemini 2.0 or newer.".into()),
                        });
                    }
                }
                "google.file_search" => {
                    if supports_file_search {
                        let mut o = Map::new();
                        let args_val = p
                            .args
                            .as_ref()
                            .map(|m| Value::Object(m.clone()))
                            .unwrap_or_else(|| Value::Object(Map::new()));
                        o.insert("fileSearch".into(), args_val);
                        google_tools.push(Value::Object(o));
                    } else {
                        warnings.push(Warning::UnsupportedTool {
                            tool: p.id.clone(),
                            details: Some(
                                "file_search supported only on Gemini 2.5 / 3 models.".into(),
                            ),
                        });
                    }
                }
                "google.vertex_rag_store" => {
                    if is_gemini2_or_newer {
                        let mut retrieval = Map::new();
                        let mut store = Map::new();
                        if let Some(args) = p.args.as_ref() {
                            if let Some(corpus) = args.get("ragCorpus") {
                                let mut rag_resources = Map::new();
                                rag_resources.insert("rag_corpus".into(), corpus.clone());
                                store.insert("rag_resources".into(), Value::Object(rag_resources));
                            }
                            if let Some(top_k) = args.get("topK") {
                                store.insert("similarity_top_k".into(), top_k.clone());
                            }
                        }
                        retrieval.insert("vertex_rag_store".into(), Value::Object(store));
                        let mut o = Map::new();
                        o.insert("retrieval".into(), Value::Object(retrieval));
                        google_tools.push(Value::Object(o));
                    } else {
                        warnings.push(Warning::UnsupportedTool {
                            tool: p.id.clone(),
                            details: Some("vertex_rag_store requires Gemini 2.0 or newer.".into()),
                        });
                    }
                }
                "google.google_maps" => {
                    if is_gemini2_or_newer {
                        let mut o = Map::new();
                        o.insert("googleMaps".into(), Value::Object(Map::new()));
                        google_tools.push(Value::Object(o));
                    } else {
                        warnings.push(Warning::UnsupportedTool {
                            tool: p.id.clone(),
                            details: Some("google_maps requires Gemini 2.0 or newer.".into()),
                        });
                    }
                }
                other => {
                    warnings.push(Warning::UnsupportedTool {
                        tool: other.into(),
                        details: None,
                    });
                }
            }
        }

        if has_function && is_gemini3_or_newer && !google_tools.is_empty() {
            let fn_decls = build_function_declarations(tools);
            let mut combined_tool_config = Map::new();
            let mut fcc = Map::new();
            fcc.insert("mode".into(), Value::String("VALIDATED".into()));
            combined_tool_config.insert("functionCallingConfig".into(), Value::Object(fcc));
            if !is_vertex {
                combined_tool_config
                    .insert("includeServerSideToolInvocations".into(), Value::Bool(true));
            }
            apply_tool_choice_to_function_calling_config(
                &mut combined_tool_config,
                tool_choice,
                false,
            );
            let mut all = google_tools;
            let mut fn_obj = Map::new();
            fn_obj.insert("functionDeclarations".into(), Value::Array(fn_decls));
            all.push(Value::Object(fn_obj));
            return PreparedTools {
                tools: Some(Value::Array(all)),
                tool_config: Some(Value::Object(combined_tool_config)),
                warnings,
            };
        }

        return PreparedTools {
            tools: if google_tools.is_empty() {
                None
            } else {
                Some(Value::Array(google_tools))
            },
            tool_config: None,
            warnings,
        };
    }

    // Pure function tools.
    let mut function_declarations: Vec<Value> = Vec::new();
    let mut has_strict = false;
    for tool in tools {
        match tool {
            Tool::Function(f) => {
                let mut fd = Map::new();
                fd.insert("name".into(), Value::String(f.name.clone()));
                fd.insert(
                    "description".into(),
                    Value::String(f.description.clone().unwrap_or_default()),
                );
                fd.insert(
                    "parameters".into(),
                    convert_json_schema_to_openapi_nested(
                        &serde_json::to_value(&f.input_schema).unwrap_or(Value::Null),
                    ),
                );
                function_declarations.push(Value::Object(fd));
                if f.strict == Some(true) {
                    has_strict = true;
                }
            }
            Tool::Provider(_) => unreachable!(),
        }
    }

    let mut wire_tools = Vec::with_capacity(1);
    let mut fn_obj = Map::new();
    fn_obj.insert(
        "functionDeclarations".into(),
        Value::Array(function_declarations),
    );
    wire_tools.push(Value::Object(fn_obj));

    let tool_config = build_function_tool_config(tool_choice, has_strict);

    PreparedTools {
        tools: Some(Value::Array(wire_tools)),
        tool_config,
        warnings,
    }
}

fn build_function_declarations(tools: &[Tool]) -> Vec<Value> {
    let mut out = Vec::new();
    for t in tools {
        if let Tool::Function(f) = t {
            let mut fd = Map::new();
            fd.insert("name".into(), Value::String(f.name.clone()));
            fd.insert(
                "description".into(),
                Value::String(f.description.clone().unwrap_or_default()),
            );
            fd.insert(
                "parameters".into(),
                convert_json_schema_to_openapi_nested(
                    &serde_json::to_value(&f.input_schema).unwrap_or(Value::Null),
                ),
            );
            out.push(Value::Object(fd));
        }
    }
    out
}

fn build_function_tool_config(tool_choice: Option<&ToolChoice>, has_strict: bool) -> Option<Value> {
    let tc = match tool_choice {
        Some(tc) => tc,
        None => {
            if has_strict {
                let mut o = Map::new();
                let mut fcc = Map::new();
                fcc.insert("mode".into(), Value::String("VALIDATED".into()));
                o.insert("functionCallingConfig".into(), Value::Object(fcc));
                return Some(Value::Object(o));
            }
            return None;
        }
    };
    let mut o = Map::new();
    let mut fcc = Map::new();
    match tc {
        ToolChoice::Auto => {
            fcc.insert(
                "mode".into(),
                Value::String(if has_strict { "VALIDATED" } else { "AUTO" }.into()),
            );
        }
        ToolChoice::None => {
            fcc.insert("mode".into(), Value::String("NONE".into()));
        }
        ToolChoice::Required => {
            fcc.insert(
                "mode".into(),
                Value::String(if has_strict { "VALIDATED" } else { "ANY" }.into()),
            );
        }
        ToolChoice::Tool { tool_name } => {
            fcc.insert(
                "mode".into(),
                Value::String(if has_strict { "VALIDATED" } else { "ANY" }.into()),
            );
            fcc.insert(
                "allowedFunctionNames".into(),
                Value::Array(vec![Value::String(tool_name.clone())]),
            );
        }
    }
    o.insert("functionCallingConfig".into(), Value::Object(fcc));
    Some(Value::Object(o))
}

fn apply_tool_choice_to_function_calling_config(
    combined: &mut Map<String, Value>,
    tool_choice: Option<&ToolChoice>,
    _has_strict: bool,
) {
    let Some(tc) = tool_choice else {
        return;
    };
    let mut fcc = Map::new();
    match tc {
        ToolChoice::Auto => return, // keep default
        ToolChoice::None => {
            fcc.insert("mode".into(), Value::String("NONE".into()));
        }
        ToolChoice::Required => {
            fcc.insert("mode".into(), Value::String("ANY".into()));
        }
        ToolChoice::Tool { tool_name } => {
            fcc.insert("mode".into(), Value::String("ANY".into()));
            fcc.insert(
                "allowedFunctionNames".into(),
                Value::Array(vec![Value::String(tool_name.clone())]),
            );
        }
    }
    combined.insert("functionCallingConfig".into(), Value::Object(fcc));
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::language_model::{FunctionTool, ProviderTool};
    use serde_json::json;

    fn fn_tool() -> Tool {
        Tool::Function(FunctionTool {
            name: "getWeather".into(),
            description: Some("Look up weather".into()),
            input_schema: serde_json::from_value(json!({
                "type":"object",
                "properties":{"city":{"type":"string"}},
                "required":["city"]
            }))
            .unwrap(),
            input_examples: None,
            strict: None,
            provider_options: None,
        })
    }

    #[test]
    fn function_tool_default() {
        let r = prepare_tools(Some(&[fn_tool()]), None, "gemini-2.5-flash", false);
        let arr = r.tools.as_ref().unwrap().as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["functionDeclarations"][0]["name"], "getWeather");
        assert!(r.tool_config.is_none());
    }

    #[test]
    fn tool_choice_required() {
        let r = prepare_tools(
            Some(&[fn_tool()]),
            Some(&ToolChoice::Required),
            "gemini-2.5-flash",
            false,
        );
        assert_eq!(
            r.tool_config.unwrap()["functionCallingConfig"]["mode"],
            "ANY"
        );
    }

    #[test]
    fn tool_choice_tool_specific() {
        let r = prepare_tools(
            Some(&[fn_tool()]),
            Some(&ToolChoice::Tool {
                tool_name: "getWeather".into(),
            }),
            "gemini-2.5-flash",
            false,
        );
        let tc = r.tool_config.unwrap();
        assert_eq!(tc["functionCallingConfig"]["mode"], "ANY");
        assert_eq!(
            tc["functionCallingConfig"]["allowedFunctionNames"],
            json!(["getWeather"])
        );
    }

    #[test]
    fn provider_tool_google_search() {
        let tool = Tool::Provider(ProviderTool {
            id: "google.google_search".into(),
            name: "google_search".into(),
            args: None,
            provider_options: None,
        });
        let r = prepare_tools(Some(&[tool]), None, "gemini-2.5-flash", false);
        let arr = r.tools.as_ref().unwrap().as_array().unwrap();
        assert!(arr[0].get("googleSearch").is_some());
    }

    #[test]
    fn provider_tool_unknown_warns() {
        let tool = Tool::Provider(ProviderTool {
            id: "google.unknown".into(),
            name: "x".into(),
            args: None,
            provider_options: None,
        });
        let r = prepare_tools(Some(&[tool]), None, "gemini-2.5-flash", false);
        assert!(r.tools.is_none());
        assert_eq!(r.warnings.len(), 1);
    }

    #[test]
    fn code_execution_pre_gemini_2_warns() {
        let tool = Tool::Provider(ProviderTool {
            id: "google.code_execution".into(),
            name: "code_execution".into(),
            args: None,
            provider_options: None,
        });
        let r = prepare_tools(Some(&[tool]), None, "gemini-1.5-flash", false);
        assert!(r.tools.is_none());
        assert_eq!(r.warnings.len(), 1);
    }
}
