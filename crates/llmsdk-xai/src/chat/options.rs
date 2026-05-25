//! Parse the `xai` slot of [`ProviderOptions`] into typed fields.
//!
//! Mirrors `xaiLanguageModelChatOptions` from
//! `@ai-sdk/xai/src/xai-chat-language-model-options.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::shared::ProviderOptions;
use serde::Deserialize;

use super::wire::WireSearchParameters;

/// Typed view of `provider_options["xai"]`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct XaiChatOptions {
    /// `reasoningEffort`: `none` / `low` / `medium` / `high`.
    pub reasoning_effort: Option<String>,
    /// `logprobs`: enable token-level logprobs.
    pub logprobs: Option<bool>,
    /// `topLogprobs`: number of top-N alternates per token (0-8).
    pub top_logprobs: Option<u32>,
    /// `parallel_function_calling`: allow parallel tool calls. Defaults true.
    #[serde(rename = "parallel_function_calling")]
    pub parallel_function_calling: Option<bool>,
    /// `searchParameters`: xAI Live Search configuration.
    pub search_parameters: Option<SearchParameters>,
}

/// `searchParameters` provider-options shape.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SearchParameters {
    pub mode: String,
    pub return_citations: Option<bool>,
    pub from_date: Option<String>,
    pub to_date: Option<String>,
    pub max_search_results: Option<u32>,
    pub sources: Option<Vec<SearchSource>>,
}

/// One source entry inside `searchParameters.sources[]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub(crate) enum SearchSource {
    Web {
        country: Option<String>,
        #[serde(rename = "excludedWebsites")]
        excluded_websites: Option<Vec<String>>,
        #[serde(rename = "allowedWebsites")]
        allowed_websites: Option<Vec<String>>,
        #[serde(rename = "safeSearch")]
        safe_search: Option<bool>,
    },
    X {
        #[serde(rename = "excludedXHandles")]
        excluded_x_handles: Option<Vec<String>>,
        #[serde(rename = "includedXHandles")]
        included_x_handles: Option<Vec<String>>,
        #[serde(rename = "postFavoriteCount")]
        post_favorite_count: Option<u32>,
        #[serde(rename = "postViewCount")]
        post_view_count: Option<u32>,
        /// Deprecated upstream alias of `includedXHandles`.
        #[serde(rename = "xHandles")]
        x_handles_legacy: Option<Vec<String>>,
    },
    News {
        country: Option<String>,
        #[serde(rename = "excludedWebsites")]
        excluded_websites: Option<Vec<String>>,
        #[serde(rename = "safeSearch")]
        safe_search: Option<bool>,
    },
    Rss {
        links: Vec<String>,
    },
}

impl SearchParameters {
    /// Map to wire format with `snake_case` fields.
    pub(crate) fn to_wire(&self) -> WireSearchParameters {
        WireSearchParameters {
            mode: self.mode.clone(),
            return_citations: self.return_citations,
            from_date: self.from_date.clone(),
            to_date: self.to_date.clone(),
            max_search_results: self.max_search_results,
            sources: self
                .sources
                .as_ref()
                .map(|srcs| srcs.iter().map(SearchSource::to_wire).collect()),
        }
    }
}

impl SearchSource {
    fn to_wire(&self) -> super::wire::WireSearchSource {
        use super::wire::WireSearchSource;
        match self {
            Self::Web {
                country,
                excluded_websites,
                allowed_websites,
                safe_search,
            } => WireSearchSource::Web {
                country: country.clone(),
                excluded_websites: excluded_websites.clone(),
                allowed_websites: allowed_websites.clone(),
                safe_search: *safe_search,
            },
            Self::X {
                excluded_x_handles,
                included_x_handles,
                post_favorite_count,
                post_view_count,
                x_handles_legacy,
            } => WireSearchSource::X {
                excluded_x_handles: excluded_x_handles.clone(),
                included_x_handles: included_x_handles
                    .clone()
                    .or_else(|| x_handles_legacy.clone()),
                post_favorite_count: *post_favorite_count,
                post_view_count: *post_view_count,
            },
            Self::News {
                country,
                excluded_websites,
                safe_search,
            } => WireSearchSource::News {
                country: country.clone(),
                excluded_websites: excluded_websites.clone(),
                safe_search: *safe_search,
            },
            Self::Rss { links } => WireSearchSource::Rss {
                links: links.clone(),
            },
        }
    }
}

/// Parse the `xai` slot of [`ProviderOptions`], or return defaults.
///
/// Unknown / non-object entries fall back to defaults rather than failing
/// the call — ai-sdk has the same forgiving behavior.
pub(crate) fn parse(options: Option<&ProviderOptions>) -> XaiChatOptions {
    let Some(map) = options else {
        return XaiChatOptions::default();
    };
    let Some(xai) = map.get("xai") else {
        return XaiChatOptions::default();
    };
    serde_json::from_value::<XaiChatOptions>(serde_json::Value::Object(xai.clone()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn opts_with(map: &serde_json::Value) -> ProviderOptions {
        let mut po = ProviderOptions::new();
        po.insert("xai".into(), map.as_object().cloned().unwrap());
        po
    }

    #[test]
    fn missing_provider_options_yields_defaults() {
        let parsed = parse(None);
        assert!(parsed.reasoning_effort.is_none());
    }

    #[test]
    fn missing_xai_key_yields_defaults() {
        let mut po = ProviderOptions::new();
        po.insert(
            "openai".into(),
            json!({"reasoningEffort": "high"})
                .as_object()
                .cloned()
                .unwrap(),
        );
        let parsed = parse(Some(&po));
        assert!(parsed.reasoning_effort.is_none());
    }

    #[test]
    fn parses_reasoning_effort() {
        let po = opts_with(&json!({"reasoningEffort": "high"}));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn parses_top_logprobs() {
        let po = opts_with(&json!({"topLogprobs": 5, "logprobs": true}));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.top_logprobs, Some(5));
        assert_eq!(parsed.logprobs, Some(true));
    }

    #[test]
    fn parses_search_parameters_web() {
        let po = opts_with(&json!({
            "searchParameters": {
                "mode": "auto",
                "returnCitations": true,
                "maxSearchResults": 10,
                "sources": [
                    { "type": "web", "country": "US", "safeSearch": true }
                ]
            }
        }));
        let parsed = parse(Some(&po));
        let sp = parsed.search_parameters.expect("search_parameters");
        assert_eq!(sp.mode, "auto");
        let wire = sp.to_wire();
        assert_eq!(wire.mode, "auto");
        assert_eq!(wire.return_citations, Some(true));
        assert_eq!(wire.max_search_results, Some(10));
        let sources = wire.sources.unwrap();
        assert_eq!(sources.len(), 1);
        let json = serde_json::to_value(&sources[0]).unwrap();
        assert_eq!(json["type"], "web");
        assert_eq!(json["country"], "US");
        assert_eq!(json["safe_search"], true);
    }

    #[test]
    fn parses_search_parameters_x_legacy_handles() {
        let po = opts_with(&json!({
            "searchParameters": {
                "mode": "on",
                "sources": [
                    { "type": "x", "xHandles": ["@elon"] }
                ]
            }
        }));
        let sp = parse(Some(&po)).search_parameters.unwrap();
        let wire = sp.to_wire();
        let sources = wire.sources.unwrap();
        let json = serde_json::to_value(&sources[0]).unwrap();
        assert_eq!(json["type"], "x");
        assert_eq!(json["included_x_handles"][0], "@elon");
    }

    #[test]
    fn parses_search_parameters_rss() {
        let po = opts_with(&json!({
            "searchParameters": {
                "mode": "auto",
                "sources": [
                    { "type": "rss", "links": ["https://example.com/feed.xml"] }
                ]
            }
        }));
        let sp = parse(Some(&po)).search_parameters.unwrap();
        let wire = sp.to_wire();
        let json = serde_json::to_value(&wire.sources.unwrap()[0]).unwrap();
        assert_eq!(json["type"], "rss");
        assert_eq!(json["links"][0], "https://example.com/feed.xml");
    }

    #[test]
    fn unknown_keys_ignored() {
        let po = opts_with(&json!({"unknownField": 42, "reasoningEffort": "low"}));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.reasoning_effort.as_deref(), Some("low"));
    }
}
