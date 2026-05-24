//! Fill missing [`EmbedOptions`] fields with provider-level defaults.
//!
//! Embedding analogue of [`super::default_settings`]. Surface is narrower:
//! `headers` and `provider_options` are the only mergeable fields; `values`
//! is caller-only.
// Rust guideline compliant 2026-02-21

use async_trait::async_trait;

use crate::embedding_model::{EmbedOptions, EmbeddingModel};
use crate::error::Result;
use crate::middleware::embedding_model::EmbeddingModelMiddleware;
use crate::shared::{Headers, ProviderOptions};

/// Middleware applying a baseline [`EmbedOptions`] to every embed call.
#[derive(Debug, Clone, Default)]
pub struct DefaultEmbeddingSettingsMiddleware {
    defaults: EmbedOptions,
}

impl DefaultEmbeddingSettingsMiddleware {
    /// Build with the given default options.
    #[must_use]
    pub fn new(defaults: EmbedOptions) -> Self {
        Self { defaults }
    }
}

#[async_trait]
impl EmbeddingModelMiddleware for DefaultEmbeddingSettingsMiddleware {
    async fn transform_params(
        &self,
        params: EmbedOptions,
        _inner: &dyn EmbeddingModel,
    ) -> Result<EmbedOptions> {
        Ok(EmbedOptions {
            values: if params.values.is_empty() {
                self.defaults.values.clone()
            } else {
                params.values
            },
            headers: merge_headers(self.defaults.headers.clone(), params.headers),
            provider_options: merge_provider_options(
                self.defaults.provider_options.clone(),
                params.provider_options,
            ),
        })
    }
}

fn merge_headers(default: Option<Headers>, caller: Option<Headers>) -> Option<Headers> {
    match (default, caller) {
        (None, c) => c,
        (Some(d), None) => Some(d),
        (Some(mut d), Some(c)) => {
            d.extend(c);
            Some(d)
        }
    }
}

fn merge_provider_options(
    default: Option<ProviderOptions>,
    caller: Option<ProviderOptions>,
) -> Option<ProviderOptions> {
    match (default, caller) {
        (None, c) => c,
        (Some(d), None) => Some(d),
        (Some(mut d), Some(c)) => {
            for (provider, caller_inner) in c {
                let entry = d.entry(provider).or_default();
                for (k, v) in caller_inner {
                    entry.insert(k, v);
                }
            }
            Some(d)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::embedding_model::EmbedResult;
    use crate::middleware::wrap_embedding_model;

    #[derive(Debug, Default)]
    struct Recorder(Mutex<Option<EmbedOptions>>);

    #[async_trait]
    impl EmbeddingModel for Recorder {
        fn provider(&self) -> &'static str {
            "rec"
        }
        fn model_id(&self) -> &'static str {
            "rec"
        }
        async fn do_embed(&self, options: EmbedOptions) -> Result<EmbedResult> {
            *self.0.lock().expect("mutex") = Some(options);
            Ok(EmbedResult {
                embeddings: vec![],
                usage: None,
                provider_metadata: None,
                request: None,
                response: None,
            })
        }
    }

    #[tokio::test]
    async fn defaults_fill_missing_provider_options() {
        let rec = Arc::new(Recorder::default());
        let mut po = ProviderOptions::default();
        po.insert(
            "openai".into(),
            serde_json::json!({"dimensions": 256})
                .as_object()
                .cloned()
                .unwrap(),
        );
        let defaults = EmbedOptions {
            provider_options: Some(po),
            ..Default::default()
        };
        let wrapped = wrap_embedding_model(
            Arc::clone(&rec) as Arc<dyn EmbeddingModel>,
            [Arc::new(DefaultEmbeddingSettingsMiddleware::new(defaults))
                as Arc<dyn EmbeddingModelMiddleware>],
        );

        wrapped
            .do_embed(EmbedOptions {
                values: vec!["x".into()],
                ..Default::default()
            })
            .await
            .expect("embed");

        let captured = rec.0.lock().expect("mutex").clone().expect("params");
        let po = captured.provider_options.expect("po set");
        let openai = po.get("openai").expect("openai key");
        assert_eq!(
            openai.get("dimensions").and_then(serde_json::Value::as_i64),
            Some(256)
        );
    }
}
