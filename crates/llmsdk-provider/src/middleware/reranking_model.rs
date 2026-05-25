//! `RerankingModelMiddleware` trait and the `wrap_reranking_model` combinator.
//!
//! Mirrors the v4 middleware pattern. Surface mirrors [`super::image_model`]
//! but operates on `do_rerank(RerankingOptions) -> RerankingResult`.
// Rust guideline compliant 2026-02-21

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::reranking_model::{RerankingModel, RerankingOptions, RerankingResult};

/// Contract for middleware that decorates a [`RerankingModel`].
#[async_trait]
pub trait RerankingModelMiddleware: Send + Sync + std::fmt::Debug {
    /// Override the provider id exposed by the wrapped model.
    fn override_provider(&self, _inner: &dyn RerankingModel) -> Option<String> {
        None
    }

    /// Override the model id exposed by the wrapped model.
    fn override_model_id(&self, _inner: &dyn RerankingModel) -> Option<String> {
        None
    }

    /// Transform the rerank options before they reach the inner model.
    ///
    /// # Errors
    ///
    /// Return a [`crate::ProviderError`] to fail the call without invoking
    /// the model.
    async fn transform_params(
        &self,
        params: RerankingOptions,
        _inner: &dyn RerankingModel,
    ) -> Result<RerankingOptions> {
        Ok(params)
    }

    /// Wrap a rerank call.
    ///
    /// Default: forwards to `next.do_rerank(params)`.
    ///
    /// # Errors
    ///
    /// Returns whatever error `next` returns, or a middleware-introduced
    /// failure.
    async fn wrap_rerank(
        &self,
        next: &dyn RerankingModel,
        params: RerankingOptions,
    ) -> Result<RerankingResult> {
        next.do_rerank(params).await
    }
}

/// Compose a reranking model with one or more middlewares.
///
/// Outer-to-inner ordering on the way in, inner-to-outer on the way out
/// (list head = outermost). Empty middleware iterator returns the model
/// unchanged.
pub fn wrap_reranking_model<I>(
    model: Arc<dyn RerankingModel>,
    middleware: I,
) -> Arc<dyn RerankingModel>
where
    I: IntoIterator<Item = Arc<dyn RerankingModelMiddleware>>,
{
    let mut layers: Vec<Arc<dyn RerankingModelMiddleware>> = middleware.into_iter().collect();
    layers.reverse();
    layers
        .into_iter()
        .fold(model, |inner, mw| Arc::new(Wrapped::new(inner, mw)))
}

/// Internal one-layer wrapper.
struct Wrapped {
    inner: Arc<dyn RerankingModel>,
    middleware: Arc<dyn RerankingModelMiddleware>,
    provider: String,
    model_id: String,
}

impl Wrapped {
    fn new(inner: Arc<dyn RerankingModel>, middleware: Arc<dyn RerankingModelMiddleware>) -> Self {
        let provider = middleware
            .override_provider(inner.as_ref())
            .unwrap_or_else(|| inner.provider().to_owned());
        let model_id = middleware
            .override_model_id(inner.as_ref())
            .unwrap_or_else(|| inner.model_id().to_owned());
        Self {
            inner,
            middleware,
            provider,
            model_id,
        }
    }
}

impl std::fmt::Debug for Wrapped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wrapped")
            .field("provider", &self.provider)
            .field("model_id", &self.model_id)
            .field("middleware", &self.middleware)
            .field("inner", &self.inner)
            .finish()
    }
}

#[async_trait]
impl RerankingModel for Wrapped {
    fn provider(&self) -> &str {
        &self.provider
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn do_rerank(&self, options: RerankingOptions) -> Result<RerankingResult> {
        let transformed = self
            .middleware
            .transform_params(options, self.inner.as_ref())
            .await?;
        self.middleware
            .wrap_rerank(self.inner.as_ref(), transformed)
            .await
    }
}

#[cfg(test)]
#[allow(
    clippy::unnecessary_literal_bound,
    reason = "trait method signatures use &str; mock implementations return string literals"
)]
mod tests {
    use super::*;
    use crate::reranking_model::RerankingDocuments;

    #[derive(Debug, Default)]
    struct MockRerank;

    #[async_trait]
    impl RerankingModel for MockRerank {
        fn provider(&self) -> &str {
            "mock"
        }
        fn model_id(&self) -> &str {
            "rr"
        }
        async fn do_rerank(&self, _options: RerankingOptions) -> Result<RerankingResult> {
            Ok(RerankingResult {
                ranking: vec![],
                warnings: vec![],
                provider_metadata: None,
                response: None,
            })
        }
    }

    #[derive(Debug)]
    struct OverrideName;

    #[async_trait]
    impl RerankingModelMiddleware for OverrideName {
        fn override_model_id(&self, _: &dyn RerankingModel) -> Option<String> {
            Some("wrapped".into())
        }
    }

    #[tokio::test]
    async fn empty_middleware_unchanged() {
        let model = Arc::new(MockRerank);
        let wrapped = wrap_reranking_model(model as _, Vec::new());
        assert_eq!(wrapped.model_id(), "rr");
    }

    #[tokio::test]
    async fn override_runs_at_construction() {
        let model = Arc::new(MockRerank);
        let wrapped = wrap_reranking_model(
            model as _,
            [Arc::new(OverrideName) as Arc<dyn RerankingModelMiddleware>],
        );
        assert_eq!(wrapped.model_id(), "wrapped");

        wrapped
            .do_rerank(RerankingOptions {
                documents: RerankingDocuments::Text {
                    values: vec!["a".into()],
                },
                query: "q".into(),
                top_n: None,
                headers: None,
                provider_options: None,
            })
            .await
            .expect("rerank");
    }
}
