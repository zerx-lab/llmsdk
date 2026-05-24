//! `EmbeddingModelMiddleware` trait and the `wrap_embedding_model` combinator.
//!
//! Mirrors `embedding-model-v4-middleware.ts` (trait surface) and
//! `wrap-embedding-model.ts` (combinator). Structurally identical to
//! [`super::language_model`]'s combinator; the only differences are the
//! callable surface (`do_embed` instead of `do_generate` / `do_stream`) and
//! two embedding-specific identity overrides (`max_embeddings_per_call`,
//! `supports_parallel_calls`).
// Rust guideline compliant 2026-02-21

use std::sync::Arc;

use async_trait::async_trait;

use crate::embedding_model::{EmbedOptions, EmbedResult, EmbeddingModel};
use crate::error::Result;

/// Contract for middleware that decorates an [`EmbeddingModel`].
///
/// Every method has a sensible default; an implementor only overrides the
/// hooks it cares about. The combinator [`wrap_embedding_model`] composes any
/// number of middlewares into a fresh `EmbeddingModel` instance.
#[async_trait]
pub trait EmbeddingModelMiddleware: Send + Sync + std::fmt::Debug {
    /// Override the provider id exposed by the wrapped model.
    fn override_provider(&self, _inner: &dyn EmbeddingModel) -> Option<String> {
        None
    }

    /// Override the model id exposed by the wrapped model.
    fn override_model_id(&self, _inner: &dyn EmbeddingModel) -> Option<String> {
        None
    }

    /// Override [`EmbeddingModel::max_embeddings_per_call`].
    ///
    /// Returns `None` to defer to the inner model.
    async fn override_max_embeddings_per_call(
        &self,
        _inner: &dyn EmbeddingModel,
    ) -> Option<Option<u32>> {
        None
    }

    /// Override [`EmbeddingModel::supports_parallel_calls`].
    async fn override_supports_parallel_calls(&self, _inner: &dyn EmbeddingModel) -> Option<bool> {
        None
    }

    /// Transform the embed options before they reach the inner model.
    ///
    /// # Errors
    ///
    /// Return a [`crate::ProviderError`] to fail the call without invoking
    /// the model.
    async fn transform_params(
        &self,
        params: EmbedOptions,
        _inner: &dyn EmbeddingModel,
    ) -> Result<EmbedOptions> {
        Ok(params)
    }

    /// Wrap an embedding call.
    ///
    /// Default: forwards to `next.do_embed(params)`. Override to add retry,
    /// caching, telemetry, etc.
    ///
    /// # Errors
    ///
    /// Returns whatever error `next` returns, or a middleware-introduced
    /// failure.
    async fn wrap_embed(
        &self,
        next: &dyn EmbeddingModel,
        params: EmbedOptions,
    ) -> Result<EmbedResult> {
        next.do_embed(params).await
    }
}

/// Compose an embedding model with one or more middlewares.
///
/// The returned `Arc<dyn EmbeddingModel>` runs middleware in outer-to-inner
/// order on the way in (`m[0].transform_params` first) and in inner-to-outer
/// order on the way out (`m[0].wrap_embed` is the outermost wrap), matching
/// the convention from `wrap_language_model`.
///
/// Passing an empty middleware iterator returns the model unchanged.
pub fn wrap_embedding_model<I>(
    model: Arc<dyn EmbeddingModel>,
    middleware: I,
) -> Arc<dyn EmbeddingModel>
where
    I: IntoIterator<Item = Arc<dyn EmbeddingModelMiddleware>>,
{
    let mut layers: Vec<Arc<dyn EmbeddingModelMiddleware>> = middleware.into_iter().collect();
    layers.reverse();
    layers
        .into_iter()
        .fold(model, |inner, mw| Arc::new(Wrapped::new(inner, mw)))
}

/// Internal one-layer wrapper.
struct Wrapped {
    inner: Arc<dyn EmbeddingModel>,
    middleware: Arc<dyn EmbeddingModelMiddleware>,
    provider: String,
    model_id: String,
}

impl Wrapped {
    fn new(inner: Arc<dyn EmbeddingModel>, middleware: Arc<dyn EmbeddingModelMiddleware>) -> Self {
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
impl EmbeddingModel for Wrapped {
    fn provider(&self) -> &str {
        &self.provider
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_embeddings_per_call(&self) -> Option<u32> {
        if let Some(custom) = self
            .middleware
            .override_max_embeddings_per_call(self.inner.as_ref())
            .await
        {
            return custom;
        }
        self.inner.max_embeddings_per_call().await
    }

    async fn supports_parallel_calls(&self) -> bool {
        if let Some(custom) = self
            .middleware
            .override_supports_parallel_calls(self.inner.as_ref())
            .await
        {
            return custom;
        }
        self.inner.supports_parallel_calls().await
    }

    async fn do_embed(&self, options: EmbedOptions) -> Result<EmbedResult> {
        let transformed = self
            .middleware
            .transform_params(options, self.inner.as_ref())
            .await?;
        self.middleware
            .wrap_embed(self.inner.as_ref(), transformed)
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[derive(Debug, Default)]
    struct MockEmbed {
        provider: String,
        model_id: String,
        calls: AtomicUsize,
        last_input_len: Mutex<usize>,
    }

    impl MockEmbed {
        fn new(provider: &str, model_id: &str) -> Self {
            Self {
                provider: provider.to_owned(),
                model_id: model_id.to_owned(),
                calls: AtomicUsize::new(0),
                last_input_len: Mutex::new(0),
            }
        }
    }

    #[async_trait]
    impl EmbeddingModel for MockEmbed {
        fn provider(&self) -> &str {
            &self.provider
        }
        fn model_id(&self) -> &str {
            &self.model_id
        }
        async fn do_embed(&self, options: EmbedOptions) -> Result<EmbedResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            *self.last_input_len.lock().expect("mutex") = options.values.len();
            Ok(EmbedResult {
                embeddings: options.values.iter().map(|_| vec![0.0; 3]).collect(),
                usage: None,
                provider_metadata: None,
                request: None,
                response: None,
            })
        }
    }

    #[derive(Debug)]
    struct OverrideAndDoubleInputs;

    #[async_trait]
    impl EmbeddingModelMiddleware for OverrideAndDoubleInputs {
        fn override_provider(&self, _inner: &dyn EmbeddingModel) -> Option<String> {
            Some("wrapped".to_owned())
        }

        async fn override_max_embeddings_per_call(
            &self,
            _inner: &dyn EmbeddingModel,
        ) -> Option<Option<u32>> {
            Some(Some(42))
        }

        async fn transform_params(
            &self,
            mut params: EmbedOptions,
            _inner: &dyn EmbeddingModel,
        ) -> Result<EmbedOptions> {
            let original = params.values.clone();
            params.values.extend(original);
            Ok(params)
        }
    }

    #[tokio::test]
    async fn empty_middleware_returns_unchanged() {
        let model = Arc::new(MockEmbed::new("p", "m"));
        let wrapped: Arc<dyn EmbeddingModel> =
            wrap_embedding_model(Arc::clone(&model) as _, Vec::new());
        assert_eq!(wrapped.provider(), "p");
        assert_eq!(wrapped.model_id(), "m");
    }

    #[tokio::test]
    async fn overrides_and_transform_run() {
        let model = Arc::new(MockEmbed::new("p", "m"));
        let wrapped = wrap_embedding_model(
            Arc::clone(&model) as _,
            [Arc::new(OverrideAndDoubleInputs) as Arc<dyn EmbeddingModelMiddleware>],
        );

        assert_eq!(wrapped.provider(), "wrapped");
        assert_eq!(wrapped.max_embeddings_per_call().await, Some(42));

        wrapped
            .do_embed(EmbedOptions {
                values: vec!["a".into(), "b".into()],
                ..Default::default()
            })
            .await
            .expect("embed");

        assert_eq!(model.calls.load(Ordering::SeqCst), 1);
        assert_eq!(*model.last_input_len.lock().expect("mutex"), 4);
    }
}
