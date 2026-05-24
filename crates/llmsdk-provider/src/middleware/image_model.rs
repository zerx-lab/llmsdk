//! `ImageModelMiddleware` trait and the `wrap_image_model` combinator.
//!
//! Mirrors `image-model-v4-middleware.ts` (trait surface) and
//! `wrap-image-model.ts` (combinator). Structurally identical to
//! [`super::language_model`] / [`super::embedding_model`] — only the callable
//! surface differs (`do_generate(ImageOptions) -> ImageResult`).
// Rust guideline compliant 2026-02-21

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::image_model::{ImageModel, ImageOptions, ImageResult};

/// Contract for middleware that decorates an [`ImageModel`].
#[async_trait]
pub trait ImageModelMiddleware: Send + Sync + std::fmt::Debug {
    /// Override the provider id exposed by the wrapped model.
    fn override_provider(&self, _inner: &dyn ImageModel) -> Option<String> {
        None
    }

    /// Override the model id exposed by the wrapped model.
    fn override_model_id(&self, _inner: &dyn ImageModel) -> Option<String> {
        None
    }

    /// Override [`ImageModel::max_images_per_call`].
    async fn override_max_images_per_call(&self, _inner: &dyn ImageModel) -> Option<Option<u32>> {
        None
    }

    /// Transform the image options before they reach the inner model.
    ///
    /// # Errors
    ///
    /// Return a [`crate::ProviderError`] to fail the call without invoking
    /// the model.
    async fn transform_params(
        &self,
        params: ImageOptions,
        _inner: &dyn ImageModel,
    ) -> Result<ImageOptions> {
        Ok(params)
    }

    /// Wrap an image-generation call.
    ///
    /// Default: forwards to `next.do_generate(params)`.
    ///
    /// # Errors
    ///
    /// Returns whatever error `next` returns, or a middleware-introduced
    /// failure.
    async fn wrap_generate(
        &self,
        next: &dyn ImageModel,
        params: ImageOptions,
    ) -> Result<ImageResult> {
        next.do_generate(params).await
    }
}

/// Compose an image model with one or more middlewares.
///
/// Outer-to-inner ordering on the way in, inner-to-outer on the way out
/// (list head = outermost). Empty middleware iterator returns the model
/// unchanged.
pub fn wrap_image_model<I>(model: Arc<dyn ImageModel>, middleware: I) -> Arc<dyn ImageModel>
where
    I: IntoIterator<Item = Arc<dyn ImageModelMiddleware>>,
{
    let mut layers: Vec<Arc<dyn ImageModelMiddleware>> = middleware.into_iter().collect();
    layers.reverse();
    layers
        .into_iter()
        .fold(model, |inner, mw| Arc::new(Wrapped::new(inner, mw)))
}

/// Internal one-layer wrapper.
struct Wrapped {
    inner: Arc<dyn ImageModel>,
    middleware: Arc<dyn ImageModelMiddleware>,
    provider: String,
    model_id: String,
}

impl Wrapped {
    fn new(inner: Arc<dyn ImageModel>, middleware: Arc<dyn ImageModelMiddleware>) -> Self {
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
impl ImageModel for Wrapped {
    fn provider(&self) -> &str {
        &self.provider
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_images_per_call(&self) -> Option<u32> {
        if let Some(custom) = self
            .middleware
            .override_max_images_per_call(self.inner.as_ref())
            .await
        {
            return custom;
        }
        self.inner.max_images_per_call().await
    }

    async fn do_generate(&self, options: ImageOptions) -> Result<ImageResult> {
        let transformed = self
            .middleware
            .transform_params(options, self.inner.as_ref())
            .await?;
        self.middleware
            .wrap_generate(self.inner.as_ref(), transformed)
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[derive(Debug, Default)]
    struct MockImage {
        provider: String,
        model_id: String,
        calls: AtomicUsize,
        last_prompt: Mutex<String>,
    }

    impl MockImage {
        fn new(provider: &str, model_id: &str) -> Self {
            Self {
                provider: provider.to_owned(),
                model_id: model_id.to_owned(),
                calls: AtomicUsize::new(0),
                last_prompt: Mutex::new(String::new()),
            }
        }
    }

    #[async_trait]
    impl ImageModel for MockImage {
        fn provider(&self) -> &str {
            &self.provider
        }
        fn model_id(&self) -> &str {
            &self.model_id
        }
        async fn do_generate(&self, options: ImageOptions) -> Result<ImageResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            *self.last_prompt.lock().expect("mutex") = options.prompt;
            Ok(ImageResult {
                images: vec![],
                warnings: vec![],
                usage: None,
                provider_metadata: None,
                request: None,
                response: None,
            })
        }
    }

    #[derive(Debug)]
    struct OverrideAndPrefix;

    #[async_trait]
    impl ImageModelMiddleware for OverrideAndPrefix {
        fn override_model_id(&self, _: &dyn ImageModel) -> Option<String> {
            Some("wrapped-model".to_owned())
        }

        async fn transform_params(
            &self,
            mut params: ImageOptions,
            _inner: &dyn ImageModel,
        ) -> Result<ImageOptions> {
            params.prompt = format!("PREFIX: {}", params.prompt);
            Ok(params)
        }
    }

    #[tokio::test]
    async fn empty_middleware_unchanged() {
        let model = Arc::new(MockImage::new("p", "m"));
        let wrapped: Arc<dyn ImageModel> = wrap_image_model(Arc::clone(&model) as _, Vec::new());
        assert_eq!(wrapped.model_id(), "m");
    }

    #[tokio::test]
    async fn overrides_and_prefix_run() {
        let model = Arc::new(MockImage::new("p", "m"));
        let wrapped = wrap_image_model(
            Arc::clone(&model) as _,
            [Arc::new(OverrideAndPrefix) as Arc<dyn ImageModelMiddleware>],
        );

        assert_eq!(wrapped.model_id(), "wrapped-model");

        wrapped
            .do_generate(ImageOptions {
                prompt: "a cat".into(),
                ..Default::default()
            })
            .await
            .expect("generate");

        assert_eq!(model.calls.load(Ordering::SeqCst), 1);
        assert_eq!(*model.last_prompt.lock().expect("mutex"), "PREFIX: a cat");
    }
}
