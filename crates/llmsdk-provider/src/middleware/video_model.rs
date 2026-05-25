//! `VideoModelMiddleware` trait and the `wrap_video_model` combinator.
//!
//! Mirrors the v4 middleware pattern from `language-model-middleware`. Surface
//! is intentionally identical to [`super::image_model`] —
//! `do_generate(VideoOptions) -> VideoResult`.
// Rust guideline compliant 2026-02-21

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::video_model::{VideoModel, VideoOptions, VideoResult};

/// Contract for middleware that decorates a [`VideoModel`].
#[async_trait]
pub trait VideoModelMiddleware: Send + Sync + std::fmt::Debug {
    /// Override the provider id exposed by the wrapped model.
    fn override_provider(&self, _inner: &dyn VideoModel) -> Option<String> {
        None
    }

    /// Override the model id exposed by the wrapped model.
    fn override_model_id(&self, _inner: &dyn VideoModel) -> Option<String> {
        None
    }

    /// Override [`VideoModel::max_videos_per_call`].
    async fn override_max_videos_per_call(&self, _inner: &dyn VideoModel) -> Option<Option<u32>> {
        None
    }

    /// Transform the video options before they reach the inner model.
    ///
    /// # Errors
    ///
    /// Return a [`crate::ProviderError`] to fail the call without invoking
    /// the model.
    async fn transform_params(
        &self,
        params: VideoOptions,
        _inner: &dyn VideoModel,
    ) -> Result<VideoOptions> {
        Ok(params)
    }

    /// Wrap a video-generation call.
    ///
    /// Default: forwards to `next.do_generate(params)`.
    ///
    /// # Errors
    ///
    /// Returns whatever error `next` returns, or a middleware-introduced
    /// failure.
    async fn wrap_generate(
        &self,
        next: &dyn VideoModel,
        params: VideoOptions,
    ) -> Result<VideoResult> {
        next.do_generate(params).await
    }
}

/// Compose a video model with one or more middlewares.
///
/// Outer-to-inner ordering on the way in, inner-to-outer on the way out
/// (list head = outermost). Empty middleware iterator returns the model
/// unchanged.
pub fn wrap_video_model<I>(model: Arc<dyn VideoModel>, middleware: I) -> Arc<dyn VideoModel>
where
    I: IntoIterator<Item = Arc<dyn VideoModelMiddleware>>,
{
    let mut layers: Vec<Arc<dyn VideoModelMiddleware>> = middleware.into_iter().collect();
    layers.reverse();
    layers
        .into_iter()
        .fold(model, |inner, mw| Arc::new(Wrapped::new(inner, mw)))
}

/// Internal one-layer wrapper.
struct Wrapped {
    inner: Arc<dyn VideoModel>,
    middleware: Arc<dyn VideoModelMiddleware>,
    provider: String,
    model_id: String,
}

impl Wrapped {
    fn new(inner: Arc<dyn VideoModel>, middleware: Arc<dyn VideoModelMiddleware>) -> Self {
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
impl VideoModel for Wrapped {
    fn provider(&self) -> &str {
        &self.provider
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_videos_per_call(&self) -> Option<u32> {
        if let Some(custom) = self
            .middleware
            .override_max_videos_per_call(self.inner.as_ref())
            .await
        {
            return custom;
        }
        self.inner.max_videos_per_call().await
    }

    async fn do_generate(&self, options: VideoOptions) -> Result<VideoResult> {
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
    use super::*;
    use crate::video_model::VideoResponseInfo;

    #[derive(Debug, Default)]
    struct MockVideo {
        provider: String,
        model_id: String,
    }

    impl MockVideo {
        fn new(provider: &str, model_id: &str) -> Self {
            Self {
                provider: provider.to_owned(),
                model_id: model_id.to_owned(),
            }
        }
    }

    #[async_trait]
    impl VideoModel for MockVideo {
        fn provider(&self) -> &str {
            &self.provider
        }
        fn model_id(&self) -> &str {
            &self.model_id
        }
        async fn do_generate(&self, _options: VideoOptions) -> Result<VideoResult> {
            Ok(VideoResult {
                videos: vec![],
                warnings: vec![],
                provider_metadata: None,
                response: VideoResponseInfo {
                    timestamp: "2026-05-25T00:00:00Z".into(),
                    model_id: "mock".into(),
                    headers: None,
                },
            })
        }
    }

    #[derive(Debug)]
    struct OverrideName;

    #[async_trait]
    impl VideoModelMiddleware for OverrideName {
        fn override_model_id(&self, _: &dyn VideoModel) -> Option<String> {
            Some("wrapped-video".into())
        }
    }

    #[tokio::test]
    async fn empty_middleware_unchanged() {
        let model = Arc::new(MockVideo::new("xai", "v1"));
        let wrapped = wrap_video_model(model as _, Vec::new());
        assert_eq!(wrapped.model_id(), "v1");
    }

    #[tokio::test]
    async fn override_runs_at_construction() {
        let model = Arc::new(MockVideo::new("xai", "v1"));
        let wrapped = wrap_video_model(
            model as _,
            [Arc::new(OverrideName) as Arc<dyn VideoModelMiddleware>],
        );
        assert_eq!(wrapped.model_id(), "wrapped-video");
    }
}
