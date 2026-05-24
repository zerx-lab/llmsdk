//! `LanguageModelMiddleware` trait and the `wrap_language_model` combinator.
//!
//! Mirrors `language-model-v4-middleware.ts` (trait surface) and
//! `wrap-language-model.ts` (combinator). The combinator merges ai-sdk's
//! `doGenerate` + `doStream` closure pair into a single `next: &dyn LanguageModel`
//! argument; middleware that wants to swap call kinds (e.g. a future
//! simulate-streaming middleware) just calls the other method on `next`.
// Rust guideline compliant 2026-02-21

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::language_model::{
    CallOptions, GenerateResult, LanguageModel, StreamResult, SupportedUrls,
};

/// Discriminates the active call kind passed to
/// [`LanguageModelMiddleware::transform_params`].
///
/// Mirrors ai-sdk's `type: 'generate' | 'stream'` discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallKind {
    /// Non-streaming generation; the wrapper will invoke
    /// [`LanguageModel::do_generate`] after the middleware chain runs.
    Generate,
    /// Streaming generation; the wrapper will invoke
    /// [`LanguageModel::do_stream`] after the middleware chain runs.
    Stream,
}

/// Contract for middleware that decorates a [`LanguageModel`].
///
/// Every method has a sensible default, so an implementor only overrides the
/// hooks it cares about. The combinator [`wrap_language_model`] composes any
/// number of middlewares into a fresh `LanguageModel` instance.
///
/// # Semantics
///
/// - `override_*`: replace the corresponding identity / metadata accessor.
/// - [`transform_params`](Self::transform_params): mutate the call options
///   before they reach the underlying model. Runs once per call, before
///   `wrap_*`.
/// - [`wrap_generate`](Self::wrap_generate) / [`wrap_stream`](Self::wrap_stream):
///   intercept the actual call. The default implementation simply forwards
///   to `next`; overrides may add retry, caching, instrumentation, swap
///   between generate/stream, etc.
///
/// `next` is the *next layer* (which may itself be a wrapped model or the
/// original provider model), not necessarily the underlying provider model.
#[async_trait]
pub trait LanguageModelMiddleware: Send + Sync + std::fmt::Debug {
    /// Override the provider id exposed by the wrapped model.
    ///
    /// Return `None` to keep `inner.provider()`. The override is read once
    /// when [`wrap_language_model`] runs, so it must not depend on call-time
    /// state.
    fn override_provider(&self, _inner: &dyn LanguageModel) -> Option<String> {
        None
    }

    /// Override the model id exposed by the wrapped model.
    ///
    /// Same caching semantics as [`Self::override_provider`].
    fn override_model_id(&self, _inner: &dyn LanguageModel) -> Option<String> {
        None
    }

    /// Override the supported-URL map exposed by the wrapped model.
    ///
    /// Unlike the identity overrides, this is re-evaluated on every
    /// [`LanguageModel::supported_urls`] call so middleware can reflect
    /// dynamic provider state.
    async fn override_supported_urls(&self, _inner: &dyn LanguageModel) -> Option<SupportedUrls> {
        None
    }

    /// Transform the call options before they reach the inner model.
    ///
    /// Runs once per call, before [`Self::wrap_generate`] / [`Self::wrap_stream`].
    /// The returned options are passed to both the next middleware layer's
    /// `transform_params` and the eventual underlying call.
    ///
    /// # Errors
    ///
    /// Return a [`crate::ProviderError`] to fail the call without invoking
    /// the model.
    async fn transform_params(
        &self,
        _kind: CallKind,
        params: CallOptions,
        _inner: &dyn LanguageModel,
    ) -> Result<CallOptions> {
        Ok(params)
    }

    /// Wrap a non-streaming generation.
    ///
    /// Default: forwards to `next.do_generate(params)`. Override to add
    /// retry, caching, telemetry, or to dispatch to `next.do_stream` instead.
    ///
    /// # Errors
    ///
    /// Returns whatever error `next` returns, or a middleware-introduced
    /// failure.
    async fn wrap_generate(
        &self,
        next: &dyn LanguageModel,
        params: CallOptions,
    ) -> Result<GenerateResult> {
        next.do_generate(params).await
    }

    /// Wrap a streaming generation.
    ///
    /// Default: forwards to `next.do_stream(params)`. Override to add
    /// retry, caching, telemetry, or to simulate streaming on top of
    /// `next.do_generate`.
    ///
    /// # Errors
    ///
    /// Returns whatever error `next` returns, or a middleware-introduced
    /// failure.
    async fn wrap_stream(
        &self,
        next: &dyn LanguageModel,
        params: CallOptions,
    ) -> Result<StreamResult> {
        next.do_stream(params).await
    }
}

/// Compose a model with one or more middlewares.
///
/// The returned `Arc<dyn LanguageModel>` runs middleware in *outer-to-inner*
/// order on the way in (`m[0].transform_params` first) and in *inner-to-outer*
/// order on the way out (`m[0].wrap_generate` is the outermost wrap). This
/// matches the convention used by `@ai-sdk/ai`'s `wrapLanguageModel`.
///
/// Passing an empty middleware iterator returns the model unchanged.
///
/// # Examples
///
/// Stacking two middlewares (the first is the outermost):
///
/// ```ignore
/// use std::sync::Arc;
/// use llmsdk_provider::{wrap_language_model, LanguageModel, LanguageModelMiddleware};
///
/// fn stack(
///     model: Arc<dyn LanguageModel>,
///     retry: Arc<dyn LanguageModelMiddleware>,
///     log: Arc<dyn LanguageModelMiddleware>,
/// ) -> Arc<dyn LanguageModel> {
///     // `log` wraps `retry` wraps `model`. Logs see every retry attempt.
///     wrap_language_model(model, [log, retry])
/// }
/// ```
pub fn wrap_language_model<I>(
    model: Arc<dyn LanguageModel>,
    middleware: I,
) -> Arc<dyn LanguageModel>
where
    I: IntoIterator<Item = Arc<dyn LanguageModelMiddleware>>,
{
    let mut layers: Vec<Arc<dyn LanguageModelMiddleware>> = middleware.into_iter().collect();
    // Apply right-most middleware first so list head ends up outermost.
    layers.reverse();
    layers
        .into_iter()
        .fold(model, |inner, mw| Arc::new(Wrapped::new(inner, mw)))
}

/// Internal one-layer wrapper that pairs a model with a single middleware.
///
/// Each call to [`wrap_language_model`] produces a stack of these. We cache
/// the identity overrides at construction time because the trait accessors
/// return `&str` while the middleware returns `Option<String>`.
struct Wrapped {
    inner: Arc<dyn LanguageModel>,
    middleware: Arc<dyn LanguageModelMiddleware>,
    provider: String,
    model_id: String,
}

impl Wrapped {
    fn new(inner: Arc<dyn LanguageModel>, middleware: Arc<dyn LanguageModelMiddleware>) -> Self {
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
impl LanguageModel for Wrapped {
    fn provider(&self) -> &str {
        &self.provider
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn supported_urls(&self) -> SupportedUrls {
        if let Some(custom) = self
            .middleware
            .override_supported_urls(self.inner.as_ref())
            .await
        {
            return custom;
        }
        self.inner.supported_urls().await
    }

    async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult> {
        let transformed = self
            .middleware
            .transform_params(CallKind::Generate, options, self.inner.as_ref())
            .await?;
        self.middleware
            .wrap_generate(self.inner.as_ref(), transformed)
            .await
    }

    async fn do_stream(&self, options: CallOptions) -> Result<StreamResult> {
        let transformed = self
            .middleware
            .transform_params(CallKind::Stream, options, self.inner.as_ref())
            .await?;
        self.middleware
            .wrap_stream(self.inner.as_ref(), transformed)
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures::StreamExt;
    use futures::stream;

    use crate::language_model::{FinishReason, FinishReasonKind, StreamPart, Usage};

    use super::*;

    /// Mock model that records every `do_generate` / `do_stream` invocation
    /// and lets the test decide what to return.
    #[derive(Debug, Default)]
    struct MockModel {
        provider: String,
        model_id: String,
        generate_calls: AtomicUsize,
        stream_calls: AtomicUsize,
        last_params: Mutex<Option<CallOptions>>,
    }

    impl MockModel {
        fn new(provider: &str, model_id: &str) -> Self {
            Self {
                provider: provider.to_owned(),
                model_id: model_id.to_owned(),
                generate_calls: AtomicUsize::new(0),
                stream_calls: AtomicUsize::new(0),
                last_params: Mutex::new(None),
            }
        }

        fn generate_count(&self) -> usize {
            self.generate_calls.load(Ordering::SeqCst)
        }

        fn stream_count(&self) -> usize {
            self.stream_calls.load(Ordering::SeqCst)
        }

        fn last_temperature(&self) -> Option<f32> {
            self.last_params
                .lock()
                .expect("mock mutex poisoned")
                .as_ref()
                .and_then(|p| p.temperature)
        }
    }

    #[async_trait]
    impl LanguageModel for MockModel {
        fn provider(&self) -> &str {
            &self.provider
        }

        fn model_id(&self) -> &str {
            &self.model_id
        }

        async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult> {
            self.generate_calls.fetch_add(1, Ordering::SeqCst);
            *self.last_params.lock().expect("mock mutex poisoned") = Some(options);
            Ok(GenerateResult {
                content: vec![],
                finish_reason: FinishReason::new(FinishReasonKind::Stop),
                usage: Usage::default(),
                provider_metadata: None,
                request: None,
                response: None,
                warnings: vec![],
            })
        }

        async fn do_stream(&self, options: CallOptions) -> Result<StreamResult> {
            self.stream_calls.fetch_add(1, Ordering::SeqCst);
            *self.last_params.lock().expect("mock mutex poisoned") = Some(options);
            let parts = stream::iter(vec![
                Ok(StreamPart::StreamStart { warnings: vec![] }),
                Ok(StreamPart::Finish {
                    usage: Usage::default(),
                    finish_reason: FinishReason::new(FinishReasonKind::Stop),
                    provider_metadata: None,
                }),
            ]);
            Ok(StreamResult {
                stream: Box::pin(parts),
                request: None,
                response: None,
            })
        }
    }

    /// Middleware that overrides identity + bumps temperature.
    #[derive(Debug)]
    struct OverrideAndTransform;

    #[async_trait]
    impl LanguageModelMiddleware for OverrideAndTransform {
        fn override_provider(&self, _inner: &dyn LanguageModel) -> Option<String> {
            Some("wrapped-provider".to_owned())
        }

        fn override_model_id(&self, _inner: &dyn LanguageModel) -> Option<String> {
            Some("wrapped-model".to_owned())
        }

        async fn transform_params(
            &self,
            _kind: CallKind,
            mut params: CallOptions,
            _inner: &dyn LanguageModel,
        ) -> Result<CallOptions> {
            params.temperature = Some(params.temperature.unwrap_or(0.0) + 1.0);
            Ok(params)
        }
    }

    /// Records the order in which `wrap_generate` runs across the stack.
    #[derive(Debug)]
    struct OrderRecorder {
        label: &'static str,
        log: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl LanguageModelMiddleware for OrderRecorder {
        async fn wrap_generate(
            &self,
            next: &dyn LanguageModel,
            params: CallOptions,
        ) -> Result<GenerateResult> {
            self.log
                .lock()
                .expect("log mutex poisoned")
                .push(format!("{}:enter", self.label));
            let res = next.do_generate(params).await;
            self.log
                .lock()
                .expect("log mutex poisoned")
                .push(format!("{}:exit", self.label));
            res
        }
    }

    /// Middleware that ignores `do_stream` and serves it from `do_generate`.
    #[derive(Debug)]
    struct StreamFromGenerate;

    #[async_trait]
    impl LanguageModelMiddleware for StreamFromGenerate {
        async fn wrap_stream(
            &self,
            next: &dyn LanguageModel,
            params: CallOptions,
        ) -> Result<StreamResult> {
            // Prove that middleware can swap call kinds via `next`.
            let _ = next.do_generate(params).await?;
            Ok(StreamResult {
                stream: Box::pin(stream::iter(vec![])),
                request: None,
                response: None,
            })
        }
    }

    #[tokio::test]
    async fn empty_middleware_returns_model_unchanged() {
        let model = Arc::new(MockModel::new("openai", "gpt-foo"));
        let wrapped: Arc<dyn LanguageModel> =
            wrap_language_model(Arc::clone(&model) as _, Vec::new());
        assert_eq!(wrapped.provider(), "openai");
        assert_eq!(wrapped.model_id(), "gpt-foo");

        wrapped
            .do_generate(CallOptions::default())
            .await
            .expect("generate succeeded");
        assert_eq!(model.generate_count(), 1);
    }

    #[tokio::test]
    async fn overrides_replace_identity_and_transform_mutates_params() {
        let model = Arc::new(MockModel::new("openai", "gpt-foo"));
        let wrapped = wrap_language_model(
            Arc::clone(&model) as _,
            [Arc::new(OverrideAndTransform) as Arc<dyn LanguageModelMiddleware>],
        );

        assert_eq!(wrapped.provider(), "wrapped-provider");
        assert_eq!(wrapped.model_id(), "wrapped-model");

        wrapped
            .do_generate(CallOptions::default())
            .await
            .expect("generate succeeded");
        assert_eq!(model.last_temperature(), Some(1.0));
    }

    #[tokio::test]
    async fn wrap_order_runs_first_middleware_outermost() {
        let model = Arc::new(MockModel::new("openai", "gpt-foo"));
        let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let m1 = Arc::new(OrderRecorder {
            label: "m1",
            log: Arc::clone(&log),
        }) as Arc<dyn LanguageModelMiddleware>;
        let m2 = Arc::new(OrderRecorder {
            label: "m2",
            log: Arc::clone(&log),
        }) as Arc<dyn LanguageModelMiddleware>;

        let wrapped = wrap_language_model(model, [m1, m2]);
        wrapped
            .do_generate(CallOptions::default())
            .await
            .expect("generate succeeded");

        let entries = log.lock().expect("log mutex poisoned").clone();
        assert_eq!(
            entries,
            vec!["m1:enter", "m2:enter", "m2:exit", "m1:exit"],
            "first middleware must be outermost",
        );
    }

    #[tokio::test]
    async fn middleware_can_swap_call_kind_via_next() {
        let model = Arc::new(MockModel::new("openai", "gpt-foo"));
        let wrapped = wrap_language_model(
            Arc::clone(&model) as _,
            [Arc::new(StreamFromGenerate) as Arc<dyn LanguageModelMiddleware>],
        );

        let mut stream = wrapped
            .do_stream(CallOptions::default())
            .await
            .expect("stream succeeded")
            .stream;
        // Drain the (empty) stream to satisfy `must_use`.
        assert!(stream.next().await.is_none());

        assert_eq!(model.generate_count(), 1, "do_generate was used internally");
        assert_eq!(model.stream_count(), 0, "do_stream on inner was bypassed");
    }
}
