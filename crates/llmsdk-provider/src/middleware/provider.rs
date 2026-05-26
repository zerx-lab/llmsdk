//! Top-level [`Provider`] decoration.
//!
//! Mirrors `wrap-provider.ts`. Decorates every model returned by an inner
//! [`Provider`] with the matching surface middleware chain. Unlike the
//! ai-sdk TS variant, we surface the three middleware lists as a typed
//! [`ProviderMiddlewareSet`] struct rather than a free-form options bag.
//!
//! Limitation (matches ai-sdk): the middleware chain is applied uniformly to
//! every model id; per-model-id routing must be implemented in a custom
//! [`Provider`].
// Rust guideline compliant 2026-02-21

use std::sync::Arc;

use crate::error::Result;
use crate::image_model::ImageModel;
use crate::language_model::LanguageModel;
use crate::provider::{DynEmbeddingModel, DynImageModel, DynLanguageModel, Provider};

use super::image_model::{ImageModelMiddleware, wrap_image_model};
use super::language_model::{LanguageModelMiddleware, wrap_language_model};

/// Two middleware chains: one for language models, one for image models.
///
/// Mirrors the upstream `wrapProvider({ languageModelMiddleware,
/// imageModelMiddleware })` surface in
/// `packages/ai/src/middleware/wrap-provider.ts:20-50`. Upstream
/// **deliberately** does not expose embedding / reranking / video / speech /
/// transcription middleware here — those model surfaces are forwarded
/// verbatim and any wrapping happens via the per-model `wrap_*` helpers.
///
/// Passing an empty `Vec` for a surface leaves that surface untouched.
#[derive(Default, Clone)]
pub struct ProviderMiddlewareSet {
    /// Middleware applied to every [`LanguageModel`] returned by
    /// [`Provider::language_model`].
    pub language: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Middleware applied to every [`ImageModel`] returned by
    /// [`Provider::image_model`].
    pub image: Vec<Arc<dyn ImageModelMiddleware>>,
}

impl std::fmt::Debug for ProviderMiddlewareSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderMiddlewareSet")
            .field("language", &self.language.len())
            .field("image", &self.image.len())
            .finish()
    }
}

/// Wrap a provider so every returned model is decorated with the matching
/// middleware chain.
///
/// Each lookup (`language_model` / `embedding_model` / `image_model`)
/// delegates to the inner provider and, on success, wraps the result with
/// the configured middleware chain. Lookups for unsupported surfaces
/// propagate the inner error unchanged.
///
/// Cloning the middleware set is shallow (each `Arc` is bumped); the cost
/// per lookup is one `Vec::clone` plus the existing `Wrapped` allocations.
pub fn wrap_provider(inner: Arc<dyn Provider>, set: ProviderMiddlewareSet) -> Arc<dyn Provider> {
    Arc::new(WrappedProvider { inner, set })
}

struct WrappedProvider {
    inner: Arc<dyn Provider>,
    set: ProviderMiddlewareSet,
}

impl std::fmt::Debug for WrappedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WrappedProvider")
            .field("inner", &self.inner)
            .field("middleware", &self.set)
            .finish()
    }
}

impl Provider for WrappedProvider {
    fn language_model(&self, model_id: &str) -> Result<DynLanguageModel> {
        let dyn_model = self.inner.language_model(model_id)?;
        if self.set.language.is_empty() {
            return Ok(dyn_model);
        }
        let arc: Arc<dyn LanguageModel> = dyn_model.into_inner();
        let wrapped = wrap_language_model(arc, self.set.language.iter().cloned());
        Ok(DynLanguageModel::from_arc(wrapped))
    }

    fn embedding_model(&self, model_id: &str) -> Result<DynEmbeddingModel> {
        // Mirror upstream `wrap-provider.ts:37` —
        // `embeddingModel: providerV4.embeddingModel` (verbatim forward,
        // no middleware). Callers needing per-call embedding middleware use
        // `wrap_embedding_model` directly on a specific model handle.
        self.inner.embedding_model(model_id)
    }

    fn image_model(&self, model_id: &str) -> Result<DynImageModel> {
        let dyn_model = self.inner.image_model(model_id)?;
        if self.set.image.is_empty() {
            return Ok(dyn_model);
        }
        let arc: Arc<dyn ImageModel> = dyn_model.into_inner();
        let wrapped = wrap_image_model(arc, self.set.image.iter().cloned());
        Ok(DynImageModel::from_arc(wrapped))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;

    use super::*;
    use crate::embedding_model::{EmbedOptions, EmbedResult, EmbeddingModel};
    use crate::language_model::{
        CallOptions, FinishReason, FinishReasonKind, GenerateResult, StreamResult, Usage,
    };

    #[derive(Debug, Default)]
    struct StubLang;

    #[async_trait]
    impl LanguageModel for StubLang {
        fn provider(&self) -> &'static str {
            "stub"
        }
        fn model_id(&self) -> &'static str {
            "stub-lm"
        }
        async fn do_generate(&self, _options: CallOptions) -> Result<GenerateResult> {
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
        async fn do_stream(&self, _options: CallOptions) -> Result<StreamResult> {
            Ok(StreamResult {
                stream: Box::pin(futures::stream::iter(vec![])),
                request: None,
                response: None,
            })
        }
    }

    #[derive(Debug, Default)]
    struct StubEmbed;

    #[async_trait]
    impl EmbeddingModel for StubEmbed {
        fn provider(&self) -> &'static str {
            "stub"
        }
        fn model_id(&self) -> &'static str {
            "stub-em"
        }
        async fn do_embed(&self, _opts: EmbedOptions) -> Result<EmbedResult> {
            Ok(EmbedResult {
                embeddings: vec![],
                usage: None,
                provider_metadata: None,
                request: None,
                response: None,
            })
        }
    }

    #[derive(Debug, Default)]
    struct StubProvider;

    impl Provider for StubProvider {
        fn language_model(&self, _model_id: &str) -> Result<DynLanguageModel> {
            Ok(DynLanguageModel::new(StubLang))
        }
        fn embedding_model(&self, _model_id: &str) -> Result<DynEmbeddingModel> {
            Ok(DynEmbeddingModel::new(StubEmbed))
        }
    }

    /// Counts how many times each surface's middleware ran.
    #[derive(Debug, Default)]
    struct Counter {
        lang_calls: AtomicUsize,
        embed_calls: AtomicUsize,
        last_temp: Mutex<Option<f32>>,
    }

    #[derive(Debug)]
    struct CountingLang(Arc<Counter>);

    #[async_trait]
    impl LanguageModelMiddleware for CountingLang {
        async fn transform_params(
            &self,
            _kind: super::super::language_model::CallKind,
            mut params: CallOptions,
            _inner: &dyn LanguageModel,
        ) -> Result<CallOptions> {
            self.0.lang_calls.fetch_add(1, Ordering::SeqCst);
            params.temperature = Some(0.5);
            *self.0.last_temp.lock().expect("mutex") = params.temperature;
            Ok(params)
        }
    }

    #[tokio::test]
    async fn wraps_language_surface_only_embedding_passes_through() {
        // Mirrors upstream wrap-provider.ts:32-37 — language goes through
        // wrap_language_model, embedding is forwarded verbatim (no
        // wrapping). The middleware never observes the embedding call.
        let counter = Arc::new(Counter::default());
        let set = ProviderMiddlewareSet {
            language: vec![Arc::new(CountingLang(Arc::clone(&counter)))],
            image: vec![],
        };
        let wrapped = wrap_provider(Arc::new(StubProvider), set);

        let lm = wrapped.language_model("anything").expect("language");
        lm.do_generate(CallOptions::default())
            .await
            .expect("generate");
        assert_eq!(counter.lang_calls.load(Ordering::SeqCst), 1);
        assert_eq!(*counter.last_temp.lock().expect("mutex"), Some(0.5));

        // Embedding model is reachable but never routed through the
        // middleware (which is *not* configured at the provider level).
        let em = wrapped.embedding_model("anything").expect("embed");
        em.do_embed(EmbedOptions::default()).await.expect("embed");
        // CountingEmbed was never installed → counter stays at 0.
        assert_eq!(counter.embed_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn unsupported_surface_propagates_inner_error() {
        let set = ProviderMiddlewareSet::default();
        let wrapped = wrap_provider(Arc::new(StubProvider), set);
        let err = wrapped.image_model("x").expect_err("inner unsupported");
        assert!(err.is_unsupported());
    }
}
