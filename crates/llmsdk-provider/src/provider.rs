//! Top-level [`Provider`] trait — the factory for model instances.
//!
//! Mirrors `provider-v4.ts`. ai-sdk uses optional methods
//! (`transcriptionModel?`, `speechModel?`, ...); Rust has none, so we
//! ship default impls that return [`crate::ProviderError::unsupported`].
//! Callers branch on [`crate::ProviderError::is_unsupported`].
// Rust guideline compliant 2026-02-21

use std::sync::Arc;

use crate::embedding_model::EmbeddingModel;
use crate::error::Result;
use crate::image_model::ImageModel;
use crate::language_model::LanguageModel;

/// Factory returning model instances by id.
///
/// Implementations typically hold a single `reqwest::Client` and shared
/// auth state, then mint thin model wrappers on demand.
pub trait Provider: Send + Sync + std::fmt::Debug {
    /// Look up a language model by id.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ProviderError::no_such_model`] if the id is unknown.
    fn language_model(&self, model_id: &str) -> Result<DynLanguageModel>;

    /// Look up a text-embedding model by id.
    ///
    /// # Errors
    ///
    /// Defaults to [`crate::ProviderError::unsupported`]; override when
    /// the provider supports embeddings. Implementations should return
    /// [`crate::ProviderError::no_such_model`] for unknown ids.
    fn embedding_model(&self, _model_id: &str) -> Result<DynEmbeddingModel> {
        Err(crate::ProviderError::unsupported("embedding_model"))
    }

    /// Look up an image-generation model by id.
    ///
    /// # Errors
    ///
    /// Same conventions as [`Self::embedding_model`].
    fn image_model(&self, _model_id: &str) -> Result<DynImageModel> {
        Err(crate::ProviderError::unsupported("image_model"))
    }
}

/// Type-erased language model handle.
///
/// Newtype over `Arc<dyn LanguageModel>` so the API does not leak a smart
/// pointer (see M-AVOID-WRAPPERS). Cheap to clone; implements
/// [`LanguageModel`] by delegation so callers can use it directly.
#[derive(Debug, Clone)]
pub struct DynLanguageModel(Arc<dyn LanguageModel>);

impl DynLanguageModel {
    /// Wrap a concrete model implementation.
    pub fn new<M: LanguageModel + 'static>(model: M) -> Self {
        Self(Arc::new(model))
    }

    /// Wrap an already-shared `Arc`.
    #[must_use]
    pub fn from_arc(model: Arc<dyn LanguageModel>) -> Self {
        Self(model)
    }

    /// Consume the wrapper and return the underlying `Arc`.
    #[must_use]
    pub fn into_inner(self) -> Arc<dyn LanguageModel> {
        self.0
    }
}

impl std::ops::Deref for DynLanguageModel {
    type Target = dyn LanguageModel;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

/// Type-erased embedding model handle. See [`DynLanguageModel`] for rationale.
#[derive(Debug, Clone)]
pub struct DynEmbeddingModel(Arc<dyn EmbeddingModel>);

impl DynEmbeddingModel {
    /// Wrap a concrete model implementation.
    pub fn new<M: EmbeddingModel + 'static>(model: M) -> Self {
        Self(Arc::new(model))
    }

    /// Wrap an already-shared `Arc`.
    #[must_use]
    pub fn from_arc(model: Arc<dyn EmbeddingModel>) -> Self {
        Self(model)
    }

    /// Consume the wrapper and return the underlying `Arc`.
    #[must_use]
    pub fn into_inner(self) -> Arc<dyn EmbeddingModel> {
        self.0
    }
}

impl std::ops::Deref for DynEmbeddingModel {
    type Target = dyn EmbeddingModel;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

/// Type-erased image model handle. See [`DynLanguageModel`] for rationale.
#[derive(Debug, Clone)]
pub struct DynImageModel(Arc<dyn ImageModel>);

impl DynImageModel {
    /// Wrap a concrete model implementation.
    pub fn new<M: ImageModel + 'static>(model: M) -> Self {
        Self(Arc::new(model))
    }

    /// Wrap an already-shared `Arc`.
    #[must_use]
    pub fn from_arc(model: Arc<dyn ImageModel>) -> Self {
        Self(model)
    }

    /// Consume the wrapper and return the underlying `Arc`.
    #[must_use]
    pub fn into_inner(self) -> Arc<dyn ImageModel> {
        self.0
    }
}

impl std::ops::Deref for DynImageModel {
    type Target = dyn ImageModel;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}
