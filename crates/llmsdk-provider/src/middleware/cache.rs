//! Cache middleware backed by a pluggable [`CacheStore`].
//!
//! Cache key is derived from a hash of the JSON-serialized [`CallOptions`].
//! On a hit, generate returns the cached [`GenerateResult`] verbatim and
//! stream replays the captured [`StreamPart`] sequence.
//!
//! # Store contract
//!
//! [`CacheStore`] is intentionally synchronous so the middleware does not
//! force a tokio `rt` dependency on this crate. In-memory backends are a
//! natural fit; remote backends should `spawn` internally if they need to
//! await network I/O.
//!
//! # Stream capture
//!
//! On a miss, [`CacheMiddleware::wrap_stream`] tees each emitted
//! [`StreamPart`] into an internal buffer and commits to the cache **only
//! if the stream completes without an outer `Err`**. Inner
//! [`StreamPart::Error`] frames (the recoverable kind) are part of the
//! stream and therefore cached as-is. Hits annotate
//! `provider_metadata.llmsdk.cache = "hit"` so downstream telemetry can
//! distinguish cached responses from fresh ones.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::Stream;
use serde_json::{Map, Value};

use crate::error::{ProviderError, Result};
#[cfg(test)]
use crate::language_model::TextPart;
use crate::language_model::{
    BoxStream, CallOptions, GenerateResult, LanguageModel, StreamPart, StreamResult,
};

use super::language_model::LanguageModelMiddleware;

/// Backing store for [`CacheMiddleware`].
///
/// Synchronous on purpose — see the module docs.
pub trait CacheStore: Send + Sync + std::fmt::Debug {
    /// Look up an entry by key. `None` is a miss.
    fn get(&self, key: &str) -> Option<CachedEntry>;

    /// Store an entry. Overwrites any existing value for `key`.
    fn put(&self, key: String, value: CachedEntry);
}

/// A cached call result.
///
/// Mirrors the two model call shapes; `Stream` keeps the full part sequence
/// so a hit can be replayed deterministically. `Generate` is boxed to keep
/// the enum size balanced.
#[derive(Debug, Clone)]
pub enum CachedEntry {
    /// Cached [`LanguageModel::do_generate`] result.
    Generate(Box<GenerateResult>),
    /// Cached [`LanguageModel::do_stream`] part sequence.
    Stream(Vec<StreamPart>),
}

/// In-memory [`CacheStore`] with optional TTL and LRU eviction.
///
/// Default constructor builds an unbounded store (no TTL, no LRU). Use
/// [`Self::builder`] to opt in to limits — e.g.
/// `MemoryCacheStore::builder().max_entries(256).max_age(Duration::from_secs(60)).build()`.
///
/// Eviction is checked lazily on `get` (expired entries are removed) and on
/// `put` (over-capacity entries are dropped, least-recently-used first). LRU
/// is approximated with a monotonic counter — no doubly-linked list — which
/// keeps the struct small at the cost of `O(n)` eviction. That's fine for the
/// caches we expect (≤ a few hundred entries).
#[derive(Debug, Default)]
pub struct MemoryCacheStore {
    inner: Mutex<MemoryCacheState>,
}

#[derive(Debug, Default)]
struct MemoryCacheState {
    entries: HashMap<String, CacheEntry>,
    /// Monotonic counter bumped on every put / hit — used for LRU.
    tick: u64,
    /// Optional capacity ceiling.
    max_entries: Option<usize>,
    /// Optional per-entry max age.
    max_age: Option<Duration>,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    value: CachedEntry,
    inserted_at: Instant,
    last_access: u64,
}

impl MemoryCacheStore {
    /// Build an unbounded store (no TTL, no LRU).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a builder for an LRU- / TTL-bounded store.
    #[must_use]
    pub fn builder() -> MemoryCacheStoreBuilder {
        MemoryCacheStoreBuilder::default()
    }

    /// Number of entries currently cached.
    ///
    /// Note: this does *not* prune expired entries first. Use
    /// [`Self::is_empty`] or a fresh `get` to trigger pruning if needed.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex was poisoned by a prior panic.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .expect("cache mutex poisoned")
            .entries
            .len()
    }

    /// `true` when the store has no entries.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex was poisoned by a prior panic.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner
            .lock()
            .expect("cache mutex poisoned")
            .entries
            .is_empty()
    }
}

/// Builder for [`MemoryCacheStore`] with TTL / LRU options.
#[derive(Debug, Default, Clone, Copy)]
pub struct MemoryCacheStoreBuilder {
    max_entries: Option<usize>,
    max_age: Option<Duration>,
}

impl MemoryCacheStoreBuilder {
    /// Cap the number of cached entries; least-recently-used are evicted first.
    #[must_use]
    pub fn max_entries(mut self, n: usize) -> Self {
        self.max_entries = Some(n);
        self
    }

    /// Drop entries older than `max_age` on the next `get` that touches them
    /// (or earlier, opportunistically on inserts that hit the capacity).
    #[must_use]
    pub fn max_age(mut self, age: Duration) -> Self {
        self.max_age = Some(age);
        self
    }

    /// Finalize.
    #[must_use]
    pub fn build(self) -> MemoryCacheStore {
        MemoryCacheStore {
            inner: Mutex::new(MemoryCacheState {
                entries: HashMap::new(),
                tick: 0,
                max_entries: self.max_entries,
                max_age: self.max_age,
            }),
        }
    }
}

impl MemoryCacheState {
    fn touch(&mut self) -> u64 {
        self.tick = self.tick.saturating_add(1);
        self.tick
    }

    /// Evict the single least-recently-used entry. No-op if empty.
    fn evict_one_lru(&mut self) {
        let victim = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_access)
            .map(|(k, _)| k.clone());
        if let Some(k) = victim {
            self.entries.remove(&k);
        }
    }

    /// Remove any expired entries opportunistically.
    fn prune_expired(&mut self) {
        let Some(age) = self.max_age else {
            return;
        };
        let now = Instant::now();
        self.entries
            .retain(|_, e| now.duration_since(e.inserted_at) <= age);
    }
}

impl CacheStore for MemoryCacheStore {
    fn get(&self, key: &str) -> Option<CachedEntry> {
        let mut guard = self.inner.lock().expect("cache mutex poisoned");
        // Lazy TTL check on the requested key.
        if let Some(age) = guard.max_age
            && let Some(entry) = guard.entries.get(key)
            && Instant::now().duration_since(entry.inserted_at) > age
        {
            guard.entries.remove(key);
            return None;
        }
        let tick = guard.touch();
        let entry = guard.entries.get_mut(key)?;
        entry.last_access = tick;
        Some(entry.value.clone())
    }

    fn put(&self, key: String, value: CachedEntry) {
        let mut guard = self.inner.lock().expect("cache mutex poisoned");
        guard.prune_expired();
        let tick = guard.touch();
        let new_entry = CacheEntry {
            value,
            inserted_at: Instant::now(),
            last_access: tick,
        };
        guard.entries.insert(key, new_entry);
        if let Some(cap) = guard.max_entries {
            while guard.entries.len() > cap {
                guard.evict_one_lru();
            }
        }
    }
}

/// Middleware that memoizes generate / stream responses keyed by
/// [`CallOptions`].
///
/// Combine with retry / logging via [`super::wrap_language_model`]; the
/// recommended order is `[logging, retry, cache, model]` so cache hits skip
/// retries entirely and logging records both fresh and cached calls.
#[derive(Debug, Clone)]
pub struct CacheMiddleware {
    store: Arc<dyn CacheStore>,
}

impl CacheMiddleware {
    /// Build a middleware backed by `store`.
    #[must_use]
    pub fn new(store: Arc<dyn CacheStore>) -> Self {
        Self { store }
    }
}

/// Hash the JSON-serialized call options into a 16-hex cache key.
///
/// Uses `std::hash::DefaultHasher`. The 64-bit output is enough for typical
/// in-process caches; swap to a stronger hash in a custom `CacheStore` if
/// you persist across processes.
fn key_for(options: &CallOptions) -> Result<String> {
    let bytes = serde_json::to_vec(options)
        .map_err(|e| ProviderError::type_validation("call_options", Value::Null, e.to_string()))?;
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    Ok(format!("{:016x}", hasher.finish()))
}

/// Annotate a `GenerateResult` so downstream telemetry can tell a hit from a
/// fresh call.
fn mark_generate_hit(result: &mut GenerateResult) {
    let entry = result.provider_metadata.get_or_insert_with(HashMap::new);
    let bucket = entry.entry("llmsdk".to_owned()).or_default();
    bucket.insert("cache".to_owned(), Value::String("hit".to_owned()));
}

/// Build the `provider_metadata.llmsdk.cache = "hit"` payload to inject
/// into the first stream frame on a cache hit.
fn hit_metadata() -> crate::shared::ProviderMetadata {
    let mut map: crate::shared::ProviderMetadata = HashMap::new();
    let mut bucket = Map::new();
    bucket.insert("cache".to_owned(), Value::String("hit".to_owned()));
    map.insert("llmsdk".to_owned(), bucket);
    map
}

/// Inject the hit marker into the first part that carries
/// `provider_metadata`, otherwise prepend a dedicated frame.
///
/// We try to avoid changing the part count so callers that count parts
/// observe the same shape on hit vs miss.
fn annotate_stream_hit(parts: &mut Vec<StreamPart>) {
    for part in parts.iter_mut() {
        if matches!(part, StreamPart::StreamStart { .. }) {
            continue;
        }
        if inject_metadata(part, &hit_metadata()) {
            return;
        }
    }
    parts.insert(
        0,
        StreamPart::Custom {
            kind: "llmsdk.cache.hit".to_owned(),
            provider_metadata: Some(hit_metadata()),
        },
    );
}

/// Merge `mark` into the part's `provider_metadata`, if any. Returns `true`
/// on success.
fn inject_metadata(part: &mut StreamPart, mark: &crate::shared::ProviderMetadata) -> bool {
    let (StreamPart::TextStart {
        provider_metadata: slot,
        ..
    }
    | StreamPart::TextDelta {
        provider_metadata: slot,
        ..
    }
    | StreamPart::TextEnd {
        provider_metadata: slot,
        ..
    }
    | StreamPart::ReasoningStart {
        provider_metadata: slot,
        ..
    }
    | StreamPart::ReasoningDelta {
        provider_metadata: slot,
        ..
    }
    | StreamPart::ReasoningEnd {
        provider_metadata: slot,
        ..
    }
    | StreamPart::ToolInputStart {
        provider_metadata: slot,
        ..
    }
    | StreamPart::ToolInputDelta {
        provider_metadata: slot,
        ..
    }
    | StreamPart::ToolInputEnd {
        provider_metadata: slot,
        ..
    }
    | StreamPart::Custom {
        provider_metadata: slot,
        ..
    }
    | StreamPart::Finish {
        provider_metadata: slot,
        ..
    }) = part
    else {
        return false;
    };
    let target = slot.get_or_insert_with(HashMap::new);
    for (provider, bucket) in mark {
        let dest = target.entry(provider.clone()).or_default();
        for (k, v) in bucket {
            dest.insert(k.clone(), v.clone());
        }
    }
    true
}

#[async_trait]
impl LanguageModelMiddleware for CacheMiddleware {
    async fn wrap_generate(
        &self,
        next: &dyn LanguageModel,
        params: CallOptions,
    ) -> Result<GenerateResult> {
        let key = key_for(&params)?;
        if let Some(CachedEntry::Generate(mut hit)) = self.store.get(&key) {
            mark_generate_hit(&mut hit);
            return Ok(*hit);
        }
        let result = next.do_generate(params).await?;
        self.store
            .put(key, CachedEntry::Generate(Box::new(result.clone())));
        Ok(result)
    }

    async fn wrap_stream(
        &self,
        next: &dyn LanguageModel,
        params: CallOptions,
    ) -> Result<StreamResult> {
        let key = key_for(&params)?;
        if let Some(CachedEntry::Stream(mut parts)) = self.store.get(&key) {
            annotate_stream_hit(&mut parts);
            let stream = futures::stream::iter(parts.into_iter().map(Ok));
            return Ok(StreamResult {
                stream: Box::pin(stream),
                request: None,
                response: None,
            });
        }
        let StreamResult {
            stream,
            request,
            response,
        } = next.do_stream(params).await?;
        let capturing = CapturingStream::new(stream, Arc::clone(&self.store), key);
        Ok(StreamResult {
            stream: Box::pin(capturing),
            request,
            response,
        })
    }
}

/// Stream wrapper that tees each `Ok` part into a buffer; commits to the
/// cache when the inner stream completes without an outer `Err`.
struct CapturingStream {
    inner: BoxStream<Result<StreamPart>>,
    store: Arc<dyn CacheStore>,
    key: Option<String>,
    captured: Vec<StreamPart>,
    poisoned: bool,
}

impl CapturingStream {
    fn new(inner: BoxStream<Result<StreamPart>>, store: Arc<dyn CacheStore>, key: String) -> Self {
        Self {
            inner,
            store,
            key: Some(key),
            captured: Vec::new(),
            poisoned: false,
        }
    }
}

impl Stream for CapturingStream {
    type Item = Result<StreamPart>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let polled = self.inner.as_mut().poll_next(cx);
        match &polled {
            Poll::Ready(Some(Ok(part))) => {
                self.captured.push(part.clone());
            }
            Poll::Ready(Some(Err(_))) => {
                self.poisoned = true;
            }
            Poll::Ready(None) => {
                if !self.poisoned
                    && let Some(key) = self.key.take()
                {
                    let captured = std::mem::take(&mut self.captured);
                    self.store.put(key, CachedEntry::Stream(captured));
                }
            }
            Poll::Pending => {}
        }
        polled
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures::StreamExt;

    use crate::language_model::{Content, FinishReason, FinishReasonKind, Usage};

    use super::*;

    #[derive(Debug)]
    struct CountingModel {
        provider: String,
        model_id: String,
        generate_calls: AtomicUsize,
        stream_calls: AtomicUsize,
    }

    impl CountingModel {
        fn new() -> Self {
            Self {
                provider: "test".to_owned(),
                model_id: "counter".to_owned(),
                generate_calls: AtomicUsize::new(0),
                stream_calls: AtomicUsize::new(0),
            }
        }
    }

    fn ok_generate(text: &str) -> GenerateResult {
        GenerateResult {
            content: vec![Content::Text(TextPart {
                text: text.to_owned(),
                provider_options: None,
            })],
            finish_reason: FinishReason::new(FinishReasonKind::Stop),
            usage: Usage::default(),
            provider_metadata: None,
            request: None,
            response: None,
            warnings: vec![],
        }
    }

    #[async_trait]
    impl LanguageModel for CountingModel {
        fn provider(&self) -> &str {
            &self.provider
        }
        fn model_id(&self) -> &str {
            &self.model_id
        }
        async fn do_generate(&self, _opts: CallOptions) -> Result<GenerateResult> {
            self.generate_calls.fetch_add(1, Ordering::SeqCst);
            Ok(ok_generate("hello"))
        }
        async fn do_stream(&self, _opts: CallOptions) -> Result<StreamResult> {
            self.stream_calls.fetch_add(1, Ordering::SeqCst);
            let parts = vec![
                Ok(StreamPart::StreamStart { warnings: vec![] }),
                Ok(StreamPart::TextStart {
                    id: "0".to_owned(),
                    provider_metadata: None,
                }),
                Ok(StreamPart::TextDelta {
                    id: "0".to_owned(),
                    delta: "hi".to_owned(),
                    provider_metadata: None,
                }),
                Ok(StreamPart::TextEnd {
                    id: "0".to_owned(),
                    provider_metadata: None,
                }),
                Ok(StreamPart::Finish {
                    usage: Usage::default(),
                    finish_reason: FinishReason::new(FinishReasonKind::Stop),
                    provider_metadata: None,
                }),
            ];
            Ok(StreamResult {
                stream: Box::pin(futures::stream::iter(parts)),
                request: None,
                response: None,
            })
        }
    }

    #[derive(Debug)]
    struct FailingStreamModel {
        provider: String,
        model_id: String,
    }

    impl Default for FailingStreamModel {
        fn default() -> Self {
            Self {
                provider: "test".to_owned(),
                model_id: "fail-stream".to_owned(),
            }
        }
    }

    #[async_trait]
    impl LanguageModel for FailingStreamModel {
        fn provider(&self) -> &str {
            &self.provider
        }
        fn model_id(&self) -> &str {
            &self.model_id
        }
        async fn do_generate(&self, _opts: CallOptions) -> Result<GenerateResult> {
            Ok(ok_generate(""))
        }
        async fn do_stream(&self, _opts: CallOptions) -> Result<StreamResult> {
            let parts: Vec<Result<StreamPart>> = vec![
                Ok(StreamPart::StreamStart { warnings: vec![] }),
                Err(ProviderError::empty_response_body()),
            ];
            Ok(StreamResult {
                stream: Box::pin(futures::stream::iter(parts)),
                request: None,
                response: None,
            })
        }
    }

    #[tokio::test]
    async fn generate_second_call_hits_cache() {
        let store = Arc::new(MemoryCacheStore::new());
        let mw = CacheMiddleware::new(Arc::clone(&store) as Arc<dyn CacheStore>);
        let model = CountingModel::new();

        let first = mw
            .wrap_generate(&model, CallOptions::default())
            .await
            .expect("first call");
        assert!(first.provider_metadata.is_none(), "miss is not annotated");

        let second = mw
            .wrap_generate(&model, CallOptions::default())
            .await
            .expect("second call");
        assert_eq!(model.generate_calls.load(Ordering::SeqCst), 1);
        let llmsdk = second
            .provider_metadata
            .as_ref()
            .and_then(|m| m.get("llmsdk"))
            .expect("hit metadata present");
        assert_eq!(llmsdk.get("cache"), Some(&Value::String("hit".to_owned())));
        assert_eq!(store.len(), 1);
    }

    #[tokio::test]
    async fn stream_second_call_replays_cached_parts() {
        let store = Arc::new(MemoryCacheStore::new());
        let mw = CacheMiddleware::new(Arc::clone(&store) as Arc<dyn CacheStore>);
        let model = CountingModel::new();

        // First call — drain to trigger commit.
        let first = mw
            .wrap_stream(&model, CallOptions::default())
            .await
            .expect("first stream");
        let first_parts: Vec<_> = first
            .stream
            .filter_map(|r| async move { r.ok() })
            .collect()
            .await;
        assert_eq!(first_parts.len(), 5);
        assert_eq!(model.stream_calls.load(Ordering::SeqCst), 1);
        assert_eq!(store.len(), 1, "stream committed after Ok completion");

        // Second call — replay from cache.
        let second = mw
            .wrap_stream(&model, CallOptions::default())
            .await
            .expect("second stream");
        let second_parts: Vec<_> = second
            .stream
            .filter_map(|r| async move { r.ok() })
            .collect()
            .await;
        assert_eq!(
            model.stream_calls.load(Ordering::SeqCst),
            1,
            "no second call"
        );
        assert_eq!(second_parts.len(), first_parts.len());

        // Hit marker landed somewhere with provider_metadata.
        let any_hit = second_parts.iter().any(|p| match p {
            StreamPart::TextStart {
                provider_metadata, ..
            }
            | StreamPart::TextDelta {
                provider_metadata, ..
            }
            | StreamPart::TextEnd {
                provider_metadata, ..
            }
            | StreamPart::Finish {
                provider_metadata, ..
            } => {
                provider_metadata
                    .as_ref()
                    .and_then(|m| m.get("llmsdk"))
                    .and_then(|b| b.get("cache"))
                    == Some(&Value::String("hit".to_owned()))
            }
            _ => false,
        });
        assert!(any_hit, "cache hit marker must be visible on replay");
    }

    #[tokio::test]
    async fn stream_does_not_cache_when_inner_errors() {
        let store = Arc::new(MemoryCacheStore::new());
        let mw = CacheMiddleware::new(Arc::clone(&store) as Arc<dyn CacheStore>);
        let model = FailingStreamModel::default();

        let result = mw
            .wrap_stream(&model, CallOptions::default())
            .await
            .expect("open succeeds");
        let parts: Vec<Result<StreamPart>> = result.stream.collect().await;
        assert_eq!(parts.len(), 2, "one Ok + one Err drained");
        assert!(parts[1].is_err());
        assert!(store.is_empty(), "must not cache a poisoned stream");
    }

    #[tokio::test]
    async fn generate_failure_is_not_cached() {
        #[derive(Debug)]
        struct AlwaysFail {
            provider: String,
            model_id: String,
        }
        #[async_trait]
        impl LanguageModel for AlwaysFail {
            fn provider(&self) -> &str {
                &self.provider
            }
            fn model_id(&self) -> &str {
                &self.model_id
            }
            async fn do_generate(&self, _opts: CallOptions) -> Result<GenerateResult> {
                Err(ProviderError::empty_response_body())
            }
            async fn do_stream(&self, _opts: CallOptions) -> Result<StreamResult> {
                unreachable!()
            }
        }
        let model = AlwaysFail {
            provider: "test".to_owned(),
            model_id: "fail".to_owned(),
        };
        let store = Arc::new(MemoryCacheStore::new());
        let mw = CacheMiddleware::new(Arc::clone(&store) as Arc<dyn CacheStore>);
        let _ = mw.wrap_generate(&model, CallOptions::default()).await;
        assert!(store.is_empty());
    }

    #[test]
    fn key_is_stable_for_equal_options() {
        let a = CallOptions::default();
        let b = CallOptions::default();
        assert_eq!(key_for(&a).unwrap(), key_for(&b).unwrap());
    }

    #[test]
    fn key_differs_when_temperature_changes() {
        let a = CallOptions {
            temperature: Some(0.1),
            ..CallOptions::default()
        };
        let b = CallOptions {
            temperature: Some(0.9),
            ..CallOptions::default()
        };
        assert_ne!(key_for(&a).unwrap(), key_for(&b).unwrap());
    }

    fn dummy_entry() -> CachedEntry {
        CachedEntry::Generate(Box::new(ok_generate("hello")))
    }

    #[test]
    fn lru_evicts_oldest_entry_over_capacity() {
        let store = MemoryCacheStore::builder().max_entries(2).build();
        store.put("a".into(), dummy_entry());
        store.put("b".into(), dummy_entry());
        // Touch "a" so "b" becomes least-recently-used.
        let _ = store.get("a");
        store.put("c".into(), dummy_entry());

        assert!(store.get("a").is_some(), "a still present after touch");
        assert!(store.get("b").is_none(), "b evicted as LRU");
        assert!(store.get("c").is_some(), "c just inserted");
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn ttl_expires_entries_on_get() {
        let store = MemoryCacheStore::builder()
            .max_age(Duration::from_millis(10))
            .build();
        store.put("a".into(), dummy_entry());
        std::thread::sleep(Duration::from_millis(20));
        assert!(store.get("a").is_none(), "expired entry pruned");
        assert_eq!(store.len(), 0);
    }
}
