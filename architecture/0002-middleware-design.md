# 0002 — Middleware Design (LanguageModel, M9 阶段)

> Status: M9 完成；M10 在三个模型表面（Language/Embedding/Image）+ 顶层
> Provider 复制了同形态（见 `0003-m10-design.md`），原 LanguageModel
> middleware 形态零破坏。
> Upstream reference: `vercel/ai` @ `packages/provider/src/language-model-middleware/v4/*` +
> `packages/ai/src/middleware/*`

## Goal

为 `LanguageModel` 引入 middleware 装饰器层，使下列横切关注点可以叠在任意 provider
之上而不修改 provider 实现：

- retry（指数退避，仅对 `ProviderError::is_retryable()` 触发）
- 结构化日志（不绑死日志后端，调用方自行实现 `Logger`）
- 响应缓存（in-memory `CacheStore` + 用户可注入实现；流式以收集 → 回放方式支持）

是 M9 首轮范围。Embedding / Image 表面 middleware 推迟到下一轮（trait 已经验证可
组合后再克隆同样形态过去）。

## 范围

第一轮覆盖：

- `LanguageModelMiddleware` trait（默认 no-op）
- `wrap_language_model(model, middleware)` 组合函数
- 三个内置实现：
  - `RetryMiddleware`
  - `LoggingMiddleware` + `Logger` trait
  - `CacheMiddleware` + `CacheStore` trait + in-memory 实现

第一轮**不**覆盖：

- Embedding / Image middleware（trait 一致化后再补）
- `WrapProvider`（top-level 工厂的批量装饰）—— 等三种模型表面都有 middleware 再做
- `default_settings` / `extract_reasoning` / `simulate_streaming` 等 ai-sdk
  内置 middleware（先证明 trait 形态成立，再按需补）
- middleware 之间共享上下文（trace span / request id 透传）

## TS → Rust 映射决策

| TS | Rust | 理由 |
|---|---|---|
| `LanguageModelV4Middleware` (object, optional methods) | `trait LanguageModelMiddleware` + 默认 no-op 方法 | Rust 没 optional method；默认实现等价 |
| `transformParams({type, params, model})` | `async fn transform_params(&self, kind: CallKind, params: CallOptions, model: &dyn LanguageModel) -> Result<CallOptions>` | `type: 'generate' \| 'stream'` → `enum CallKind` |
| `wrapGenerate({doGenerate, doStream, params, model})` | `async fn wrap_generate(&self, next: &dyn LanguageModel, params: CallOptions) -> Result<GenerateResult>` | next 是下一层（不是原始 model）；middleware 可选择 `next.do_generate` 或 `next.do_stream` |
| `wrapStream({...})` | `async fn wrap_stream(&self, next: &dyn LanguageModel, params: CallOptions) -> Result<StreamResult>` | 同上 |
| `overrideProvider({model}) => string` | `fn override_provider(&self, inner: &dyn LanguageModel) -> Option<String>` | 同步 + `Option`（不覆盖时返回 `None`） |
| `overrideModelId` | `fn override_model_id(&self, inner: &dyn LanguageModel) -> Option<String>` | 同 |
| `overrideSupportedUrls` | `async fn override_supported_urls(&self, inner: &dyn LanguageModel) -> Option<SupportedUrls>` | 异步（与 trait 上的 `supported_urls` 一致） |
| `wrapLanguageModel({model, middleware[]})` | `fn wrap_language_model(model: Arc<dyn LanguageModel>, middleware: impl IntoIterator<Item = Arc<dyn LanguageModelMiddleware>>) -> Arc<dyn LanguageModel>` | reduce 顺序：列表末尾最贴近 model，开头最外层 |
| `LanguageModelV2 \| V3 \| V4` 兼容 | 不实现 | 我们只有 v4 |

### 为什么把 `doGenerate` / `doStream` 合并成 `next: &dyn LanguageModel`

ai-sdk 同时传 `doGenerate` 和 `doStream`，是为了让一个 middleware 决定要用哪
个底层调用（典型场景：simulate-streaming 在 `wrapStream` 里调 `doGenerate`）。
Rust 用 trait object 一并表达更自然：middleware 只要拿到 `next`，想要哪条路径
就调哪个方法。少一层闭包，避免 async-closure 生命周期纠葛。

## Trait 形态

```rust
#[async_trait]
pub trait LanguageModelMiddleware: Send + Sync + std::fmt::Debug {
    fn override_provider(&self, _inner: &dyn LanguageModel) -> Option<String> { None }
    fn override_model_id(&self, _inner: &dyn LanguageModel) -> Option<String> { None }

    async fn override_supported_urls(
        &self,
        _inner: &dyn LanguageModel,
    ) -> Option<SupportedUrls> { None }

    async fn transform_params(
        &self,
        _kind: CallKind,
        params: CallOptions,
        _inner: &dyn LanguageModel,
    ) -> Result<CallOptions> { Ok(params) }

    async fn wrap_generate(
        &self,
        next: &dyn LanguageModel,
        params: CallOptions,
    ) -> Result<GenerateResult> {
        next.do_generate(params).await
    }

    async fn wrap_stream(
        &self,
        next: &dyn LanguageModel,
        params: CallOptions,
    ) -> Result<StreamResult> {
        next.do_stream(params).await
    }
}

pub enum CallKind { Generate, Stream }
```

全部方法都有默认实现 → 单一关注点 middleware 只重写感兴趣的方法。

## 组合实现

`wrap_language_model` 把 `(model, [m1, m2, m3])` 包成新的 `Arc<dyn LanguageModel>`。
组合顺序：**列表末尾最贴近真实 model，开头最外层**（和 ai-sdk 相同；调用顺序：
`m1.transform → m2.transform → m3.transform → m3.wrap → m2.wrap → m1.wrap → model`）。

实现思路：

```rust
let wrapped = middleware
    .into_iter()
    .rev()
    .fold(model, |inner, mw| Arc::new(Wrapped::new(inner, mw)));
```

`Wrapped<M: LanguageModelMiddleware>` 内部持有 `Arc<dyn LanguageModel>` + `Arc<M>`，
实现 `LanguageModel`：

- `provider()` / `model_id()` / `supported_urls()`：先问 middleware 的 override，
  否则回落到 inner。
- `do_generate(p)`：先 `transform_params(Generate, p, inner)` → `wrap_generate(inner, p)`。
- `do_stream(p)`：先 `transform_params(Stream, p, inner)` → `wrap_stream(inner, p)`。

`provider()` / `model_id()` 是 `&str`，override 返回 `String` —— `Wrapped`
内部缓存（在构造时一次性算好），避免每次借用临时 `String`。

## 内置 middleware 设计

### RetryMiddleware

```rust
pub struct RetryMiddleware {
    max_attempts: u32,        // 默认 3
    initial_backoff: Duration,// 默认 100ms
    backoff_multiplier: f32,  // 默认 2.0
    max_backoff: Duration,    // 默认 5s
}
```

- 只重试 `ProviderError::is_retryable() == true` 的错误（其它直接抛）。
- `wrap_generate`：失败 → 退避 → 重试。
- `wrap_stream`：只在**打开 stream 前**重试（即 `next.do_stream` 返回 `Err`
  的情况）；stream 中途的 `Err` 不重试（保持流语义，调用方决定是否重新发起）。
- 退避使用 `tokio::time::sleep`；不引入 `rand`（首版不抖动，未来必要时再加）。

### LoggingMiddleware

为了不引入 `tracing` 依赖，定义自有 `Logger` trait：

```rust
pub trait Logger: Send + Sync + std::fmt::Debug {
    fn log_call_start(&self, event: &LogCallStart<'_>);
    fn log_call_end(&self, event: &LogCallEnd<'_>);
    fn log_call_error(&self, event: &LogCallError<'_>);
}

pub struct LoggingMiddleware {
    logger: Arc<dyn Logger>,
    log_prompt: bool,    // 默认 false（PII / 体积考虑）
}
```

- 内置一个 `StderrLogger` 作为示例 + 测试钩子；用户可在自己的 crate 里实现接
  `tracing` / `log` / 任何系统。
- 事件字段：`provider`, `model_id`, `call_kind`, `started_at`, `elapsed`,
  `usage`（仅 generate 完成时）, `finish_reason`（同）, `error_kind`。

### CacheMiddleware

```rust
#[async_trait]
pub trait CacheStore: Send + Sync + std::fmt::Debug {
    async fn get(&self, key: &str) -> Option<CachedEntry>;
    async fn put(&self, key: String, value: CachedEntry);
}

pub enum CachedEntry {
    Generate(GenerateResult),
    Stream(Vec<StreamPart>),
}

pub struct CacheMiddleware {
    store: Arc<dyn CacheStore>,
}
```

- key = `sha256(serde_json::to_vec(&CallOptions)?)` 的 hex（也可以更轻的哈希；
  AGENTS.md 禁新依赖 → 使用 `std::hash::DefaultHasher` 的 64-bit 输出 + hex
  即可，不引 sha2）。第一版用 DefaultHasher。
- `wrap_generate`：命中 → 直接返回；未命中 → 调 `next.do_generate`，成功后写入。
- `wrap_stream`：
  - 命中 → 用 `tokio::sync::mpsc::unbounded_channel` 把缓存的 `Vec<StreamPart>`
    回放成一个新 `BoxStream`，包成 `StreamResult` 返回（`request` / `response`
    字段为 `None`，因为没有真实 HTTP 请求）。
  - 未命中 → `next.do_stream`，把 stream 收集到 `Vec<StreamPart>`（边收集边
    转发）；流终止后异步写入缓存（用 `tokio::spawn`）。
- 失败（任何 `Err`）不写缓存。
- in-memory 实现：`MemoryCacheStore { inner: Mutex<HashMap<String, CachedEntry>> }`。

**已知权衡 / 推迟**：

- TTL / LRU 没有第一版；MemoryCacheStore 是无限增长的 HashMap，仅作测试 / 教学。
  用户接 Redis / etc 自行实现。
- key 不包含 `Headers`（HTTP 透传字段属调用上下文，不属语义输入）。
- 命中时 stream 的 `request`/`response` 为 `None`；如调用方依赖 telemetry，
  缓存层会标注一个 `provider_metadata.llmsdk.cache = "hit"`。

## 模块布局

```
crates/llmsdk-provider/src/
├── language_model/...           # 现有
├── middleware/
│   ├── mod.rs                   # 公开导出 + wrap_language_model
│   ├── language_model.rs        # LanguageModelMiddleware trait + Wrapped
│   ├── retry.rs                 # RetryMiddleware
│   ├── logging.rs               # LoggingMiddleware + Logger trait + StderrLogger
│   └── cache.rs                 # CacheMiddleware + CacheStore + MemoryCacheStore
└── lib.rs                       # `pub mod middleware;` + 重新导出常用项
```

`lib.rs` 顶层重新导出：

```rust
pub use middleware::{
    CacheMiddleware, CacheStore, CachedEntry, CallKind, LanguageModelMiddleware,
    LoggingMiddleware, Logger, MemoryCacheStore, RetryMiddleware, wrap_language_model,
};
```

## 改动范围 / 不动什么

- **不动** `LanguageModel` / `EmbeddingModel` / `ImageModel` trait 现有签名。
- **不动** `Provider` trait（顶层 wrap 留到三种表面齐全后再统一设计）。
- **不动** provider 实现（openai / anthropic 不感知 middleware）。
- 新增：`middleware/` 模块 + 6 个公开类型 / trait（一次性出，避免后续微调）。

## 测试策略

- **trait + wrap**：MockLanguageModel + 一个 stub middleware，覆盖
  - override_provider/model_id 生效
  - transform_params 修改后 inner 收到的是修改后的 params
  - wrap 调用顺序 (m1.wrap 在 m2.wrap 外层)
- **retry**：MockLanguageModel 前 N 次返回 retryable error，第 N+1 次成功；
  断言 `max_attempts` 边界 + 非 retryable 立刻抛。
- **logging**：用自定义 `Logger` 收集事件到 `Mutex<Vec<...>>`；断言调用次数 + 字段。
- **cache**：
  - generate 第二次命中（断言 inner 只被调一次）。
  - stream 第一次收集 + 第二次回放（断言序列相同）。
  - 错误不缓存。
- 集成 sanity：openai contract test 通过 `wrap_language_model` 包一层 retry +
  cache，跑 `chat_basic` 不退化（不真打 OpenAI；用 mock server 即可）。

## 里程碑

- **M9.1**: 0002 文档定稿 ← **本 PR**
- **M9.2**: trait + wrap_language_model + 单测 → 停下来等审核
- **M9.3**: retry + logging + cache 内置实现 + 单测
- **M9.4**: 整合 sanity test；todo.md / AGENTS.md / 0001 同步

## 改动本文档需走的流程

1. 改这份文档的 PR 必须先单独提
2. 改完 → 同步更新 AGENTS.md 中的"移植原则"或里程碑片段（如适用）
3. 通过后再改代码
