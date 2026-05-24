# 0001 — Trait Design (Rust port of ai-sdk v4)

> Status: accepted, M1 in progress
> Upstream reference: `vercel/ai` @ `packages/provider/src/**/v4/*`

## Goal

为 Rust 版 llmsdk 定义最小、稳定、与 ai-sdk v4 行为一致的 provider 抽象层。
作为后续所有 provider 实现 / 提示词 / 契约测试的 **ground truth**。

## 范围

第一阶段覆盖：

- `LanguageModel` (do_generate / do_stream)
- `EmbeddingModel`
- `ImageModel`
- `Provider` (工厂入口)
- `ProviderError`
- 共享类型: `Prompt` / `Message` / `Content` / `StreamPart` / `Usage` / `FinishReason` / `ProviderOptions` / `Warning`

不在第一阶段：

- ~~middleware (`*-middleware/`)~~ — `LanguageModel` middleware 已在 M9 落地，
  设计见 `0002-middleware-design.md`；Embedding / Image middleware 推迟
- reranking / transcription / speech / video
- files / skills 接口
- `tool-approval-*` 流程（暂保留类型，但不强制）

## TS → Rust 映射决策

| TS | Rust | 理由 |
|---|---|---|
| `LanguageModelV4` (object) | `trait LanguageModel` | 自然映射 |
| `specificationVersion: 'v4'` | crate 级常量 `SPECIFICATION_VERSION` | Rust 类型系统已保证；无需 marker |
| `PromiseLike<T>` | `async fn` / `impl Future` | 语义对齐 |
| `ReadableStream<T>` | `Pin<Box<dyn Stream<Item = Result<T, E>> + Send>>` | 标准 futures stream |
| `AbortSignal` | `tokio_util::sync::CancellationToken`（在 utils 提供）；trait 层不直接出现 | 抽象层用 dropping future 即可取消 |
| `JSONValue` | `serde_json::Value` | 一一对应 |
| `JSONSchema7` | `schemars::Schema` (M10+；之前为 `serde_json::Value`) | 类型化 + 支持 `derive(JsonSchema)` 自动生成；wire 透明 |
| `Uint8Array` | `bytes::Bytes` | 零拷贝 |
| Tagged union (`type` 字段) | `enum` + `#[serde(tag = "type", rename_all = "kebab-case")]` | 与 ai-sdk JSON 兼容 |
| `& { providerOptions?: ... }` 交叉类型 | 每个 variant 平铺字段 | 不用 wrapper struct |
| `APICallError` class | `ProviderError` struct + private `ErrorKind` | 符合 M-ERRORS-CANONICAL-STRUCTS |
| `instanceof APICallError` | `error.is_api_call()` 等 helper | 同上 |

## 流的错误语义（关键）

ai-sdk 的 `StreamPart` 里有 `{ type: 'error', error: unknown }` —— **流活着，但中间出错**。
Rust 侧保留两层：

- 外层 `Result::Err(ProviderError)`：流本身挂掉（连接断、解析失败）
- 内层 `StreamPart::Error { error }`：流活着、provider 报告业务错误（content-filter 等）

这是**有意保留的双层**，不要合并。

## Provider trait 的可选方法

TS 用 `transcriptionModel?(modelId)`。Rust 没可选方法 → **默认实现返回 `ProviderError::unsupported(...)`**。
调用方统一 `match` 错误，不暴露 trait 边界差异。

## Provider 返回 dyn

`Provider::language_model` 返回 `DynLanguageModel`（newtype wrapping `Arc<dyn LanguageModel>`），
不直接返回 `Arc<dyn ...>`，符合 M-AVOID-WRAPPERS（自定义 wrapper）。

## 错误类型层级

```
ProviderError                       // 公开 struct
├── kind: ErrorKind                 // 私有 enum
├── backtrace: Backtrace
└── helpers: is_api_call() / is_retryable() / status_code() / ...
```

`ErrorKind` variants（合并自 ai-sdk 18 个 error class）：

- `ApiCall { url, status_code, response_headers, response_body, request_body, is_retryable, source }`
- `InvalidArgument { argument, message }`
- `InvalidPrompt { message }`
- `TypeValidation { path, value }`
- `JsonParse { text }`
- `EmptyResponseBody`
- `NoContentGenerated`
- `NoSuchModel { model_id, model_type }`
- `Unsupported { functionality }`
- `LoadApiKey { message }`
- `TooManyEmbeddingValues { max, actual }`

## 文件 → ai-sdk 对照表

| Rust | ai-sdk |
|---|---|
| `error.rs` | `provider/src/errors/*` |
| `provider.rs` | `provider/src/provider/v4/provider-v4.ts` |
| `language_model/mod.rs` | `provider/src/language-model/v4/language-model-v4.ts` |
| `language_model/call_options.rs` | `language-model-v4-call-options.ts` |
| `language_model/prompt.rs` | `language-model-v4-prompt.ts` |
| `language_model/content.rs` | `language-model-v4-content.ts` + `-text` `-reasoning` `-file` `-source` `-tool-call` `-tool-result` `-tool-approval-request` `-custom-content` `-reasoning-file` |
| `language_model/stream_part.rs` | `language-model-v4-stream-part.ts` |
| `language_model/result.rs` | `-generate-result.ts` + `-stream-result.ts` + `-response-metadata.ts` |
| `language_model/usage.rs` | `language-model-v4-usage.ts` |
| `language_model/finish_reason.rs` | `language-model-v4-finish-reason.ts` |
| `language_model/tool.rs` | `-function-tool.ts` + `-provider-tool.ts` + `-tool-choice.ts` |
| `embedding_model/mod.rs` | `embedding-model/v4/*` |
| `image_model/mod.rs` | `image-model/v4/*` |
| `shared.rs` | `provider/src/shared/v4/*` |
| `json.rs` | `provider/src/json-value/*` |

## 里程碑

- **M1**: `llmsdk-provider` 编译通过；trait + 全部类型签名 ready
- **M2**: `llmsdk-provider-utils` HTTP/SSE/load_api_key
- **M3**: `llmsdk-openai` `do_generate` + 契约测试 `chat_basic`
- **M4**: `llmsdk-openai` `do_stream` + 契约测试 `chat_stream`
- **M5**: `llmsdk-openai` `EmbeddingModel` + 契约测试 `embed_basic`

跨里程碑禁止。

## M10 增量 trait 改动（记录）

`ImageModel`：
- `ImageOptions` 新增 `files: Option<Vec<FilePart>>` + `mask: Option<FilePart>`（edits / variations 端点用，普通 `do_generate` 忽略）
- `ImageResult` 新增 `usage: Option<ImageUsage>`（gpt-image-1 系列报；其它返回 `None`）
- 新增 `ImageUsage { input_tokens, output_tokens, input_tokens_details }` + `ImageUsageInputDetails { text_tokens, image_tokens }`

`JsonSchema` 类型别名：
- 之前：`pub type JsonSchema = serde_json::Value;`
- 现在：`pub type JsonSchema = schemars::Schema;`
- 行为变化：wire 不变；构造方式新增 `schemars::json_schema!` 宏 / `schema_for!` derive；从 raw JSON 构造需要 `serde_json::from_value::<JsonSchema>(v)?`（Schema 在 deserialize 时会验证）

下游影响：
- `llmsdk-openai`：3 处 `.clone()` → `.clone().into()`（wire 仍接 `Value`）；`image.rs` `ImageResult` 加 `usage: None`
- `llmsdk-anthropic`：1 处 `.clone()` → `.clone().into()`
- 测试：2 处 `input_schema: json!(...)` → `input_schema: serde_json::from_value(json!(...)).unwrap()`

## 改动本文档需走的流程

1. 改这份文档的 PR 必须先单独提
2. 改完 → 同步更新 AGENTS.md 中的"移植原则"片段（如适用）
3. 通过后再改代码
