---
description: 接入一个新的 AI provider（实现 LanguageModel/EmbeddingModel/ImageModel）。用法 /provider <name> <capability>。例：/provider openai text，/provider anthropic stream。
---

你在运行 **/provider 工作流**，给一个 provider 加一项能力。**一次只做一项能力**，不准同时做 text + stream。

## 输入

- `$ARGUMENTS` 格式：`<provider name> <capability>`
- `capability` 取值：`text` / `stream` / `tool` / `embed` / `image`

格式不对就停下问。

## 前置条件（必须先检查）

1. `architecture/0001-trait-design.md` 当前里程碑允许做这个 capability 吗？
   - M3 只准做 openai text
   - M4 只准做 openai stream
   - M5 只准做 openai embed
   - 其它 provider 在 M5 之后才能开始
   - 越界 → 停下报告

2. `crates/llmsdk-<name>` 已存在吗？
   - 不存在：用 `cargo new --lib crates/llmsdk-<name>` 创建，但**必须先停下让用户确认**
   - 存在：继续

3. `crates/llmsdk-provider-utils` 已就绪吗？
   - capability ∈ {text, stream, tool, embed}：必须依赖 `llmsdk-provider-utils`（M2 完成）才能开始
   - 否则停下报告

## 红线（与 /port 相同 + 额外）

11. **不准**在 provider crate 里再造 HTTP / SSE / retry 逻辑；只用 `llmsdk-provider-utils`
12. **不准**在 provider crate 暴露公开类型（除了 `OpenAi` 这种构造入口）；其它都用 `pub(crate)`
13. **不准**跳过契约测试就报告完成
14. **不准**用真实 API key 跑测试 —— 用 `wiremock` / 录制的 fixture

## 流程

### Phase 1 — 找上游对照

读 ai-sdk 的对应文件：
- `text` → `packages/<name>/src/<name>-chat-language-model.ts` 的 `doGenerate`
- `stream` → 同上的 `doStream`
- `tool` → 同上的 tool 序列化 / 解析逻辑
- `embed` → `packages/<name>/src/<name>-embedding-model.ts`
- `image` → `packages/<name>/src/<name>-image-model.ts`

用 Read 完整读。提取：
- 请求 URL / 方法
- 请求 body 形状（哪些字段必填、哪些可选）
- 响应形状
- 错误码 → ProviderError 的映射
- provider-specific options 字段（写在 `provider_options["<name>"]` 下）

### Phase 2 — 设计 prompt 映射

我们的 `Prompt = Vec<Message>` → provider native 请求。
**必须**写一张映射表注释在代码顶部：

```rust
//! Prompt mapping (llmsdk -> openai):
//!   Message::System { content }            -> { "role": "system", "content": <text> }
//!   Message::User { content: parts }        -> { "role": "user", "content": [...] }
//!     UserPart::Text                         -> { "type": "text", "text": ... }
//!     UserPart::File { media_type: image/* } -> { "type": "image_url", "image_url": { "url": ... } }
//!   ...
```

任何 ai-sdk 行为我们**不打算复现**的，在这张表里明确标注 "(skipped: <reason>)"。不准默默丢弃。

### Phase 3 — 实现

按这个顺序：

1. **类型**：`<Name>ChatRequest` / `<Name>ChatResponse`（serde struct）→ `cargo check`
2. **prompt 转换函数**：`fn convert_prompt(p: &Prompt) -> Vec<RequestMessage>` → 单元测试覆盖每种 part → `cargo nextest`
3. **响应解析函数**：`fn parse_response(r: <Name>ChatResponse) -> GenerateResult` → 单元测试 → `cargo nextest`
4. **错误映射**：HTTP 状态 → `ProviderError::api_call_builder()...build()`，单元测试覆盖每种状态 → `cargo nextest`
5. **拼装 trait impl**：`impl LanguageModel for <Name>ChatModel`，方法体只是 "拼参数 → call util → parse" → `cargo check`

每步失败停。

### Phase 4 — 契约测试

`crates/llmsdk-<name>/tests/contract_<capability>.rs`：

- 用 `wiremock` 起一个 mock server
- 录制至少 3 个场景：
  - happy path
  - HTTP 429（验证 `is_retryable()`）
  - 响应缺字段（验证 `ProviderError`）
- text 还要验证：finish_reason 映射、usage 正确解析
- stream 还要验证：StreamPart 顺序、Finish frame 必出现一次

跑：
```
cargo nextest run -p llmsdk-<name> --test contract_<capability>
```

### Phase 5 — 门禁

```bash
cargo fmt --check
cargo clippy -p llmsdk-<name> --all-targets -- -D warnings
cargo nextest run -p llmsdk-<name>
cargo test -p llmsdk-<name> --doc
```

全绿才算完。

### Phase 6 — 报告

```
## /provider <name> <capability> 完成

ai-sdk 对照：<path>
契约测试：tests/contract_<capability>.rs（N 个场景）

prompt 映射偏离（如有）：
- <part>: <说明>

错误映射：
- 400 -> ProviderError::invalid_argument
- 401 -> ProviderError::api_call (non-retryable)
- 429 -> ProviderError::api_call (retryable)
- 5xx -> ProviderError::api_call (retryable)

门禁：
- fmt/clippy/nextest/doctest: 全绿

下一步建议：<只列 1 项；下一个 capability 或下一个 provider，让用户决定>
```

不要主动开始下一个 capability。
