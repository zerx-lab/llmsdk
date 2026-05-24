# AGENTS.md

> llmsdk是对标vercel ai-sdk的一个rust的实现，目标是在安全稳定的情况下接入完善更多的AI Api的支持

## 强制规则
- 禁止新增 dependency，需要时先在 PR/对话里说明理由并等确认
- 禁止 `unsafe`，除非显式批准
- 禁止 `unwrap()` / `expect()` 在非测试代码中出现；用 `?` + `thiserror`
- 公开 API 必须有 doc comment + 至少一个 doctest 或 example
- 改动前先跑 `cargo check -p <crate>`（不是整个 workspace）
- 提交前必须通过：`cargo fmt --check && cargo clippy -- -D warnings`
- 验证编译时优先 `cargo check -p <crate> --lib`
- 跑测试时优先 `cargo nextest run -p <crate> <filter>`，不要 `cargo test --workspace`
- 使用cargo管理依赖，禁止直接编辑`Cargo.toml`进行版本管理
- 测试 provider 兼容性时调用 `provider-contract-test` skill

## 代码风格
- 优先复用项目已有的 trait / error 类型，不要平行造轮子
- 单文件超过 400 行考虑拆分；单函数超过 80 行需要说明
- 异步默认 `tokio`，不要混用其它 runtime

## 查文档优先级
1. `cargo path <crate>` 看本地源码（最权威）
2. `cargo doc --open` 或 docs.rs
3. 最后才是 web 搜索

## Rust 编码触发规则
写或改 `.rs` 文件前，先判断本次改动是否涉及以下任一项：
- 新增/修改 public API、trait、error 类型
- 写 unsafe / FFI / 性能关键路径
- 新增 crate 或调整 workspace 结构
- 写文档注释（doc comment）

若**命中任一项**，必须先读 `ms-rust` skills。
若仅是改变量名、调格式、加日志等局部改动，可跳过。

## 移植原则（ai-sdk → Rust）

**事实基础**：`architecture/0001-trait-design.md` 是 ground truth；与该文档冲突的实现必须停下来同步文档。

**上游路径**：ai-sdk 仓库位于 `/home/zero/Desktop/code/github/ai/`，对照 `packages/provider/src/**/v4/*`。

- 不要逐字翻译 TS 类型；先理解 ai-sdk 语义意图，再用 Rust 惯用法重新设计
- TS discriminated union（`{ type: 'x', ... }`）→ Rust `enum` + `#[serde(tag = "type", rename_all = "kebab-case")]`
- TS `& { providerOptions?: ... }` 交叉类型 → Rust 每个 variant 平铺字段，**不要**用 wrapper struct
- TS `Promise<T>` → `async fn`；TS `ReadableStream<T>` → `Pin<Box<dyn Stream<Item = Result<T, E>> + Send>>`
- TS `JSONValue` / `JSONSchema7` → `serde_json::Value`（M1-M5 阶段不引 schemars）
- TS `Uint8Array` → `bytes::Bytes`
- TS `AbortSignal` → 不在 trait 暴露；调用方靠 drop future / stream 取消
- JSON wire 字段名保持与 ai-sdk 一致（`providerOptions` / `toolCallId` 等），Rust 侧用 snake_case + serde rename
- 每个 `.rs` 文件顶部用 `//! Mirrors <ai-sdk relative path>` 注释指出对照文件

## 里程碑约束（强制）

当前进度：M1–M5 全部完成。OpenAI provider 已具备 chat (text + stream) +
embedding 能力，64 个 workspace 测试全绿。

```
M1 ✓ llmsdk-provider 编译通过；trait + 类型 ready
M2 ✓ llmsdk-provider-utils: HTTP/SSE/load_api_key
M3 ✓ llmsdk-openai: do_generate + contract::chat_basic 通过
M4 ✓ llmsdk-openai: do_stream + contract::chat_stream 通过
M5 ✓ llmsdk-openai: EmbeddingModel + contract::embed_basic 通过
```

**下一阶段候选**（待规划）：
- 第二个 reference provider（Anthropic / Gemini）
- ImageModel 实现
- Reasoning / search-preview / annotations 等 M3–M5 期间被推迟的特性
- middleware 层

**跨越里程碑/阶段禁止**。开新阶段前必须停下来对齐。

## Checkpoint 规则

- 每完成 1 个 trait 定义 → 停下来等审核，不要立刻 impl
- 每完成 1 个 provider 的 1 个 capability（text / stream / tool / embed）→ 跑契约测试，不要继续下一个
- 需要修改 `crates/llmsdk-provider` 的 trait → 必须停下来说明影响范围，不准静默改动
- 需要新增依赖 → 必须在对话里列出依赖名 + 用途，等确认后用 `cargo add` 添加
