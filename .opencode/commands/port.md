---
description: 把一个 ai-sdk TypeScript 文件移植到 Rust。用法 /port <crate>::<rust-path> <- <ai-sdk relative path>。例：/port llmsdk-provider-utils::http::post_json <- packages/provider-utils/src/post-to-api.ts
---

你在运行 **/port 工作流**，把 ai-sdk 上游的一个 TS 文件移植成 Rust。

## 输入

- ai-sdk 仓库根：`/home/zero/Desktop/code/github/ai/`
- 移植目标：`$ARGUMENTS`，格式 `<crate>::<rust-path> <- <ai-sdk relative path>`

如果 `$ARGUMENTS` 不符合该格式，停下来问用户确切的源文件与目标位置，不要猜。

## 红线（必须遵守，违反一条立即停）

1. 不准 `unwrap()` / `expect()` 在非测试代码
2. 不准 `unsafe`
3. 不准编辑 `Cargo.toml` 改版本；用 `cargo add` / `cargo remove`
4. 不准新增 dependency 而不在对话里先列出原因
5. 不准把 TS class 1:1 翻译成 `struct + impl`
6. 不准在 trait 里暴露 `AbortSignal` / `Promise<T>`（用 drop-to-cancel + `async fn`）
7. 不准跨越当前里程碑边界（看 AGENTS.md "里程碑约束"）
8. 不准修改 `crates/llmsdk-provider` 的公开 trait — 如必须，停下来说明
9. 不准跑 `cargo test --workspace`；只跑 `cargo nextest run -p <crate>`
10. 不准跑整个 workspace 的 `cargo check`；只 `cargo check -p <crate> --lib`

## 流程

### Phase 1 — 读上游

用 Read 工具完整读 `<ai-sdk relative path>`（不要只读片段）。
- 列出文件导出的所有类型 / 函数 / 接口
- 标出它的依赖（import 哪些其它 ai-sdk 文件）
- 如果依赖文件**还没有 Rust 对应物**，停下来报告，让用户决定先移植依赖还是用占位

### Phase 2 — 对照事实基础

读 `architecture/0001-trait-design.md`：
- 检查映射表里有没有相关条目
- 如果上游文件涉及 trait / 错误类型，必须与 0001 文档一致；冲突就停下来同步文档，不要静默修改代码

读 `crates/<crate>/src/` 现有结构，找复用点：
- 已有的 trait / error variant / 类型能复用就复用，不要平行造轮子
- 用 `semble search` 而不是 grep

### Phase 3 — 翻译规则（强制）

| TS 写法 | Rust 写法 |
|---|---|
| `type Foo = { type: 'a' \| 'b'; ... }` | `enum Foo { A {...}, B {...} }` + `#[serde(tag = "type", rename_all = "kebab-case")]` |
| `type Foo = A & { providerOptions?: X }` | enum 每个 variant 平铺 `provider_options` 字段，不用 wrapper |
| `Promise<T>` | `async fn ... -> Result<T, ProviderError>` |
| `ReadableStream<T>` | `Pin<Box<dyn Stream<Item = Result<T, ProviderError>> + Send>>` |
| `JSONValue` | `serde_json::Value`（用 `crate::json::JsonValue` alias） |
| `JSONSchema7` | `crate::json::JsonSchema`（目前是 `JsonValue` 别名） |
| `Uint8Array` | `bytes::Bytes` |
| TS class with `instanceof` 检查 | struct + 私有 `ErrorKind` enum + `is_*()` helpers |
| TS `Record<string, T>` | `HashMap<String, T>` |
| TS `Record<string, X> & PromiseLike<...>` | 只取 plain object 那侧，不暴露 promise |

JSON wire 字段名**必须与 ai-sdk 完全一致**（`providerOptions` / `toolCallId` / `cacheRead`）。Rust 字段名 snake_case，用 `#[serde(rename = "...")]`。

每个新 `.rs` 文件顶部加：
```rust
//! <一句话说明>.
//!
//! Mirrors `<ai-sdk relative path>`.
```

### Phase 4 — 写代码

按这个顺序，**每一步完成后停下确认编译**：

1. 写类型定义（struct / enum）→ `cargo check -p <crate> --lib`
2. 写 trait → `cargo check -p <crate> --lib`
3. 写 impl → `cargo check -p <crate> --lib`
4. 加 doctest（每个 public item 至少 1 个，否则停下问）→ `cargo test -p <crate> --doc`
5. 加单元测试（如果有逻辑分支）→ `cargo nextest run -p <crate>`

任何一步失败 → 停下报告错误，不要堆 fix。

### Phase 5 — 门禁

按顺序，全绿才算完成：

```bash
cargo fmt --check
cargo clippy -p <crate> --all-targets -- -D warnings
cargo nextest run -p <crate>
cargo test -p <crate> --doc
```

如果 clippy 有 warning，先尝试修；如必须 `#[expect]` / `#[allow]`，必须带 `reason = "..."`。

### Phase 6 — 报告

用结构化输出：

```
## /port <target> 完成

源：ai-sdk: <path>
目标：<crate>::<rust-path>

文件改动：
- <path>: <action>

新增依赖：（如有）
- <name>: <用途>

设计偏离（如有，必须列）：
- <字段/方法>: TS 是 X，Rust 是 Y，因为 <理由>

门禁：
- fmt: ok
- clippy: ok (0 warning)
- nextest: N passed
- doctest: N passed

下一步建议：<只列 1 项；不要主动开始>
```

不要主动开始下一个 port —— 等用户确认。
