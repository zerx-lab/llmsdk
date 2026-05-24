# 0003 — M10 全量对齐设计（ai-sdk feature parity）

> Status: completed (M10 landed; 191 workspace tests green; fmt + clippy clean)
> Upstream reference: `vercel/ai` @ `packages/{provider,ai,openai,anthropic}/src/**/v4/*`
> Prereqs: `0001-trait-design.md`、`0002-middleware-design.md`

## Goal

把 `todo.md` 里"仍然推迟（M10+）"段所有项目（除 Gemini）一次性消化，使 llmsdk
达到与 ai-sdk v4 的 **OpenAI / Anthropic provider 完整对齐**，并补齐 Embedding /
Image middleware 表面。

强制规则要求"启动新阶段前必须列出全部范围"——本文档即范围 ground truth；中途
发现复杂度超预期必须停下来同步文档（扩范围 / 改阶段定义 / 拒绝某项），不静默推迟。

## 范围（全部纳入）

### A. 基础设施 / Trait

1. **schemars 1.x 引入**：替换 `JsonSchema = serde_json::Value`，保留向后
   兼容（`From<&schemars::Schema> for JsonValue`）。
2. **Trait 改动**：
   - `ImageResult` 加 `usage: Option<ImageUsage>`
   - `ImageOptions` 加 `files: Option<Vec<FilePart>>` + `mask: Option<FilePart>`
   - 同步更新 `0001-trait-design.md`
3. **新增依赖**（cargo add）：
   - `schemars = "1.0"`（features: `derive`, `std`）—— Q1 调研确认 1.x 稳定且兼容 OpenAI/Anthropic schema 要求。

### B. Middleware 表面统一

4. `EmbeddingModelMiddleware` trait + `wrap_embedding_model`
5. `ImageModelMiddleware` trait + `wrap_image_model`
6. `wrap_provider`：顶层 Provider 工厂批量装饰（统一中间件链；不支持按 model id 分流，与 ai-sdk 一致）
7. 内置 5 个 ai-sdk 风格 middleware：
   - `default_settings`（语言）+ `default_embedding_settings`
   - `extract_reasoning`（按 tag 分割 text → reasoning）
   - `simulate_streaming`（generate → stream 转换）
   - `extract_json`（剥 markdown fence → JSON）
   - `add_tool_input_examples`（把 examples 拼到 description）
8. `CacheMiddleware`：TTL + LRU 淘汰（max_entries / max_age）
9. `RetryMiddleware`：jitter_ratio（0.0–1.0），随机源用 `SystemTime::now()` 纳秒 + SplitMix64（自实现，零新依赖）
10. `LoggingMiddleware`：加 `log_stream_part` 事件钩子，默认 off
11. Middleware 共享上下文：复用 `CallOptions.provider_options["llmsdk"]` 透传 `request_id` / `parent_span`，不动 trait

### C. OpenAI Chat 补齐

12. **provider_options.openai 透传**：
    - `prediction` / `store` / `metadata` / `service_tier`（含 flex/priority 校验）
    - `safety_identifier` / `prompt_cache_key`
    - `text.verbosity` / `parallel_tool_calls` / `user`
    - `logit_bias`（reasoning 模型自动剥离 + warning）
    - `top_logprobs` 独立字段（与 logprobs 解耦）
    - `strict_json_schema`（替换硬编码 `true`）
13. **响应类**：
    - `usage.completion_tokens_details.accepted_prediction_tokens` / `rejected_prediction_tokens` → `provider_metadata.openai.usage.*`
    - 流式 `error` chunk 提取 `code` / `type` → `ProviderError.kind` 的可见字段
14. **provider-defined tools**：用现有 `Tool::Provider { id, name, args, provider_options }`；OpenAI crate 内识别 `id == "openai.web_search_preview" | "openai.web_search" | "openai.file_search" | "openai.code_interpreter" | ...`，发往 Chat（仅 web_search_preview）或 Responses API
    - **本轮范围**：仅实现 Chat API 支持的 `web_search_preview`。其它 10 个工具 wire 形式需要 Responses API endpoint（我们尚未接入），文档里登记 + warning 提示用户切换至 Responses（**记入 todo.md，本阶段不实现 Responses API 端点**）。

### D. OpenAI Image 补齐

15. `POST /v1/images/edits`：用 multipart/form-data；接受 `ImageOptions.files[0]`（必填）+ `mask`（可选）+ prompt
16. `POST /v1/images/variations`：multipart；接受 `ImageOptions.files[0]`（必填），无 prompt（剥离 + warning）
17. `inputFidelity` provider option（仅 edits）
18. `ImageResult.usage` 字段：gpt-image-1 usage → `ImageUsage { input_tokens, output_tokens }`
19. `ResponseInfo.id` 用 `created` 时间戳合成（`"openai-img-{unix_timestamp}"`）

### E. Anthropic Messages 补齐

20. **服务器工具**（8 种）：用 `Tool::Provider { id: "anthropic.web_search" | ... }`
    - 请求 wire：`type` 字段映射，`args` 平铺到 wire 顶层；自动加对应 beta header
    - 响应 wire：`server_tool_use` block → `Content::ToolCall { provider_executed: true }`；
      `*_tool_result` blocks → `Content::ToolResult`
    - 8 种内置：`web_search` / `web_fetch` / `code_execution` / `mcp` / `bash` / `text_editor` / `tool_search` / `advisor`
21. **citations**：
    - 请求侧：`FilePart.provider_options.anthropic.citations = { enabled: true }` → wire `citations:{enabled:true}`
    - 响应侧：text block 上的 `citations[]` → `Content::Source::Url / Document`，关联到上一段 text 的 provider_metadata
22. **cache_control**：通过 `provider_options.anthropic.cache_control = { type: "ephemeral", ttl?: "5m"|"1h" }`，落到对应块 wire 上
23. **context_management**：`provider_options.anthropic.context_management` 透传（`clear_tool_uses_20250919` / `clear_thinking_20251015` / `compact_20260112`）+ 触发对应 beta header
24. **containers**：`provider_options.anthropic.container = { id, skills: [...] }` 透传 + 触发 skills beta
25. **非图片文件**：`convert_to_anthropic_prompt` 支持
    - PDF：`document` block + `source.media_type: "application/pdf"`
    - text/plain：`document` block + `source.media_type: "text/plain"`
    - audio：暂不支持（Anthropic API 不支持音频输入，发 warning 剥离）
    - 通用 document：按 media_type 路由
26. **compaction 块**：响应 text block 的 `type: "compaction"` 子类 → `Content::Text` + `provider_metadata.anthropic.kind = "compaction"`；流 `compaction_delta` 走 text-delta + provider_metadata
27. **tool_use 元数据**：`caller` / `dynamic` / `programmatic-tool-call` 透传到 `ToolCallPart.provider_options.anthropic`
28. **include_raw_chunks**：尊重 `CallOptions.include_raw_chunks`，原始 SSE event JSON → `StreamPart::Raw`
29. **thinking adaptive**：`thinking = { type: "adaptive" }` 与 enabled/disabled 同源透传
30. **thinking budget 默认 + 校验**：缺 budget 时默认 1024；`max_tokens + budget` 检查超过模型上限时 clamp + warning

### F. 测试 + 文档

31. provider-contract-test：M10 不新增 contract case；现有 chat_basic/chat_stream/embed_basic 全绿即可
32. 单元 / 集成测试：每个特性 ≥1 个测试
33. 更新文档：`0001` / `0002` / `0003` / `todo.md` / `CLAUDE.md`

## 不在 M10 范围

- **Gemini provider**：用户明确推迟（记入 todo.md）
- **OpenAI Responses API endpoint**：除 `web_search_preview` 外的 9 个 OpenAI provider-defined tools 需要 Responses endpoint；本轮不实现，登记到 todo.md
- **Anthropic Files API endpoint**：当前用 inline base64 / URL；Files API 上传记入 todo.md
- **schemars 0.8 兼容**：直接用 1.x，0.8 不维护
- **Middleware tracing 集成**：只提供透传字段，不绑 tracing crate

## TS → Rust 映射决策（增量）

| TS | Rust | 理由 |
|---|---|---|
| `JsonSchema7` JS 对象 | `schemars::Schema` | 1.x API 稳定；Anthropic 自带 sanitize；OpenAI strict 由 `strict_json_schema` 选项控制 |
| `wrapProvider({provider, languageMiddleware, ...})` | `wrap_provider(provider, ProviderMiddlewareSet { language, embedding, image })` 返回 `Box<dyn Provider>` | Rust 静态分发；不支持按 model id 分流（与 ai-sdk 一致） |
| `extractReasoning({tagName, startWithReasoning})` | `ExtractReasoningMiddleware::new(tag_name, start_with_reasoning)` | 状态机 stream + generate 一次性提取 |
| `simulateStreaming` | `SimulateStreamingMiddleware` —— `wrap_stream` 内调 `next.do_generate`，按 content variant 顺序发射 `*Start/*Delta/*End` | 块粒度（非 token） |
| `extractJson` | `ExtractJsonMiddleware`：去 markdown fence + trim；流式三态机（prefix/streaming/buffering） | 用现有 `StreamPart::TextDelta` |
| `addToolInputExamples` | `AddToolInputExamplesMiddleware`：`transform_params` 内遍历 `Tool::Function`，把 `input_examples` 拼到 `description` | 不修改 tool 结构 |
| `defaultSettings` | `DefaultSettingsMiddleware`：深度合并 `CallOptions`（caller 覆盖 default） | mergeObjects 逻辑 |

## Trait 改动详情（A.2）

```rust
// image_model.rs
pub struct ImageOptions {
    // ...existing
    pub files: Option<Vec<FilePart>>,    // 新增（edits/variations 用）
    pub mask: Option<FilePart>,          // 新增（edits 用）
}

pub struct ImageResult {
    // ...existing
    pub usage: Option<ImageUsage>,       // 新增（gpt-image-1）
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub input_tokens_details: Option<ImageUsageInputDetails>,
}

pub struct ImageUsageInputDetails {
    pub text_tokens: Option<u64>,
    pub image_tokens: Option<u64>,
}
```

下游影响：OpenAI Image impl 同步更新；Anthropic 暂无 Image impl，无影响。

## Middleware 表面统一（B.4–B.6）

```rust
// 复用 M9 LanguageModelMiddleware 形态
#[async_trait]
pub trait EmbeddingModelMiddleware: Send + Sync + std::fmt::Debug {
    fn override_provider(&self, _: &dyn EmbeddingModel) -> Option<String> { None }
    fn override_model_id(&self, _: &dyn EmbeddingModel) -> Option<String> { None }
    async fn override_max_embeddings_per_call(&self, _: &dyn EmbeddingModel) -> Option<u32> { None }
    async fn override_supports_parallel_calls(&self, _: &dyn EmbeddingModel) -> Option<bool> { None }
    async fn transform_params(&self, params: EmbedOptions, _: &dyn EmbeddingModel) -> Result<EmbedOptions> { Ok(params) }
    async fn wrap_embed(&self, next: &dyn EmbeddingModel, params: EmbedOptions) -> Result<EmbedResult> {
        next.do_embed(params).await
    }
}

pub fn wrap_embedding_model(
    model: Arc<dyn EmbeddingModel>,
    middleware: impl IntoIterator<Item = Arc<dyn EmbeddingModelMiddleware>>,
) -> Arc<dyn EmbeddingModel> { /* fold reverse */ }
```

Image 同形态；`override_max_images_per_call` + `transform_options` + `wrap_generate`。

```rust
pub struct ProviderMiddlewareSet {
    pub language: Vec<Arc<dyn LanguageModelMiddleware>>,
    pub embedding: Vec<Arc<dyn EmbeddingModelMiddleware>>,
    pub image: Vec<Arc<dyn ImageModelMiddleware>>,
}

pub fn wrap_provider(
    inner: Arc<dyn Provider>,
    set: ProviderMiddlewareSet,
) -> Arc<dyn Provider> { /* WrappedProvider */ }
```

## OpenAI provider-defined tools 实现（C.14）

```rust
// llmsdk-openai 内部识别
match tool {
    Tool::Provider(p) if p.id == "openai.web_search_preview" => {
        // 平铺到 wire 顶层 type + args
        wire_tools.push(json!({
            "type": "web_search_preview",
            ..p.args.unwrap_or_default()
        }));
    }
    Tool::Provider(p) if p.id.starts_with("openai.") => {
        warnings.push(Warning::UnsupportedTool {
            tool: p.id.clone(),
            details: Some("requires Responses API endpoint (todo)".into()),
        });
    }
    // ...其它 provider 的 Tool::Provider 也是 unsupported
    Tool::Function(f) => { /* 现有逻辑 */ }
}
```

## Anthropic 服务器工具实现（E.20）

类似 OpenAI；按 id 路由到对应 wire `type` + 自动附加 beta header：

| id | wire type | beta header |
|---|---|---|
| `anthropic.web_search` | `web_search_20250305` | `web-search-2025-03-05` |
| `anthropic.web_fetch` | `web_fetch_20250910` | `web-fetch-2025-09-10` |
| `anthropic.code_execution` | `code_execution_20250825` | `code-execution-2025-08-25` |
| `anthropic.mcp` | `mcp_20250508` | `mcp-2025-05-08` |
| `anthropic.bash` | `bash_20250124` | （inherit code_execution） |
| `anthropic.text_editor` | `text_editor_20250728` | （inherit） |
| `anthropic.tool_search` | `tool_search_regex_20251020` | `tool-search-2025-10-20` |
| `anthropic.advisor` | `advisor_20251020` | `advisor-2025-10-20` |

响应 wire：
- `content[].type == "server_tool_use"` → `Content::ToolCall { provider_executed: Some(true), provider_options.anthropic.server_tool: true }`
- `content[].type` 以 `_tool_result` 结尾 → `Content::ToolResult`，按工具名归一 `tool_name`

## 实施顺序（强制）

按依赖关系拓扑排序，**不允许跳序**：

1. **A.1 + A.2 + A.3**：基础设施 + trait 改动（一起做，一次性破坏一次下游）
2. **B**：Middleware 表面 + 内置 ai-sdk middleware（独立于 provider 实现）
3. **C + D**：OpenAI Chat + Image 补齐（trait 改完后做 Image）
4. **E**：Anthropic 补齐
5. **F**：测试 + 文档收尾

每一组完成后：
- `cargo check -p <crate>`
- `cargo nextest run -p <crate>`
- 启动 `Explore` subagent 审 1 次（按 CLAUDE.md Checkpoint 协议）
- PASS → 进下一组；FAIL → 修复后重审

## 风险登记

- **schemars 1.x 与 OpenAI strict**：1.x 默认 draft-2020-12，OpenAI 接受但 strict mode 要求 `additionalProperties: false`。我们用 `strict_json_schema` provider option（默认 true）+ 让用户在自己的 schema 上加约束；不在框架层 sanitize。
- **Anthropic schema sanitize**：上游 ai-sdk anthropic crate 自带 `sanitize-json-schema.ts`（强制 `additionalProperties:false`、`oneOf→anyOf` 等）；我们在 `llmsdk-anthropic` 内部移植同等 sanitize（不引新依赖）。
- **CacheMiddleware LRU**：自实现 `LinkedHashMap` 风格（双向链表 + HashMap），避免引入 `linked-hash-map` / `lru` crate。规模 ~100 行。
- **RetryMiddleware jitter 随机源**：`SystemTime::now().subsec_nanos()` seed → SplitMix64 输出（自实现，~20 行）；不引 `rand`。

## 改动本文档需走的流程

1. 改这份文档的 PR 必须先单独提
2. 改完 → 同步 `CLAUDE.md` 里程碑段
3. 通过后才改代码
