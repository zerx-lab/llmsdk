# todo.md — 推迟事项与未完成项

> 入口文件：跨里程碑追踪所有 deferred / TODO 项目；任何阶段性停顿都先记到这里。

## 当前里程碑：M12 Anthropic Full API Parity 完成 ✓

M12 把 Anthropic 剩余三条 API 路径全部接入，达成 ai-sdk Anthropic 包 100% feature parity：

- **新增 2 个 trait**：`FilesModel` + `SkillsModel` + 关联类型（纯新增，不破坏）
- **AnthropicFiles** (`POST /v1/files`, beta `files-api-2025-04-14`)：multipart 上传
  + 响应映射到 `provider_metadata.anthropic.{filename,mimeType,sizeBytes,createdAt,downloadable}`
- **AnthropicSkills** (`POST /v1/skills` + `GET /v1/skills/{id}/versions/{v}`,
  beta `skills-2025-10-02`)：multipart files[]+display_title + version metadata
  二次回填 name/description
- **20 个 typed tool factory**：advisor_20260301 / bash×2 / code_execution×3 /
  computer×3 / memory / text_editor×4 / web_fetch×2 / web_search×2 /
  tool_search×2，全部 typed args + snake_case wire
- **AnthropicBuilder 扩展**：auth_token（与 api_key 互斥）/ name / chat /
  language_model / files / skills 工厂；新增 ANTHROPIC_AUTH_TOKEN env var
- **响应元数据深度解析**：usageIterations 三 variant + container + applied_edits
  三 variant 全部 typed 写入 `provider_metadata.anthropic.*`
- 设计文档：`architecture/0005-m12-anthropic-full-design.md`
- 3 套契约测试（17 wiremock 用例）+ 23 个新单元测试
- workspace 健康：375 测试全绿（M11 → M12 +54）；fmt + clippy 通过
- subagent 审核 PASS（与 ai-sdk Anthropic 包 100% feature parity）

### M11（历史）

M11 完整接入 OpenAI 第二条 LanguageModel 端点 `POST /v1/responses`，与现有 Chat
端点并存：
- 22 项 `provider_options.openai.*` 全量透传 + 模型能力校验（reasoning effort /
  serviceTier flex/priority / conversation+previousResponseId 互斥）
- 11 个 provider-defined tools 完整 args / output / tool_choice 路由：
  web_search / web_search_preview / file_search / code_interpreter /
  image_generation / local_shell / shell / apply_patch / mcp / custom /
  tool_search
- 18 种 output item type 非流式解析 + 4 种 annotation → Content::Source
- 20+ SSE event type 流式状态机（含 reasoning summary 三态机 active/can-conclude/
  concluded + apply_patch hasDiff/endEmitted 状态 + image_generation
  partial_image preliminary 标记）
- Prompt → input items 转换（systemMessageMode 3 态 + reasoning 模型自动 developer
  + 11 种 assistant tool-call 路由 + MCP approval_response 透传 + file_id 引用）
- 1 处 trait 改动：`ToolCallPart` 加 `dynamic: Option<bool>`（对齐
  `StreamPart::ToolCall.dynamic`，向后兼容）
- 设计文档：`architecture/0004-m11-responses-design.md`
- 5 套契约测试（23 个 wiremock 用例）+ 64 个 responses 单元测试
- workspace 健康：321 测试全绿（M10.5 → M11 +100）；
  `cargo fmt --check`、`cargo clippy --workspace --all-targets -- -D warnings` 通过
- subagent 审核 PASS（与 ai-sdk 上游 100% 特性对齐）

### M10 / M10.5（历史）

M10 单一阶段对齐了 ai-sdk v4 的 OpenAI / Anthropic provider 全部推迟特性 + 三个
模型表面的 middleware + 内置 5 个 ai-sdk 风格 middleware + schemars 切换。

M10.5 review fix-pack 补齐 ai-sdk Chat API 对齐审核中发现的所有偏差。设计文档
`architecture/0003-m10-design.md`。

## 仍然推迟（M13+）

### Provider 扩展

- **Gemini provider**（M13 候选）：用户已推迟两轮（M12 也优先 Anthropic
  feature parity）。验证 trait 抽象的第三个 provider。
- ~~**Anthropic Files API endpoint**~~：M12 完成。
- ~~**Anthropic Skills API endpoint**~~：M12 完成。
- **fileIdPrefixes 用户可配置化**：M11 暂硬编码常量；M13+ 改为
  `OpenAi::builder().file_id_prefixes(...)` 公开 API。

### Middleware

- ai-sdk 还在更新的内置 middleware（未在第一轮中复刻）：随上游 ai-sdk 后续版本
  再补。
- Middleware 间共享上下文（`MiddlewareContext`）目前是单向透传：上游写、下游读。
  双向回填 / tracing span 自动衔接是下一步。
- CacheMiddleware：分布式（Redis）reference impl；按 prefix 分桶；条件失效。
- RetryMiddleware：指数退避之外的策略（fixed / fibonacci）。

### OpenAI

- 流式 `usage` 字段的 prediction tokens 在 `Finish` 帧里二次确认（M10 仅在非流式
  确认 `accepted/rejected_prediction_tokens` 入 metadata）。
- `flex` / `priority` service tier 上的请求路径 SLA 差异 — 当前只是字符串校验。
- Provider-defined tools 的结果块在 Chat API（`web_search_call` 中间结构）尚未
  形成 `Content::Source` —— 仅在 Responses API 才完整呈现。

### Anthropic

- `compact_20260112` / `clear_thinking_20251015` 之类的 beta header 自动添加
  通过字符串搜 wire JSON 来粗略推断；改成结构化解析更稳妥。
- `tool_use` 元数据中的 `programmatic-tool-call` 完整结构（caller.type +
  caller.tool_id）现在以原始 JSON 形式透传到 `provider_options.anthropic.caller`，
  未拆字段。
- `thinking budget + max_tokens` 校验当前用 `saturating_add`；模型上下文上限
  应改为 clamp + warning（M10 已经识别但未做模型上限表）。
- 服务器工具结果块（`web_search_tool_result` 等）目前以
  `ToolResult { output: ToolResultOutput::Json(raw_value) }` 形式呈现；后续可
  拆出结构化字段（`urls[]`、`citations[]`、`code_output`）。

### 通用

- `JSONSchema` 切到 `schemars::Schema` 后，部分上游 sanitize 行为（如 Anthropic
  的 `additionalProperties:false` 强制）还在工具用户侧处理；ai-sdk anthropic
  crate 的 `sanitize-json-schema.ts` 完整移植到 llmsdk-anthropic 内部尚未做。
- Embedding `JSONSchema` derive 自动生成端到端示例。
- contract test：M11+ 新增 image-edit / image-variation / anthropic-server-tool
  契约用例。

## 已完成（历史）

### M10（最新）

依赖增量：`schemars = "1.0"`（features: derive + std）。

Trait 改动：
- `JsonSchema = schemars::Schema`（之前 `serde_json::Value`）
- `ImageOptions` 新增 `files: Option<Vec<FilePart>>` + `mask: Option<FilePart>`
- `ImageResult` 新增 `usage: Option<ImageUsage>` + `ImageUsage` / `ImageUsageInputDetails`

Middleware 表面统一：
- `EmbeddingModelMiddleware` + `wrap_embedding_model`
- `ImageModelMiddleware` + `wrap_image_model`
- `wrap_provider(inner, ProviderMiddlewareSet { language, embedding, image })`
- 5 个内置 ai-sdk 风格 middleware：
  - `DefaultSettingsMiddleware` / `DefaultEmbeddingSettingsMiddleware`
  - `ExtractReasoningMiddleware`（tag-based reasoning 切分）
  - `SimulateStreamingMiddleware`（generate → stream 块级转换）
  - `ExtractJsonMiddleware`（markdown fence 剥离，缓冲到块结束）
  - `AddToolInputExamplesMiddleware`（examples 拼到 description）
- `CacheMiddleware` 加 TTL + LRU：`MemoryCacheStore::builder().max_entries()
  .max_age()`，自实现 LRU（counter + O(n) eviction），零新依赖
- `RetryMiddleware` 加 `jitter_ratio`：自实现 SplitMix64 + SystemTime 纳秒
  seed，零新依赖
- `LoggingMiddleware` 加 `with_stream_parts(true)` 启用 per-frame
  `Logger::log_stream_part` 事件
- `MiddlewareContext`（request_id / trace_id / parent_span_id / operation）
  通过 `provider_options["llmsdk"]` 透传，不动 trait

OpenAI Chat 补齐：
- 透传 `prediction` / `store` / `metadata` / `service_tier`（含 flex/priority
  校验）/ `safety_identifier` / `prompt_cache_key` / `parallel_tool_calls` /
  `user` / `logit_bias` / `text.verbosity`
- `top_logprobs` 独立字段 + 兼容旧 `logprobs` 复合写法
- `strict_json_schema` provider option（替换硬编码 `true`）
- reasoning 模型自动剥离 `logit_bias` + warning
- 流式 error chunk 提取 `type` / `code` 进 `StreamPart::Error.error` JSON
- `accepted_prediction_tokens` / `rejected_prediction_tokens` 入
  `provider_metadata.openai.prediction`
- Chat-API provider-defined tool `web_search_preview` 路由（其它 OpenAI
  provider-defined tools 留 todo，需要 Responses endpoint）

OpenAI Image 补齐：
- `POST /v1/images/edits`：multipart/form-data，接受多 `files` + `mask` +
  `inputFidelity` provider option
- `POST /v1/images/variations`：multipart/form-data，单 file 必填，prompt 剥离
- `EndpointMode::Generate` / `Edit` / `Variation` 自动路由
- `gpt-image-1` `usage` 字段映射到 `ImageResult.usage`
- `ResponseInfo.id` 用 `openai-img-{created}` 合成
- 新增 `llmsdk-provider-utils::http::post_raw` + `RawRequest`
- 新增 `llmsdk-provider-utils::multipart` 模块（hand-rolled RFC 7578，零新依赖）

Anthropic Messages 补齐：
- 8 种服务器工具（`web_search` / `web_fetch` / `code_execution` / `mcp` /
  `bash` / `text_editor` / `tool_search` / `advisor`）通过
  `Tool::Provider { id: "anthropic.X", name, args }` 路由 + 自动 beta header
- `WireTool` 改为 `untagged` enum（Function / Server）
- 响应侧 9 种 `*_tool_use` / `*_tool_result` block 解析为
  `Content::ToolCall { provider_executed: true }` / `Content::ToolResult`
- `provider_options.anthropic.cache_control` → 所有块（text / image / document /
  tool_result）的 wire `cache_control` 字段
- `provider_options.anthropic.citations` + `title` + `context` →
  document block 字段
- `provider_options.anthropic.context_management` / `container` 透传到 wire 顶层
- 非图片文件：`application/pdf` / `text/plain` 走 `document` block，audio
  发 warning 剥离
- `compaction` 响应 block → `Content::Custom { kind: "anthropic.compaction" }`
- `tool_use` block 上的 `caller` / `dynamic` 元数据 → `ToolCallPart.
  provider_options.anthropic`
- text block 上的 `citations[]` → `TextPart.provider_options.anthropic.citations`
- `include_raw_chunks` 选项：每个 SSE event → `StreamPart::Raw` prepend
- `thinking adaptive` 类型 + thinking budget 默认 1024
  (`DEFAULT_THINKING_BUDGET`)

Workspace 健康：
- 191 个测试全绿（166 → 191：新增 25 个）
- `cargo fmt --check` 通过
- `cargo clippy --workspace --all-targets -- -D warnings` 通过
- 设计文档：`architecture/0003-m10-design.md`

### M9
- `LanguageModelMiddleware` trait（6 个方法全 no-op）+
  `wrap_language_model(model, [m1, m2])`
- 内置 `RetryMiddleware` / `LoggingMiddleware` / `CacheMiddleware`
- 设计文档：`architecture/0002-middleware-design.md`

### M8
- OpenAI Image generation (DALL-E 3 + gpt-image-1\*)
- `ImageResult.warnings` 字段

### M7
- OpenAI reasoning models（o1/o3/o4-mini/gpt-5\*）
- OpenAI logprobs + url_citation annotations
- Anthropic thinking blocks（visible + redacted）

### M1–M6
见 CLAUDE.md 里程碑约束部分。
