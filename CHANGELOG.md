# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-05-27

首个公开 release —— Vercel `ai-sdk` 的 Rust 实现，目标是用 Rust 惯用法
完整复刻 `ai-sdk` v4 的 provider/middleware/model 抽象与一线大厂全部端点。

### Workspace

- 13 个 crate 的 cargo workspace（`resolver = "3"`, `edition = "2024"`,
  `rust-version = "1.95"`, `license = "Apache-2.0"`）
- 推荐入口 [`llmsdk`](crates/llmsdk) umbrella crate：把
  `llmsdk-provider` 在根部 re-export，每个具体 provider 走独立
  cargo feature；个别 crate 仍可单独依赖作为版本/发布单元
- 工程约束：`#![forbid(unsafe_code)]`，禁止 `unwrap()` / `expect()`
  出现在非测试代码；workspace 启用 `clippy::pedantic` + 选定
  restriction lints；最小依赖（jitter / LRU / multipart / base64 全部
  in-tree，仅 `schemars` / `aws-sigv4` / `aws-smithy-eventstream` /
  `gcp_auth` 由协议强制引入）

### Core traits (`llmsdk-provider`)

提供 7 类模型表面 + 2 类上传模型表面 + 1 个 provider 工厂 trait + 5 层
middleware，与 `@ai-sdk/provider` v4 一一对应：

- **模型表面**：`LanguageModel` · `EmbeddingModel` · `ImageModel` ·
  `VideoModel` · `RerankingModel` · `SpeechModel` · `TranscriptionModel`
- **上传表面**：`FilesModel` · `SkillsModel`
- **聚合**：`Provider`
- **错误**：统一的 `ProviderError` + `thiserror` 派生 + 结构化
  `provider_metadata` 透传
- **流式**：返回 `Pin<Box<dyn Stream<Item = Result<StreamPart, _>> + Send>>`，
  靠 drop future / stream 实现取消（不在 trait 暴露 `AbortSignal`）
- **JSON Schema**：`JsonSchema = schemars::Schema`
- **wire 兼容**：JSON 字段名保持与 `ai-sdk` 一致（`providerOptions` /
  `toolCallId` …），Rust 侧用 snake_case + serde rename
- **公开 API**：每条 public 都带 doc comment + 至少 1 个 doctest 或 example

### Middleware (`llmsdk-provider`)

- `RetryMiddleware` —— 指数退避 + jitter，仅重试 `is_retryable` 错误，
  builder：`max_attempts` / `initial_backoff` / `multiplier` /
  `max_backoff` / `jitter_ratio`
- `LoggingMiddleware` —— start / end / error 三类事件 + 可选每帧 stream
  事件；自有 `Logger` trait + `StderrLogger` 示例，不绑 `tracing`
- `CacheMiddleware` —— TTL + LRU；流式用 `CapturingStream` 边走边收，
  错误不写入；命中标 `provider_metadata.llmsdk.cache = "hit"`
- `DefaultSettingsMiddleware` / `DefaultEmbeddingSettingsMiddleware` ——
  注入默认 call options（深度合并 `provider_options`）
- `ExtractReasoningMiddleware` —— 标签驱动的 reasoning 提取（增量状态机）
- `SimulateStreamingMiddleware` —— 把 `do_generate` 结果转成 stream
- `ExtractJsonMiddleware` —— 从 markdown 围栏中剥离 JSON（3 态机 +
  自定义 transform）
- `AddToolInputExamplesMiddleware` —— 在 tool description 末尾追加示例
- 组合器：`wrap_language_model` / `wrap_embedding_model` /
  `wrap_image_model` / `wrap_video_model` / `wrap_reranking_model`
  （列表头最外层）+ `wrap_provider`（`ProviderMiddlewareSet`，整 provider 包装）

### `llmsdk-provider-utils`

- HTTP client + SSE 解析 + multipart/form-data（手写 RFC 7578）
- `load_api_key` 环境变量加载
- 可选 features：
  - `aws-sigv4` —— `SigV4Fetch` / `AwsCredentials` / `AwsCredentialsProvider`
    / `sign_request` / `sign_post`
  - `aws-event-stream` —— `EventStreamMessage` / `EventStreamValue` /
    `decode_event_stream`

### Providers

10 个一线大厂 provider crate，全部 1:1 复刻 `ai-sdk` 上游：

#### OpenAI (`llmsdk-openai`)

- **Chat Completions** (`POST /v1/chat/completions`)：do_generate + do_stream，
  function/JSON-mode/strict tools；reasoning models（o1 / o3 / o4-mini /
  gpt-5\*）`reasoning_effort` 透传、不支持参数剥离、`max_completion_tokens`
  映射、`system→developer` 角色；search-preview 模型 `temperature` 剥离；
  `logprobs` 透传 + `provider_metadata` 收集；`url_citation` annotations
  → `Content::Source` / `StreamPart::Source`
- **Completion** (`POST /v1/completions`)：旧版 text completion 端点
  （gpt-3.5-turbo-instruct 等）
- **Responses API** (`POST /v1/responses`)：`OpenAi::responses(model_id)` 工厂；
  22 项 `provider_options.openai.*` 透传 + 模型能力校验；11 个 provider-defined
  tools（web_search / web_search_preview / file_search / code_interpreter /
  image_generation / local_shell / shell / apply_patch / mcp / custom /
  tool_search）完整 args/output/tool_choice 路由；18 种 output item 非流式
  解析；20+ SSE event type 流式状态机（reasoning summary 三态机 +
  apply_patch hasDiff/endEmitted + image_generation partial_image 标记）；
  `store=false` 无 encrypted reasoning 自动过滤
- **Embedding** (`POST /v1/embeddings`)：`text-embedding-3-*`
- **Image** (`POST /v1/images/generations` + `/edits` + `/variations`)：
  DALL-E 3 / gpt-image-1\*；multipart edits + variations；`quality` /
  `style` / `background` / `outputFormat` / `outputCompression` /
  `moderation` / `user` 透传；按 model id 自动判断
  `max_images_per_call` + 是否发 `response_format`；usage 字段（含
  `image_tokens` / `text_tokens`）入 `ImageResult.usage`；
  `revised_prompt` / `size` / `created` 入 `provider_metadata`
- **Files** (`POST /v1/files`)：`OpenAiFiles`
- **Skills** (`POST /v1/skills`)：`OpenAiSkills`
- **Speech** (`POST /v1/audio/speech`)：tts-1 / tts-1-hd / gpt-4o-mini-tts；
  `voice` / `responseFormat` / `speed` / `instructions` 透传
- **Transcription** (`POST /v1/audio/transcriptions`)：whisper-1 /
  gpt-4o-transcribe / gpt-4o-mini-transcribe / gpt-4o-transcribe-diarize；
  multipart 上传；`verbose_json` / `json` / `text` / `srt` / `vtt` 全格式
  解析 → `TranscriptionResult { text, segments, language, duration_seconds }`
- **`internal` 模块**：`Inner` / `UrlStrategy` 暴露供 `llmsdk-azure` 复用

#### Anthropic (`llmsdk-anthropic`)

- **Messages API**：do_generate + do_stream；extended thinking blocks
  （visible + redacted；SSE `thinking_delta` / `signature_delta`；
  请求 `thinking` 字段 + 采样参数剥离 + `max_tokens` 自动加 budget；
  默认 budget 1024）
- **20 个 typed tool factory**（带版本号 ID，对齐 `ai-sdk` 上游）：
  `code_execution_{20250522,20250825,20260120}` /
  `computer_{20241022,20250124,20251124}` /
  `text_editor_{20241022,20250124,20250429,20250728}` /
  `bash_{20241022,20250124}` / `memory_20250818` /
  `web_fetch_{20250910,20260209}` /
  `web_search_{20250305,20260209}` /
  `tool_search_{regex,bm25}_20251119` / `advisor_20260301`；
  每个 ID 带正确的 wire `type` + 强制 `name` + 对应 beta header tokens
- **Files API** (`POST /v1/files`)：`AnthropicFiles`，multipart + beta
  `files-api-2025-04-14`
- **Skills API** (`POST /v1/skills` + `GET /v1/skills/{id}/versions/{v}`)：
  `AnthropicSkills`，beta `skills-2025-10-02`
- **provider options**（11 项）：`sendReasoning` / `structuredOutputMode` /
  `disableParallelToolUse` / `cacheControl` / `metadata.userId` /
  `mcpServers` / `toolStreaming` / `effort` / `taskBudget` / `speed` /
  `inferenceGeo` / `anthropicBeta`；`cache_control` 5 种位置；
  `citations` + title + context 进 document block；非图片文件
  （PDF/text/plain）→ document，audio 剥离；`context_management` /
  `container` 透传 + beta header；compaction 响应块；text 上的
  `citations[]` 透传；`include_raw_chunks` 选项
- **响应元数据深度解析**：iterations / container / context_management
  applied_edits 全部映射到 `provider_metadata.anthropic.*`
- **builder 扩展**：`auth_token`（与 api_key 互斥）/ `name`（自定义
  provider 名 + 自动派生 `.files` / `.skills` 后缀）/ `chat(id)` /
  `language_model(id)`（messages 的别名）/ `files()` / `skills()`；
  新增 `ANTHROPIC_AUTH_TOKEN` env var
- **`RequestAuth` async trait hook**：供 `llmsdk-anthropic-aws` 复用
- **`internal` 模块**：`Inner` / `InnerBuilder`（endpoint +
  body_transform hooks）暴露供 `llmsdk-amazon-bedrock` /
  `llmsdk-google-vertex` 复用
- **`sanitize_json_schema` 模块**：完整移植 `ai-sdk` 上游
  `sanitize-json-schema.ts`

#### xAI (`llmsdk-xai`)

- Chat + Responses + Image + Video（首个 `VideoModel` impl，4 模式 +
  异步轮询 LRO）+ Files
- 7 个 typed tools：`web_search` / `x_search` / `code_execution` /
  `view_image` / `view_x_video` / `file_search` / `mcp_server`

#### Mistral (`llmsdk-mistral`)

- Chat + Embedding
- 特性：`prefix` 续写 / `random_seed` / `document_image_url` /
  `safe_prompt` / magistral 的 `thinking` + `reasoning_effort`
  （`reasoning_content` → `Content::Reasoning`）

#### Azure OpenAI (`llmsdk-azure`)

- 复用 `llmsdk-openai::internal::{Inner, UrlStrategy}`
- Chat + Completion + Responses + Embedding + Image + Speech + Transcription
- URL 双模式：deployments-based 与 `openai/v1` 兼容；api-key 认证

#### Cohere (`llmsdk-cohere`)

- Chat + Embedding + Reranking（首个 `RerankingModel` impl）
- Cohere v2 wire；`tool_plan` → `Content::Reasoning`；
  `citations` → `Content::Source`

#### Google Gemini (`llmsdk-google`)

- Language（Gemini）+ Embedding + Image（Imagen）+ Video（Veo LRO）+
  Files（resumable upload）+ Interactions 模块（Gemini 多轮交互专用表面）
- 8 个 typed tools：`google_search` / `google_search_retrieval` /
  `enterprise_web_search` / `code_execution` / `url_context` /
  `file_search` / `google_maps` / `vertex_rag_store`
- JSON Schema → OpenAPI 3.0 转换
- `includeThoughts` + `thinkingBudget` reasoning

#### Anthropic on AWS (`llmsdk-anthropic-aws`)

- Claude on AWS（Anthropic 自有 AWS 部署，service =
  `aws-external-anthropic`）
- 双认证：SigV4 或 API Key
- workspace-id header

#### Amazon Bedrock (`llmsdk-amazon-bedrock`)

- Converse API + ConverseStream（EventStream binary）
- Embedding（Titan / Cohere Embed / Nova family dispatch）
- Image（Nova，5 task types）
- Anthropic on Bedrock（复用 `llmsdk-anthropic::internal::InnerBuilder`
  的 endpoint + body_transform hooks）
- Reranking（`amazon.rerank-v1` / `cohere.rerank-v3-5`）
- Mantle provider（自托管 Mantle 入口）
- reasoning effort 完整映射 + Anthropic 三分支路由 + assistant
  part-level cache points + thinking 完整路由

#### Google Vertex (`llmsdk-google-vertex`)

- Vertex Gemini + Embedding + Image + Video
- Anthropic on Vertex + xAI on Vertex + MaaS（OpenAI-compatible）
- 双模式：Express Mode（API Key）+ Standard Mode（OAuth via `gcp_auth`）
- global location 特殊处理

### Provider matrix

| Provider | Lang | Stream | Tools | Reason. | Embed | Image | Video | Rerank | Files | Skills | Speech / STT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenAI               | ✓ Chat + Completion + Responses | ✓ | 11 (Responses) | ✓ | ✓ | ✓ (DALL-E 3 / gpt-image-1) | — | — | ✓ | ✓ | ✓ / ✓ |
| Anthropic            | ✓ Messages | ✓ | 20 | ✓ | — | — | — | — | ✓ | ✓ | — / — |
| xAI                  | ✓ Chat + Responses | ✓ | 7  | ✓ | — | ✓ | ✓ | — | ✓ | — | — / — |
| Mistral              | ✓ Chat | ✓ | — | ✓ | ✓ | — | — | — | — | — | — / — |
| Azure (OpenAI)       | ✓ Chat + Completion + Responses | ✓ | 11 | ✓ | ✓ | ✓ | — | — | — | — | ✓ / ✓ |
| Cohere               | ✓ Chat | ✓ | — | ✓ | ✓ | — | — | ✓ | — | — | — / — |
| Google Gemini        | ✓ | ✓ | 8 | ✓ | ✓ | ✓ (Imagen) | ✓ (Veo) | — | ✓ | — | — / — |
| Anthropic on AWS     | ✓ Messages | ✓ | 20 | ✓ | — | — | — | — | ✓ | ✓ | — / — |
| Amazon Bedrock       | ✓ Converse + Anthropic | ✓ | 19 | ✓ | ✓ | ✓ (Nova) | — | ✓ | — | — | — / — |
| Google Vertex        | ✓ Gemini + Anthropic + xAI + MaaS | ✓ | 8 + 20 | ✓ | ✓ | ✓ | ✓ | — | — | — | — / — |

### Documentation

- `architecture/0001-trait-design.md` —— trait 层 ground truth
- `architecture/0002-middleware-design.md` —— middleware 设计
- `architecture/0003-m10-design.md` —— ai-sdk v4 全量对齐
- `architecture/0004-m11-responses-design.md` —— OpenAI Responses API
- `architecture/0005-m12-anthropic-full-design.md` —— Anthropic 全量
- `architecture/0006-m13-design.md` —— 8 个一线大厂 provider 接入

### Compatibility

- 与 `ai-sdk` v4 wire 协议完全对齐（`Tool::Provider` 序列化 tag 为
  `"provider"`、`ToolCallPart.dynamic` 字段、`StreamPart::File` /
  `StreamPart::ReasoningFile` variant、`LanguageModel`
  `specificationVersion`、Anthropic `supportsStrictTools` …）
- 命名差异：Rust API 命名遵循 Rust 习惯（snake_case），JSON wire
  字段名与 `ai-sdk` 保持一致
- Rust 侧 trait 表面在 M1–M14 期间共 14 处变更，**全部为新增/补漏**，
  无破坏性历史

[0.1.0]: https://github.com/zerx-lab/llmsdk/releases/tag/v0.1.0
