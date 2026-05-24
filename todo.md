# todo.md — 推迟事项与未完成项

> 入口文件：跨里程碑追踪所有 deferred / TODO 项目；任何阶段性停顿都先记到这里。

## 当前里程碑：M8 完成 ✓

新增 20 个测试，总数 146 全绿。详细范围见下方"已完成"。

## 下一阶段候选（待选定）

按 AGENTS.md 顺序，二选一开新阶段前需先对齐：

1. **M9 middleware 层** — retry / 日志 / 缓存；需要设计装饰器模式与 LanguageModel 配合
2. **M9 第三个 provider (Gemini)** — 进一步压力测试 trait 抽象

## 仍然推迟（M9+）

### OpenAI Chat
- prediction / store / metadata / service_tier / safety_identifier / prompt_cache_key (provider option 透传)
- text verbosity / parallel_tool_calls / user
- logit_bias 透传 + reasoning 模型上的剥离 warning
- top_logprobs（独立于 logprobs 的纯数字字段）— 当前合并在 logprobs 处理中
- strict_json_schema provider option（当前硬编码 `true`）
- flex / priority service tier 校验
- accepted_prediction_tokens / rejected_prediction_tokens 入 provider_metadata
- provider-defined tools（web_search_preview / file_search 等）
- 错误 envelope 在流中区分（已有 outer / `error` chunk，但未关联 provider-specific 错误代码）

### OpenAI Image
- image editing (`POST /v1/images/edits`)：需要 `ImageOptions::files` / `mask` 字段（trait 改动）
- image variations (`POST /v1/images/variations`)：同上
- `Usage` 字段（gpt-image-1 系列会报）：需要 `ImageResult::usage` 字段（trait 改动）
- input fidelity (`inputFidelity`) provider option：仅 editing 用得到，待 editing 落地一起
- ResponseInfo.id 关联：当前 `id=None`，OpenAI Images API 不返回 response id；可考虑用 `created` 时间戳合成

### Anthropic Messages
- 服务器工具：web_search / web_fetch / code_execution / mcp / bash / text_editor / tool_search / advisor
- citations / cache_control / context_management / containers
- 非图片文件部分（audio / pdf / document）
- compaction 块
- tool_use 中的 caller / dynamic / programmatic-tool-call 元数据
- raw chunks 透传 (`include_raw_chunks`) — 当前 stream 完全忽略
- thinking `adaptive` type（仅实现了 `enabled` / `disabled`）
- thinking budget 缺失时的默认 1024 + max_output_tokens 范围校验（当前直接 saturating_add）

### 通用
- middleware 层（retry / logging / caching） — 已在 AGENTS.md 列为下一阶段
- 第三个 provider（Gemini） — 已在 AGENTS.md 列为下一阶段
- schemars 切换（`JsonSchema = serde_json::Value` 现状）

## 已完成（历史）

### M8（最新）
- OpenAI Image Generation (`POST /v1/images/generations`) — DALL-E 3 / DALL-E 2 / gpt-image-1\* / chatgpt-image-\*
- 按 model id 自动判断 `max_images_per_call`（dall-e-3 / chatgpt-image-\* = 1，dall-e-2 / gpt-image-\* = 10，未知 = 1 安全默认）
- 按 model id 自动判断是否发送 `response_format=b64_json`（gpt-image-1\* / chatgpt-image-\* 默认 b64，其它必须显式）
- `provider_options.openai`: quality / style / background / outputFormat / outputCompression / moderation / user 透传
- `aspectRatio` / `seed` 自动告警 + 剥离（OpenAI Images API 不支持）
- `b64_json` → 内置 base64 decoder → `GeneratedImage.bytes`（避免引入 base64 依赖）
- `media_type` 检测：服务端 `output_format` 优先，否则 PNG/JPEG/WEBP/GIF magic bytes 嗅探，最后默认 `image/png`
- `revised_prompt` / `created` / `size` / `quality` / `background` / `outputFormat` 收集到 `provider_metadata.openai.images[]`
- `usage`（gpt-image-1）原样收集到 `provider_metadata.openai.usage`（等 trait 加 usage 字段后再上提）
- **Trait 改动**：`ImageResult` 增 `warnings: Vec<Warning>`（首个 ImageModel impl，零下游破坏；与 GenerateResult/EmbedResult 对齐）
- 错误路径：base64 解码失败 → `ProviderError::type_validation`；HTTP 400/429 复用现有 `rewrite_openai_error`

### M7
- OpenAI reasoning 模型族识别（o1/o3/o4-mini/gpt-5*，gpt-5-chat* 排除）
- `reasoning_effort` 透传：`CallOptions::reasoning` + `provider_options.openai.reasoningEffort` 双源（后者优先）
- reasoning 模型上自动剥离 temperature/top_p/frequency/presence/logprobs/top_logprobs + warning
- reasoning 模型 `max_tokens` → `max_completion_tokens` 自动映射
- reasoning 模型 system → developer 角色映射
- gpt-5.1+ + reasoning_effort=none 时保留 temperature/top_p/logprobs（`supports_non_reasoning_parameters`）
- `force_reasoning` provider option 覆盖 id 检测
- search-preview 模型族 temperature 自动剥离
- `provider_options.openai.logprobs`（bool / 数字）→ `logprobs` + `top_logprobs` wire fields
- `choice.logprobs.content` → `provider_metadata.openai.logprobs`（非流式 + 流式）
- `choice.message.annotations[].url_citation` → `Content::Source::Url`（非流式）/ `StreamPart::Source`（流式）
- Anthropic `thinking` provider option（enabled / disabled，含 budget_tokens）
- Anthropic thinking 启用时自动剥离 temperature/top_p/top_k + 调高 max_tokens (`+budget`)
- Anthropic `ResponseContent::Thinking` + `RedactedThinking` → `Content::Reasoning` + `provider_options.anthropic.signature` / `redactedData`
- Anthropic 出站 `AssistantPart::Reasoning` → wire `thinking` / `redacted_thinking` 块（signature 通过 provider_options 回传）
- Anthropic SSE `thinking_delta` / `signature_delta` / `redacted_thinking` block start → `StreamPart::Reasoning*`

### M1–M6
见 AGENTS.md 里程碑约束部分。
