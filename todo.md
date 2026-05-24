# todo.md — 推迟事项与未完成项

> 入口文件：跨里程碑追踪所有 deferred / TODO 项目；任何阶段性停顿都先记到这里。

## 当前里程碑：M7 完成 ✓

新增 34 个测试，总数 126 全绿。详细范围见下方"已完成"。

## 下一阶段候选（待选定）

按 AGENTS.md 顺序，三选一开新阶段前需先对齐：

1. **M8 ImageModel (OpenAI DALL-E 3)** — trait 已就绪，零 trait 改动；范围最小
2. **M8 middleware 层** — retry / 日志 / 缓存；需要设计装饰器模式与 LanguageModel 配合
3. **M8 第三个 provider (Gemini)** — 进一步压力测试 trait 抽象

## 仍然推迟（M8+）

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
- ImageModel 实现（DALL-E 3） — 已在 AGENTS.md 列为下一阶段
- 第三个 provider（Gemini） — 已在 AGENTS.md 列为下一阶段
- schemars 切换（`JsonSchema = serde_json::Value` 现状）

## 已完成（历史）

### M7（最新）
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
