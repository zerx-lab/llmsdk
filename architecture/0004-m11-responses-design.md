# 0004 — M11 OpenAI Responses API 全量对齐

> Status: in progress
> Upstream reference: `vercel/ai` @ `packages/openai/src/responses/**` + `packages/openai/src/tool/**`
> Prereqs: `0001-trait-design.md`、`0002-middleware-design.md`、`0003-m10-design.md`

## Goal

把 `POST /v1/responses` 端点（OpenAI Responses API）完整接入 llmsdk，
作为与现有 `OpenAiChatModel` 并存的第二条 LanguageModel 实现 `OpenAiResponsesLanguageModel`。
对齐 ai-sdk `OpenAIResponsesLanguageModel`（2326 行 `openai-responses-language-model.ts` +
981 行 `convert-to-openai-responses-input.ts` + 1401 行 `openai-responses-api.ts` +
466 行 `openai-responses-prepare-tools.ts` + 11 个 provider-defined tool schema 共
1433 行）。

强制规则要求"启动新阶段前必须列出全部范围"——本文档即范围 ground truth；不允许
中途静默推迟。开始前已与用户对齐 3 处范围决策（见末尾 "Open Questions Resolved"）。

## 范围（全部纳入）

### A. 端点与模型表面

1. `POST /v1/responses`（非流 + 流式 SSE）
2. 新 `OpenAiResponsesLanguageModel` 实现 `LanguageModel` trait
3. 与现有 `OpenAiChatModel` 并存；Chat API 路径不动（含 `web_search_preview` 路由）
4. `OpenAi::responses(model_id)` provider 入口（与 `chat` / `embedding` / `image` 并列）
5. 复用现有 `chat::capabilities`（isReasoningModel / systemMessageMode 推断 / flex / priority）

### B. Provider Options（22 项，全量透传）

`provider_options.openai.*`：

| key                          | 类型                                                     | 行为                                                                |
| ---------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------- |
| `conversation`               | `Option<String>`                                         | 与 `previousResponseId` 互斥；同时设置 → warning + 走 previous       |
| `previousResponseId`         | `Option<String>`                                         | 顶级 `previous_response_id`                                          |
| `include`                    | `Option<Vec<IncludeValue>>`                              | 7 值枚举 + auto-include (logprobs/web_search sources/code_interp outputs/reasoning encrypted) |
| `instructions`               | `Option<String>`                                         | 顶级 `instructions`                                                  |
| `logprobs`                   | `Option<LogprobsOpt>` (bool 或 1..=20)                   | 自动 include `message.output_text.logprobs`；进 metadata          |
| `maxToolCalls`               | `Option<u32>`                                            | 顶级 `max_tool_calls`                                                |
| `metadata`                   | `Option<JsonValue>`                                      | 顶级 `metadata`                                                      |
| `parallelToolCalls`          | `Option<bool>`                                           | 顶级 `parallel_tool_calls`                                           |
| `promptCacheKey`             | `Option<String>`                                         | 顶级 `prompt_cache_key`                                              |
| `promptCacheRetention`       | `Option<"in_memory" \| "24h">`                           | 顶级 `prompt_cache_retention`                                        |
| `reasoningEffort`            | `Option<String>` (含 `none`/`xhigh` 模型校验)            | 仅 reasoning 模型；非 reasoning → warning 剥离                       |
| `reasoningSummary`           | `Option<String>` (`auto` / `detailed`)                   | 仅 reasoning 模型                                                    |
| `safetyIdentifier`           | `Option<String>`                                         | 顶级 `safety_identifier`                                             |
| `serviceTier`                | `Option<"auto" \| "flex" \| "priority" \| "default">`    | flex/priority 模型能力校验，不支持 → warning + 剥离                  |
| `store`                      | `Option<bool>` (默认 true)                               | 顶级 `store`；非 store + reasoning model → 自动加 reasoning.encrypted_content include |
| `passThroughUnsupportedFiles`| `Option<bool>`                                           | 非 image/pdf 文件是否走 `input_file`                                 |
| `strictJsonSchema`           | `Option<bool>` (默认 true)                               | JSON schema response format strict                                   |
| `textVerbosity`              | `Option<"low" \| "medium" \| "high">`                    | 顶级 `text.verbosity`                                                |
| `truncation`                 | `Option<"auto" \| "disabled">`                           | 顶级 `truncation`                                                    |
| `user`                       | `Option<String>`                                         | 顶级 `user`                                                          |
| `systemMessageMode`          | `Option<"system" \| "developer" \| "remove">`            | 手动覆盖；不设走 capabilities 自动判断                              |
| `forceReasoning`             | `Option<bool>`                                           | 强制视为 reasoning model；同时 systemMessageMode 默认 developer       |
| `contextManagement`          | `Option<Vec<{ type: "compaction", compactThreshold }>>` | 顶级 `context_management`                                            |
| `allowedTools`               | `Option<{ toolNames, mode? }>`                           | 覆盖 toolChoice → `tool_choice: { type: "allowed_tools", ... }`      |

### C. 11 个 Provider-Defined Tools

每个 tool 包含：args schema (输入) + output schema (响应/result) + wire 路由。

| `Tool::Provider { id }`       | wire `type`        | args 字段                                                      | output 字段                                         |
| ----------------------------- | ------------------ | -------------------------------------------------------------- | --------------------------------------------------- |
| `openai.web_search_preview`   | `web_search_preview` | `searchContextSize?`, `userLocation?`                          | wraps web search action                             |
| `openai.web_search`           | `web_search`       | `externalWebAccess?`, `filters?`, `searchContextSize?`, `userLocation?` | `action: { search/openPage/findInPage }`, `sources` |
| `openai.file_search`          | `file_search`      | `vectorStoreIds`, `maxNumResults?`, `ranking?`, `filters?`     | `queries[]`, `results[]?`                            |
| `openai.code_interpreter`     | `code_interpreter` | `container` (string 或 `{ fileIds? }` 或 null→auto)             | `outputs[]` (logs/image)                             |
| `openai.image_generation`     | `image_generation` | 12 字段 (background/inputFidelity/inputImageMask/model/moderation/outputCompression/outputFormat/partialImages/quality/size 等) | `result: base64`                                     |
| `openai.local_shell`          | `local_shell`      | 无参                                                            | `action: { type: "exec", command[], timeoutMs?, user?, workingDirectory?, env? }` |
| `openai.shell`                | `shell`            | `environment?: local/containerAuto/containerReference` + skills | `output: [{ stdout, stderr, outcome: exit/timeout }]` |
| `openai.apply_patch`          | `apply_patch`      | 无参                                                            | `operation: create_file/delete_file/update_file`     |
| `openai.mcp`                  | `mcp`              | `serverLabel`, `allowedTools?`, `requireApproval?`, `authorization?`, `connectorId?`, `headers?`, `serverDescription?`, `serverUrl?` | `{ type: "call", serverLabel, name, arguments, output?, error? }` |
| `openai.custom`               | `custom`           | `description?`, `format?: grammar/text`                         | input string                                         |
| `openai.tool_search`          | `tool_search`      | `execution?`, `description?`, `parameters?`                     | `tools[]`                                            |

### D. 输入转换 `convert_prompt`

对照 `convert-to-openai-responses-input.ts` (981 行)：

- **systemMessageMode**：3 种（system/developer/remove）+ reasoning model 自动 developer
- **user message** content parts：
  - text → `input_text`
  - image (url/base64) → `input_image`
  - pdf → `input_file` (file_url/file_data/file_id)
  - 其它文件 → `input_file` (受 `passThroughUnsupportedFiles` 控制；否则 warning 剥离)
  - source(url) → 文本拼接（ai-sdk 行为）
- **assistant message** content parts (11 种)：
  - text → `assistant` message item (含 phase)
  - reasoning (带 `encryptedContent`) → `reasoning` item
  - tool-call (function) → `function_call` item
  - tool-call (custom) → `custom_tool_call` item
  - tool-call (provider executed - 9 种) → 对应 *_call item
- **tool message** → `function_call_output` / `custom_tool_call_output` / 各 provider tool 的 *_output item
- MCP `tool-approval-response` → `mcp_approval_response` item
- file_id 前缀：来自 `OpenAiConfig.fileIdPrefixes` (本阶段简化为常量列表)
- `conversation` / `previousResponseId` 影响序列化：仅传 delta items

### E. 输出解析（非流 18 种 item types）

按 `output[].type` 路由：

| item type                | → Content[]                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `reasoning`              | `Reasoning` (每个 summary 一条；空 summary push 一条空)      |
| `message`                | `Text` + 4 种 annotation → `Source`                         |
| `function_call`          | `ToolCall`                                                   |
| `custom_tool_call`       | `ToolCall`                                                   |
| `web_search_call`        | `ToolCall` (provider_executed) + `ToolResult` (action 映射)  |
| `file_search_call`       | `ToolCall` (provider_executed) + `ToolResult` (queries/results) |
| `code_interpreter_call`  | `ToolCall` (provider_executed) + `ToolResult` (outputs)      |
| `image_generation_call`  | `ToolCall` (provider_executed) + `ToolResult` (result)       |
| `local_shell_call`       | `ToolCall` (with action)                                     |
| `shell_call`             | `ToolCall` (provider_executed 视 environment 而定)           |
| `shell_call_output`      | `ToolResult` (output[])                                      |
| `mcp_call`               | `ToolCall` (dynamic=true, provider_executed) + `ToolResult`  |
| `mcp_list_tools`         | skip                                                         |
| `mcp_approval_request`   | `ToolCall` + `ToolApprovalRequest`                           |
| `computer_call`          | `ToolCall` (provider_executed) + `ToolResult` (status)       |
| `apply_patch_call`       | `ToolCall` (with operation)                                  |
| `tool_search_call`       | `ToolCall` (server → provider_executed)                      |
| `tool_search_output`     | `ToolResult` (tools[])                                       |
| `compaction`             | `Custom { kind: "openai.compaction" }`                       |

### F. SSE 流式状态机（30+ event types）

按 `chunk.type` 路由：

- `response.created` → `StreamPart::ResponseMetadata { id, timestamp, modelId }`
- `response.output_item.added` → text-start / reasoning-start / tool-input-start / tool-call(预发) 等
- `response.output_item.done` → text-end / 各种 tool-call / tool-result final 等
- `response.output_text.delta` → `TextDelta`
- `response.output_text.annotation.added` → `Source` (+ 收集到 text-end metadata)
- `response.reasoning_summary_part.added/done` + `response.reasoning_summary_text.delta`
  → reasoning-start/end + reasoning-delta；`store=true` 时立即 conclude，否则 can-conclude
- `response.function_call_arguments.delta` → `ToolInputDelta`
- `response.custom_tool_call_input.delta` → `ToolInputDelta`
- `response.code_interpreter_call_code.delta/done` → escapeJSON 包装为 JSON input delta
- `response.image_generation_call.partial_image` → `ToolResult { preliminary: true }`
- `response.apply_patch_call_operation_diff.delta/done` → JSON input delta + end
- `response.completed` / `response.incomplete` → 收集 usage + finishReason
- `response.failed` → 收集 error + finishReason
- `error` (顶层) → `StreamPart::Error`
- 未知 chunk → 透传 raw（如果 `include_raw_chunks`）

### G. Annotations → `Content::Source` / `StreamPart::Source`

| annotation type            | Source variant                              | providerMetadata                            |
| -------------------------- | ------------------------------------------- | ------------------------------------------- |
| `url_citation`             | `Source::Url { url, title }`                | -                                           |
| `file_citation`            | `Source::Document { mediaType: "text/plain", title, filename }` | `{ type: "file_citation", fileId, index }` |
| `container_file_citation`  | `Source::Document { ... }`                  | `{ type: "container_file_citation", fileId, containerId }` |
| `file_path`                | `Source::Document { mediaType: "application/octet-stream", title=fileId, filename=fileId }` | `{ type: "file_path", fileId, index }` |

### H. Provider Metadata / Finish reason / Usage

- `provider_metadata.openai`：
  - `responseId` / `serviceTier` / `logprobs[]?`
  - text item：`itemId` / `phase?` / `annotations[]?`
  - reasoning：`itemId` / `reasoningEncryptedContent?`
  - compaction：`type: "compaction"`, `itemId`, `encryptedContent`
  - source document：上表 4 种
- **finish reason** 映射 (`map_openai_responses_finish_reason`)：
  - `undefined`/`null` + has fn call → `tool-calls`
  - `undefined`/`null` 否则 → `stop`
  - `max_output_tokens` → `length`
  - `content_filter` → `content-filter`
  - default + has fn call → `tool-calls`
  - default 否则 → `other`
- **usage** (`convert_openai_responses_usage`)：
  - `inputTokens.total = input_tokens`
  - `inputTokens.cacheRead = input_tokens_details.cached_tokens`
  - `inputTokens.noCache = total - cacheRead`
  - `outputTokens.total = output_tokens`
  - `outputTokens.reasoning = output_tokens_details.reasoning_tokens`
  - `outputTokens.text = total - reasoning`

### I. Trait 改动（仅 1 处）

**`ToolCallPart` 加 `dynamic: Option<bool>` 字段**（与 `StreamPart::ToolCall.dynamic` 对齐）：
- MCP 工具非流式输出时表达 `dynamic=true` (运行时才知道 tool name)
- `Option<bool>` + `serde(default, skip_serializing_if = "Option::is_none")` → 向后兼容
- 同步更新 `0001-trait-design.md`

### J. 契约测试（7 套新增）

1. `contract_responses_basic` — 非流文本 + reasoning + usage
2. `contract_responses_stream` — 流式文本 + reasoning
3. `contract_responses_tools_function` — function tool 调用 + tool result 回环
4. `contract_responses_tools_provider` — web_search / file_search / code_interpreter / image_generation
5. `contract_responses_options` — 22 个 provider option 全部透传 + reasoning 模型校验
6. `contract_responses_mcp_approval` — MCP approval_request → ToolApprovalRequest + approval_response 回环
7. `contract_responses_apply_patch_stream` — apply_patch 流式 diff delta 拼接

## 不在本阶段（推迟到 M12+）

- Anthropic Files API endpoint（独立工作）
- Gemini provider（独立工作）
- Responses API 的 fileIdPrefixes 用户可配置化（本阶段硬编码常量）
- MCP approval 的双向回填（本阶段单次调用内 stateless）

## 模块拆分（新增文件）

```
crates/llmsdk-openai/src/responses/
├── mod.rs                      # OpenAiResponsesLanguageModel + LanguageModel impl
├── model.rs                    # struct + builder + supported_urls
├── options.rs                  # 22 项 provider options + 校验
├── tools/
│   ├── mod.rs
│   ├── web_search.rs           # args + output
│   ├── web_search_preview.rs
│   ├── file_search.rs
│   ├── code_interpreter.rs
│   ├── image_generation.rs
│   ├── local_shell.rs
│   ├── shell.rs                # 含 environment 三态 + skills
│   ├── apply_patch.rs
│   ├── mcp.rs
│   ├── custom.rs
│   └── tool_search.rs
├── wire/
│   ├── mod.rs
│   ├── request.rs              # RequestBody + input items (19 种)
│   ├── response.rs              # ResponseBody + output items (18 种)
│   └── chunk.rs                 # 30+ SSE event 类型
├── convert_prompt.rs            # Prompt → input items
├── parse_response.rs            # output items → Content[]
├── stream.rs                    # SSE 状态机
├── prepare_tools.rs             # Tool 路由
├── finish_reason.rs             # map_openai_responses_finish_reason
└── usage.rs                     # convert_openai_responses_usage
```

预估 LOC：~3500 行 Rust（含 inline 测试 + 契约测试 ~800 行）。

## Open Questions Resolved（已与用户对齐）

1. **trait 改动**：`ToolCallPart` 加 `dynamic: Option<bool>` ✓
2. **Chat API web_search_preview**：保留双路径 ✓
3. **computer_use tool**：响应侧解析纳入；请求侧 args schema 不暴露（与上游 stealth 一致）✓
