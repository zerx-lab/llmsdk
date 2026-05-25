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
- 禁止估算任务工作时间，不能因为时长而去过度分割工作
- 测试 provider 兼容性时调用 `provider-contract-test` skill
- **禁止"推迟特性到下一里程碑"作为默认**：从 M10 起，新阶段启动前必须列出本阶段
  要实现的**全部**特性并落地；实施中如发现某特性比预期复杂，必须停下来与用户对齐
  （扩范围 / 改阶段定义 / 拒绝该特性），不允许静默推迟到下阶段。设计文档不再写
  "第一轮不覆盖"段作为常态选项。历史推迟项（M1–M9 累积，见 `todo.md`）保留作为
  事实记录，启动 M10 时必须全部纳入（除非用户明确放弃某项）。

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

当前进度：M1–M11 完成。321 个 workspace 测试全绿；
`cargo fmt --check`、`cargo clippy --workspace --all-targets -- -D warnings` 通过。

```
M1 ✓ llmsdk-provider 编译通过；trait + 类型 ready
M2 ✓ llmsdk-provider-utils: HTTP/SSE/load_api_key
M3 ✓ llmsdk-openai: do_generate + contract::chat_basic 通过
M4 ✓ llmsdk-openai: do_stream + contract::chat_stream 通过
M5 ✓ llmsdk-openai: EmbeddingModel + contract::embed_basic 通过
M6 ✓ llmsdk-anthropic: Messages API (do_generate + do_stream)
M7 ✓ 推迟特性补齐：
     - OpenAI reasoning models（o1/o3/o4-mini/gpt-5*；reasoning_effort 透传、
       不支持参数剥离、max_completion_tokens 映射、system→developer 角色）
     - OpenAI logprobs（provider_options.openai.logprobs 透传 + provider_metadata 收集）
     - OpenAI url_citation annotations → Content::Source / StreamPart::Source
     - OpenAI search-preview 模型 temperature 剥离
     - Anthropic thinking blocks（visible + redacted；
       Content::Reasoning 出入；SSE thinking_delta / signature_delta；
       请求 thinking 字段 + 采样参数剥离 + max_tokens 自动加 budget）
     - 不动 provider trait；新增 chat::capabilities / chat::options /
       messages::options 内部模块
M8 ✓ llmsdk-openai: ImageModel (DALL-E 3 + gpt-image-1*)
     - POST /v1/images/generations，b64_json 解码 → GeneratedImage
     - provider_options.openai: quality/style/background/outputFormat/
       outputCompression/moderation/user 透传
     - aspectRatio + seed 自动告警（OpenAI Images API 不支持）
     - 按 model id 自动判断 max_images_per_call + 是否发 response_format
     - revised_prompt / size / created 等收集到
       provider_metadata.openai.images[]
     - trait 改动：ImageResult 加 warnings 字段（与 GenerateResult /
       EmbedResult 对齐；此前为首个 ImageModel impl 故零下游破坏）
     - 内嵌 base64 decoder（RFC 4648 §4），不引新依赖
M9 ✓ llmsdk-provider: LanguageModel middleware 层
     - `LanguageModelMiddleware` trait（6 个方法全部默认 no-op）+
       `wrap_language_model(model, [m1, m2])` 组合器（列表头最外层）
     - `Wrapped` 包装：override_provider / override_model_id 构造时缓存，
       override_supported_urls / transform_params / wrap_generate /
       wrap_stream 每次调用
     - 内置 RetryMiddleware：max_attempts/initial_backoff/multiplier/
       max_backoff builder，仅重 `is_retryable`，stream 仅打开前重
     - 内置 LoggingMiddleware：自有 Logger trait（不绑 tracing）+
       StderrLogger 示例；start/end/error 三类事件；log_prompt 默认 off
     - 内置 CacheMiddleware：CacheStore（同步签名，避免 rt feature）
       + MemoryCacheStore；key = DefaultHasher(serde_json::to_vec
       (&CallOptions))；stream 用 CapturingStream 边走边收，错误不写入；
       命中标 `provider_metadata.llmsdk.cache = "hit"`
     - 依赖：tokio time feature（生产）+ macros/rt/rt-multi-thread/
       test-util（dev）；零其它新增
     - trait 零改动；与 0001 trait 完全兼容
     - 设计文档 `architecture/0002-middleware-design.md`；新增 20 测试
       （4 trait + 6 retry + 4 logging + 6 cache）
M10 ✓ 全量 ai-sdk feature parity（除 Gemini / Responses 端点 / Files 端点）：
     - 依赖增量：schemars 1.x（仅此一项；jitter / multipart / LRU 全部
       自实现）
     - trait 改动：JsonSchema = schemars::Schema；ImageOptions.files/mask；
       ImageResult.usage + ImageUsage / ImageUsageInputDetails
     - Middleware 表面统一：EmbeddingModelMiddleware + ImageModelMiddleware
       + wrap_embedding_model / wrap_image_model / wrap_provider
       (ProviderMiddlewareSet)
     - 5 个内置 ai-sdk middleware：default_settings(_embedding) /
       extract_reasoning / simulate_streaming / extract_json /
       add_tool_input_examples
     - CacheMiddleware TTL + LRU (builder)；RetryMiddleware jitter_ratio；
       LoggingMiddleware with_stream_parts；MiddlewareContext via
       provider_options["llmsdk"]
     - OpenAI Chat：prediction/store/metadata/service_tier(flex/priority
       校验)/safety_identifier/prompt_cache_key/parallel_tool_calls/user/
       logit_bias(reasoning 自动剥离)/text.verbosity/top_logprobs/
       strict_json_schema 全部 provider option 透传；流式 error chunk
       type+code 提取；accepted/rejected_prediction_tokens 入
       provider_metadata；web_search_preview tool 路由（其它 9 个
       Responses-API 工具 todo）
     - OpenAI Image：POST /v1/images/edits + variations（multipart/form-data，
       手写 RFC 7578）；inputFidelity；gpt-image-1 usage → ImageResult.usage；
       ResponseInfo.id 用 created 时间戳合成
     - llmsdk-provider-utils：新增 multipart 模块 + post_raw / RawRequest
     - Anthropic Messages：8 种服务器工具路由 + 自动 anthropic-beta header；
       响应 9 种 *_tool_use / *_tool_result 块解析；cache_control 5 种位置；
       citations + title + context 进 document block；非图片文件
       (PDF/text/plain) → document，audio 剥离；context_management /
       container 透传 + beta header；compaction 响应块；tool_use caller /
       dynamic metadata 透传；text 上的 citations[] 透传；
       include_raw_chunks 选项；thinking adaptive type；thinking budget
       默认 1024
     - workspace 健康：191 测试全绿；fmt + clippy -D warnings 通过
     - 设计文档：`architecture/0003-m10-design.md`
M10.5 ✓ Chat API review fix-pack：
     - **trait 改动**（已对齐用户）：
       * StreamPart 新增 `File(FilePart)` + `ReasoningFile { data, media_type,
         provider_metadata }` variant — 对齐 ai-sdk LanguageModelV4StreamPart
       * `Tool::Provider` 序列化 tag 从 `"provider-defined"` 改为 `"provider"`
         — 对齐 ai-sdk v4 wire 格式（破坏性变更）
     - **OpenAI Chat 补齐 3 项 provider options**：
       * `promptCacheRetention`（`in_memory` / `24h`）透传到 wire 顶级
       * `systemMessageMode`（`system` / `developer` / `remove`）手动覆盖
         自动模式识别 + Remove 变体丢弃 system 消息
       * `maxCompletionTokens` 显式透传（优先于 max_output_tokens 自动映射）
       * capabilities.rs 加 `supports_flex_processing` + `supports_priority_processing`；
         service_tier flex/priority 按模型能力剥离并 warning（之前仅格式校验）
     - **Anthropic Messages 补齐 11 项 provider options**：
       sendReasoning（false 时剥离 reasoning 块）/ structuredOutputMode（outputFormat
       路径走 output_config.format + sanitize_json_schema）/ disableParallelToolUse
       （进 tool_choice 三个 variant）/ cacheControl（顶级 wire）/ metadata.userId
       → metadata.user_id / mcpServers（camelCase→snake_case 字段重命名 + 嵌套
       toolConfiguration）/ toolStreaming（默认 true → 函数工具 eager_input_streaming）
       / effort + taskBudget（进 output_config）/ speed / inferenceGeo / anthropicBeta
       （加 anthropic-beta header tokens）
     - **Anthropic server tools 完整对齐 ai-sdk 上游 20 个带版本号 tool ID**：
       移除原 8 个简写（anthropic.web_search 等）（破坏性变更），改用 ai-sdk 原始
       命名：anthropic.code_execution_{20250522,20250825,20260120} /
       anthropic.computer_{20241022,20250124,20251124} /
       anthropic.text_editor_{20241022,20250124,20250429,20250728} /
       anthropic.bash_{20241022,20250124} / anthropic.memory_20250818 /
       anthropic.web_fetch_{20250910,20260209} /
       anthropic.web_search_{20250305,20260209} /
       anthropic.tool_search_{regex,bm25}_20251119 /
       anthropic.advisor_20260301（旧 anthropic.advisor → advisor_20251020 升版）
       每个 ID 带正确的 wire `type` + 强制 `name`（如 text_editor → str_replace_*）
       + 对应 beta header tokens
     - **新增 sanitize_json_schema 模块**（llmsdk-anthropic 内部）：
       完整移植 ai-sdk 上游 sanitize-json-schema.ts，零新依赖
     - workspace 健康：221 测试全绿（+28 新契约测试）；fmt + clippy 通过
     - 新增契约测试文件：`crates/llmsdk-anthropic/tests/contract_messages_options.rs`
M11 ✓ OpenAI Responses API 全量接入（POST /v1/responses）：
     - 新 `OpenAiResponsesLanguageModel` 与 `OpenAiChatModel` 并存；
       `OpenAi::responses(model_id)` 工厂入口
     - 22 项 `provider_options.openai.*` 全量透传 + 模型能力校验：
       conversation/previousResponseId（互斥）/include/instructions/logprobs（bool|≤20）/
       maxToolCalls/metadata/parallelToolCalls/promptCacheKey/promptCacheRetention/
       reasoningEffort（none/xhigh 校验）/reasoningSummary/safetyIdentifier/
       serviceTier（flex/priority 模型能力校验剥离）/store/passThroughUnsupportedFiles/
       strictJsonSchema/textVerbosity/truncation/user/systemMessageMode/forceReasoning/
       contextManagement/allowedTools（覆盖 toolChoice）
     - 11 个 provider-defined tools 完整 args/output/tool_choice 路由：
       web_search / web_search_preview / file_search / code_interpreter /
       image_generation / local_shell / shell（containerAuto/containerReference 标
       provider_executed）/ apply_patch / mcp（serverUrl 或 connectorId 强制校验）/
       custom（含 grammar/text format）/ tool_search（server/client 双路径）
     - 18 种 output item type 非流式解析（reasoning/message/function_call/
       custom_tool_call/web_search_call/file_search_call/code_interpreter_call/
       image_generation_call/local_shell_call/shell_call/shell_call_output/
       mcp_call/mcp_list_tools/mcp_approval_request/computer_call/apply_patch_call/
       compaction/tool_search_call/tool_search_output）+ 4 种 annotation
       → Content::Source（url_citation/file_citation/container_file_citation/file_path）
     - 20+ SSE event type 流式状态机：reasoning summary 三态机
       （active/can-conclude/concluded，store=true 即时 conclude vs store=false
       延迟到 output_item.done 一次性 conclude）+ apply_patch hasDiff/endEmitted +
       image_generation partial_image preliminary metadata 标记
     - Prompt → input items 转换：systemMessageMode 3 态（system/developer/remove）
       + reasoning 模型自动 developer + 11 种 assistant tool-call 路由
       （function/custom + apply_patch + local_shell + provider-executed →
       item_reference 回填）+ user 内容 5 种（text/image url/image data/image
       file_id ref/pdf/passThrough file）+ MCP approval_response
     - **trait 改动 1 处**：`ToolCallPart` 加 `dynamic: Option<bool>` 字段，与
       `StreamPart::ToolCall.dynamic` 对齐（MCP 工具非流式表达 runtime tool name）；
       Option + serde skip_if 向后兼容
     - 模块结构：`crates/llmsdk-openai/src/responses/`：mod / model / options /
       finish_reason / usage / convert_prompt / parse_response / stream /
       prepare_tools / tools/{11 个 tool}.rs / wire/{request,response,chunk}.rs
     - 设计文档：`architecture/0004-m11-responses-design.md`
     - 5 套契约测试：contract_responses_{basic,stream,tools,options,advanced}.rs
       （23 个 wiremock 用例）+ 64 个 responses 单元测试
     - workspace 健康：321 测试全绿（M10.5 → M11 +100）；fmt + clippy 通过
     - subagent 审核 PASS（与 ai-sdk 上游 100% 特性对齐）
```

**已验证的 trait 抽象**：M1–M11 累计 trait 改动仅 9 处（M8 `ImageResult.warnings`
补漏 + M10 `JsonSchema = schemars::Schema`、`ImageOptions.files/mask`、
`ImageResult.usage`、新增 `ImageUsage`/`ImageUsageInputDetails` + M10.5 `StreamPart::File` /
`StreamPart::ReasoningFile` 两个 variant + `Tool::Provider` wire tag 变更 + M11
`ToolCallPart.dynamic`）。三个模型表面（Language/Embedding/Image）+ 三层 middleware
+ 两个 provider 三个端点（Chat + Responses + Anthropic Messages）的所有 ai-sdk v4
特性都基于这套 trait 消化。

**下一阶段候选**（待规划）：
- 第三个 provider（Gemini）以验证 trait 抽象在第三家上的稳定性
- Anthropic Files API 端点
- 详见 `todo.md` 中"M12+ 推迟"段

**跨越里程碑/阶段禁止**。开新阶段前必须停下来对齐，并按"强制规则"末条
列出本阶段**全部**特性（不允许中途推迟，详见强制规则段）。

## Checkpoint 规则

- 每完成 1 个 trait 定义 → **启动 subagent 对照 ai-sdk 上游审核能力一致性**；通过则继续 impl，不通过则按 subagent 反馈修正后再审一次
- 每完成 1 个 provider 的 1 个 capability（text / stream / tool / embed）→ 跑契约测试 + 启动 subagent 审核；都通过则继续，否则停下来反馈
- 需要修改 `crates/llmsdk-provider` 的 trait → 必须停下来说明影响范围，不准静默改动（此项仍需人工审核）
- 需要新增依赖 → 必须在对话里列出依赖名 + 用途，等确认后用 `cargo add` 添加（此项仍需人工审核）

### Subagent 审核协议

启动 `Explore` 类型 subagent，prompt 必须包含：

1. 本轮 Rust 改动落地的文件路径 + 公开 API/trait 签名
2. 对照的 ai-sdk 上游路径（`/home/zero/Desktop/code/github/ai/packages/...`）
3. `architecture/` 下相关设计文档路径
4. 要求 subagent 检查：
   - 上游每一个公开能力（method / hook / 字段）Rust 侧是否都有对应表达，或在文档中显式声明推迟
   - Rust 侧是否多出上游没有的语义（若有，必须在设计文档解释）
   - 与设计文档的偏差
5. 要求 subagent 输出："PASS" + 一句结论；或 "FAIL" + 缺失/偏差清单（按修复优先级排序）

PASS 即可继续下一步；FAIL 则按清单修复后重审。审核结果摘要直接说给用户听，不要存到 memory。
