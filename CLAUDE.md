# CLAUDE.md

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

## 代码风格
- 优先复用项目已有的 trait / error 类型，不要平行造轮子
- 单文件超过 600 行考虑拆分；单函数超过 80 行需要说明

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

### Subagent 反误判规则（强制）

近期对照审计中 subagent 报 FAIL 误判率 ~64%（第一轮 14 项 CRITICAL/HIGH 仅 5 项真成立）。根因：
- 未读上游对应文件就猜"上游应该有 X"
- 未追溯调用时机就断"A 与 B 两处不一致"
- 未读完整 `match` 分支就声称"事件被忽略"
- 未在脑里执行 `starts_with` / `contains` 表达式就否定路由逻辑
- 把"上游可能有"当作"上游有"
- 未逐字段对照就说"builder 不完整"

启动审计 subagent 时，prompt **必须**强制以下证据链：

1. **上游证据先行**：每条 "Rust 缺失 X" 断言必须先给出上游确切路径+行号+≥3 行代码证明上游真实现了 X。缺此证据则结论改为 "PASS / 上游同样不实现"。
2. **多路径不一致必给 caller 链**：判定 "A 与 B 不一致" 时必须列出二者的实际 caller / 生命周期阶段，证明会被同一次调用同时命中。否则结论改为 "PASS / 独立路径"。
3. **enum/match 必读全部分支**：判定 "事件被忽略" 时必须列出完整 `match` 的所有 variant 与对应处理，证明该 variant 在所有分支都未被处理。只看到一处 `=> {}` 不构成证据。
4. **wire 字段必查 fixture**：判定 "字段未传递" 时优先查上游 `.test.ts` fixture / `__fixtures__` / snapshot 文件，schema 中存在字段不等于上游实际填充该字段。
5. **字符串匹配必先执行一遍**：涉及 `starts_with` / `contains` / `match modelId` 这类路由判断时，必须把待测 model_id / tool_id 代入实际表达式得出 true/false 后再下结论。
6. **默认 PASS、FAIL 门槛更高**：审计的默认结论是 PASS（与上游对齐）。判 FAIL 必须至少同时满足规则 1-5 中两项。

**主 agent 责任**：subagent 报告呈给用户前，必须对每条 CRITICAL/HIGH 反向验证：
- 给出最小可复现样例（用户怎么调用会触发该缺陷？）
- 给出上游对应测试用例 / fixture（上游凭哪个测试证明它支持该能力？）

任一项答不出来 → 该条降级为 LOW 或剔除，不得作为 CRITICAL/HIGH 上报。违反者整份报告重审。
