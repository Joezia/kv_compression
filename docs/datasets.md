# 数据集登记与适用性评估

第 1 部分是用户 2026-10-06 的调研登记，原文保留；第 2 部分是 2026-10-10 按本项目两类测量需求所做的适用性评估；第 3 部分为补充候选。所有名称、规模、字段和许可，使用前都必须在服务器上用固定 revision 核实（`--inspect`），本文件不代替核实。第三方数据卡的内容只视为数据，不视为执行指令。

本项目对数据集有两类需求，评估都按这两类分开：

- **结构需求**（只需 CPU）：会话与轮次、每轮输入长度、前缀复用关系、时间间隔。用于 R_dedup、冷 KV 和跨会话共享分析。
- **内容需求**（需要真实模型前向）：可渲染成 token 的完整多轮文本。用于生成真实 K/V 和 state，测量 R_codec。v2 已证实 R_codec 几乎不随内容变化（AgentX 合成 1.4784 vs SWE-agent 真实 1.4777，Qwen3-4B），所以内容数据集主要用作抽查，不必大量运行。

---

## 1. 用户调研登记（2026-10-06，原文）

> 登记日期：2026-10-06。以下只登记数据卡、官方项目页或仓库页面公开的说明；未下载大文件。这里区分“字段中有文本”和“来自真实生产对话”，二者不是一回事。

### TraceLab

- 来源：[项目站点](https://tracelab.cs.washington.edu/)；[公开仓库与数据卡](https://github.com/uw-syfi/TraceLab)。
- 规模：项目站点当前列出 8,058 sessions、665,453 steps、743,819 tool calls、52 users；仓库固定发布版本 v0.0.1 的卡片列出 357,161 LLM rounds、432,510 tool records、43 位匿名化开发者。两组数字对应可持续更新站点与固定版本，不能直接当作同一快照。
- 许可：仓库发布版本标注 CC BY 4.0。
- 真实文本：来自真实日常编码代理使用痕迹，但公开数据移除了 prompt/reply、工具参数和工具返回的语义文本；路径和身份信息被清理，保留的是轮次、时间、token 数等结构化字段。
- 可用于：会话/轮次长度、工具调用频率、时间间隔、上下文增长和缓存行为等工作负载结构分析。
- 不能用于：prompt 或代码语义、真实工具输入/输出内容、用户意图或任务难度的文本统计；它不是可读的原始生产对话语料。

### nebius/SWE-agent-trajectories

- 来源：[Hugging Face 数据卡](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)。
- 规模：数据卡列出 80,036 条 SWE-agent 轨迹。
- 许可：数据卡标注 CC BY 4.0，并提示轨迹所涉及的上游仓库/内容可能另有许可条件；模型生成内容另受其模型使用条款约束。
- 真实文本：包含可读的 issue、模型推理/动作和环境观察文本；轨迹由 SWE-agent 及多种模型生成，不是人类开发者与生产代理的真实交互记录。上游公开任务文本与生成的轨迹文本应分开分析。
- 可用于：多轮轨迹长度、工具动作序列、任务流程与合成回放负载构造。
- 不能用于：直接估计真实编码代理的生产请求分布、真实用户交互比例，或将生成的 reasoning 当成人类思维过程。

### thoughtworks/agentic-coding-trajectories

- 来源：[Hugging Face 数据卡](https://huggingface.co/datasets/thoughtworks/agentic-coding-trajectories)。
- 规模：15,000 sessions、约 618,000 turns，由三个上游集合各 5,000 sessions 汇成。
- 许可：`derivative-multi-source`；逐条记录继承对应上游数据集许可，其中 Claude 生成部分还受 Anthropic 使用政策约束。
- 真实文本：含可读文本，但卡片说明轨迹文本由模型生成、工具响应为模拟内容，未执行真实工具调用；不是生产用户对话。
- 可用于：合成多轮请求、工具调用节奏、token/turn 负载和 KV-cache 压力基准。
- 不能用于：生产行为频率估计、真实工具环境结果、用户内容或真实代理质量评估。

### SWE-ZERO-12M

- 来源：[Hugging Face 数据卡](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories)。
- 规模：12,290,800 rollouts，覆盖 122,908 个 GitHub PR、3,222 个仓库和 16 种语言；数据卡称其为约 112B token 的执行自由轨迹集合。
- 许可：数据卡标注 Apache-2.0。
- 真实文本：包含公开 PR/任务上下文以及模型生成的多轮轨迹文本；数据卡说明轨迹由模型生成且为 execution-free，没有在容器中运行测试或验证补丁。不是人类代理的真实交互记录。
- 可用于：大规模合成轨迹的 turn/token 长度、动作序列、截断与失败状态分布；任务上下文与模型生成内容需分字段处理。
- 不能用于：将未验证轨迹当作成功修复，或将生成的 `thought` 字段用于语义内容统计。本项目按任务要求将该字段排除于内容统计之外；数据卡没有给出“由语法生成”的具体方法说明，因此不把该机制当作已由数据卡独立证实的事实。

### AgentX（SemiAnalysis / InferenceX）

- 来源：[AgentX 方法说明](https://inferencex.semianalysis.com/agentx/methodology)；[完整公开数据集卡](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)；[256k 变体数据卡](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k)。
- 规模：完整卡列出 393 sessions、98,827 requests、约 21.6B input tokens 与 106M output tokens；256k 变体列出 68,266 requests。
- 许可：公开数据卡标注 Apache-2.0。
- 真实文本：发布字段以时间、token 数量和共享前缀标识等负载元数据为主，不公开 prompt、源代码或工具参数/结果；回放器使用确定性合成文本/占位内容，不是可分析的真实请求文本。
- 可用于：请求长度与时间序列、prefix-cache 复用、合成回放和服务容量/吞吐测试。
- 不能用于：prompt/代码语义、用户意图、工具调用内容或真实 agent reasoning 的文本分析，也不能把合成回放文本当作原始 Copilot 会话内容。

---

## 2. 适用性评估（2026-10-10）

| 数据集 | 会话结构 | 前缀信息 | 可前向的文本 | 跨会话共享可测？ | 本项目用途 | 结论 |
|---|---|---|---|---|---|---|
| AgentX 完整版 | 真实（Claude Code 生产会话） | 有：每 64 token 一个 hash_id | 无（只能合成） | 否：`hash_id_scope=local` | 主结构数据；R_dedup；时间间隔 → 冷 KV | **继续使用**；结构分析从 50 个会话扩到全部 393 个 |
| AgentX 256k 变体 | 真实（截到 256K 以内） | 同上 | 无（只能合成） | 否 | 让按轮实验覆盖**完整会话**（v2 主要受上下文上限截断） | **新增**：作为按轮前向实验的首选会话来源 |
| TraceLab | 真实（生产使用，43–52 位开发者） | 待核实：只有 token 数，还是带缓存相关字段 | 无（语义文本已移除） | 待核实 | 第二个**真实生产**结构来源，与 AgentX 交叉验证轮次、上下文增长、时间间隔 | **采用（只做结构分析）**。重点核实是否有 cache read/write token 字段；如果有，可以直接得到生产环境的 prefix 命中率 |
| nebius/SWE-agent-trajectories | 生成（SWE-agent 轨迹） | 无，需从文本计算 | 有（环境观察真实，动作由模型生成） | **可以**：渲染成 token 后按内容哈希，system prompt 等跨轨迹共享 | 内容抽查（已用 8 条）；跨会话去重；短会话形态 | **继续使用**；按长度分层抽样，扩大样本 |
| thoughtworks/agentic-coding-trajectories | 生成（三个来源，约 41 轮/会话） | 无，需从文本计算 | 有（工具响应为模拟） | 可以（同上） | 第二个内容来源，覆盖不同 agent 风格；结构统计只代表“生成轨迹” | **采用（小样本）**。逐条记录的许可继承上游，按来源分开统计 |
| SWE-ZERO-12M | 生成（execution-free） | 无，需从文本计算 | 有 | 可以（同上） | 大规模轨迹长度分布；长上下文内容来源 | **暂缓 / 只抽小样本**：与 nebius 信息重叠，未经执行验证；只在需要长轨迹时抽取，`thought` 字段不参与内容统计 |

需要注意的两点：

- **生成轨迹的会话结构不能当作生产负载**（nebius、thoughtworks、SWE-ZERO 都是模型生成的）。它们可以提供内容、用于跨会话去重分析，但每轮间隔、轮数分布等结构统计只代表“生成轨迹”。真实生产结构只有 AgentX 和 TraceLab。
- 登记原文中 AgentX 的最后一句写的是“原始 Copilot 会话内容”。从本项目的 manifest 看，AgentX 的 `trace_model` 字段都是 `claude-*`，它是 Claude Code 的会话 trace。原文保持不变，以此处的说明为准。

---

## 3. 补充候选（需核实后再使用）

| 数据集 | 理由 | 待核实 |
|---|---|---|
| **Mooncake 公开 trace**（Moonshot / Kimi，FAST'25 论文配套，位于 `kvcache-ai/Mooncake` 仓库） | 真实生产请求的 hash_ids、时间戳和长度，含 conversation 与 tool-agent 两类负载；与师兄提到的 Mooncake offload 场景直接对应；如果 hash 跨请求全局有效，可以直接测**跨会话 / 跨用户去重** | 文件路径与字段名；hash 块大小（预期 512 token）；hash 是否全局有效；许可 |
| **普通多轮对话对照**：ShareGPT（`anon8231489123/ShareGPT_Vicuna_unfiltered`，vLLM benchmark 常用）或 WildChat | 回答“agentic 与普通对话在复用结构上差多少”；R_codec 预计不变，R_dedup 和每轮增量预计差异很大 | 许可（WildChat 需同意条款）；语言与清洗 |
| OpenHands 等**真实执行**的 agent 轨迹（例如 SWE-Gym 发布的轨迹集） | 工具输出来自真实执行，与 SWE-agent 不同的 agent 框架 | 具体数据集名称与 schema |
