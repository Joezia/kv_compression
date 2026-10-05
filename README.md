# Agentic Inference Cache Lossless Compression

本项目研究 agentic workload（多轮、工具调用、前缀高度复用）中持久推理缓存的无损压缩空间：full-attention K/V，以及混合架构实际保存的 recurrent/convolution state。核心粒度是**按轮**：每一轮请求在 prefix cache 之上新增了多少状态、这些新增状态能被无损压缩多少，并且去重收益与熵编码收益分开核算。ANS 是首先评估的编码方法，不是预设一定优于其他方案的结论。

当前状态：P1（单请求闭环）已完成；P2（agentic 按轮测量）首轮已完成：两个模型、6 个 AgentX 会话与 8 条 SWE-agent 轨迹，报告见 `reports/meeting_2026-10-06_v2.md`。具体进度见 STATE 和当前周计划。

## 两层计划的职责

本 README 是项目级路线：研究问题、阶段、阶段验收、统计口径和非目标，不承载每日待办。`plans/weekly/YYYY-MM-DD.md` 是以组会日期命名的周计划，周会后保留原计划并追加复盘。`STATE.md` 只记事实、最近结果、阻塞与下一步。`AGENTS.md` 是持续适用的执行规则。`CODEX_TASK.md` 是首轮启动任务书，已被本 README 与周计划取代的部分以后二者为准。

当前周计划：[2026-10-06](plans/weekly/2026-10-06.md)。

## 研究问题

Q1（P2）：在真实 agentic 会话结构下，每一轮新增的持久状态有多少？前缀去重（prefix cache）之后剩下的唯一状态，用无损编码还能再压多少？K/V 与混合模型的 state snapshot 各占多少存储？

Q2（P3）：Q1 的结论在多大范围内成立——不同模型架构（GQA / 混合线性注意力 / MLA / 滑窗）、不同缓存 dtype（BF16 / FP8）、不同内容来源（AgentX 合成内容 / 真实 agent 轨迹 / 普通多轮对话）？

Q3（P4）：编码器怎样设计才实用——概率表如何共享、编码单元按什么粒度切、建模上限在哪里、速度是否够？

Q4（P5，探索）：冷 prefix cache 有多少、跨用户能否联合压缩、在 DRAM offload / Mooncake 路径上压缩能否带来净收益？

Q1 是当前主线；Q4 目前只是方向性直觉，先不做实验，只在 P5 登记。

## 统计口径（所有阶段共用）

对一个会话的第 t 轮输入 x_t，设缓存块大小为 B tokens（默认 64，与 AgentX hash block 一致），b 为每 token 的 K/V 字节数：

```
Snapshot   = Σ_t full_blocks(x_t) · B · b       # 每轮都存全量（无 prefix cache）
Unique     = Σ_t new_blocks(x_t) · B · b        # prefix cache 实际需要存的新块
States     = Σ_{t: 有新块} state_snapshot_bytes  # 混合模型每轮一份，不可跨轮去重
Compressed = Σ 新块与 state snapshot 的 compressed_total_bytes

R_dedup = Snapshot / Unique       ← trace/会话结构属性，与 codec 无关
R_codec = Unique / Compressed     ← 这才叫“无损压缩率”，属于模型/dtype/codec 属性
```

- 块身份用前缀链 key：key_i = H(key_{i−1} ‖ block_i)。只有整个前缀相同的块才可复用，这与 KV 依赖完整前缀一致。只缓存完整块；每轮不足一块的尾部计数但不存。
- 总压缩比一律为 `Σraw / Σcompressed_total`。compressed_total 包含 payload、块内概率表、header；共享概率表单独计费一次（按模型的表集合大小），并给出 payload-only 比值作为“表完全共享”时的上限参照。
- 编码单元必须按 token 对齐：一个单元 = (层, K 或 V, 一段 token 范围, 全部 KV heads)，与缓存系统存取、驱逐、传输的单位一致；单元内部的字节顺序可以自由选择。对 order-0 ANS，单元内部的排列不影响结果，只有“哪些元素被分在同一单元/同一流”才有影响。
- K/V、recurrent、convolution 分别报告。“完整持久状态”的比值随上下文长度变化（固定大小的 state 与随长度增长的 K/V 混合），不能作为单一结论，必须同时给出长度与各部分字节数。
- headline 用最佳的**实用**配置（目前为按元素宽度分路、共享概率表的 ANS，64-token 单元），并与 raw-byte ANS、zstd 基线以及 order-0 经验熵（诊断用）并列。不以最弱 codec 的结果作为结论。
- 每个计入结果的单元都做 encode→decode 逐字节比对；每个会话结束时，从压缩归档重建最终前缀的 cache，并与 live cache 比对、续算一个 token 核对。

## 数据、模型与 dtype 矩阵

数据源（按优先级）：

| 数据 | 结构 | 内容 | 用途 |
|---|---|---|---|
| AgentX `semianalysisai/cc-traces-weka-062126` | 真实 Claude Code 会话：轮次、前缀关系（hash_ids）、时序、subagent | AIPerf 合成（模板代码窗口拼接，每 64 token 一个分隔 token） | 主数据：按轮结构、R_dedup |
| 真实 agent 轨迹（候选：SWE-agent / OpenHands 公开轨迹，名称与 schema 需在服务器上核实） | 真实多轮 | 真实代码/工具输出 | 内容代表性对照 |
| 普通多轮对话（候选：ShareGPT / LMSYS 类） | 多轮 | 真实对话 | “agentic vs chat”对照（P3 之后） |

AgentX 的 `hash_id_scope=local`，不能表达跨会话共享；跨用户实验需要其他数据（P5）。

模型按顺序逐个运行，每个新模型先做缓存结构审计（`run_turns.py audit`），确认层类型、shape、dtype、每 token 字节数之后，再跑按轮实验：

| 顺序 | 模型 | 架构/缓存 | 原生上下文 | 资源/备注 |
|---|---|---|---|---|
| 1 | Qwen/Qwen3.5-9B | 混合：8 层 GQA K/V（BF16）+ 24 层 Gated DeltaNet（FP32 recurrent + BF16 conv） | 长（已在 8K 实测） | 单卡可跑，已审计 |
| 2 | Qwen/Qwen3-4B-Instruct-2507 | 标准 dense GQA，全部层为 K/V | 256K（待审计） | 单卡；同家族对照，隔离架构因素 |
| 3 | meta-llama/Llama-3.1-8B-Instruct | 另一家族 dense GQA | 128K | 需要 HF gated 许可 |
| 4 | openai/gpt-oss-20b | 滑窗与全注意力交替 + attention sink | 128K | runner 需支持滑窗层；MXFP4 权重在 Ampere 上的加载需核实 |
| 5 | deepseek-ai/DeepSeek-V2-Lite-Chat | MLA：实际应缓存 latent（576 维/层/token） | 32K | HF 实现缓存的是展开后的 K/V，需加 hook 取 latent |
| 6 | Qwen3-Coder-30B-A3B 或同类 MoE agentic 模型 | MoE + GQA | 256K | 单张 A6000 放不下 BF16；需第二张卡或量化权重，先申请资源 |

dtype 与模型并不完全绑定：

- **模型决定的部分**：哪些状态存在、各自原生 dtype（例如 Qwen3.5 的 FP32 recurrent，MLA 的 latent）。这部分随模型矩阵自然覆盖。
- **部署决定的部分**：K/V 以 BF16 还是 FP8 存放（vLLM/SGLang 的 `kv_cache_dtype=fp8` 很常见）。每个模型都测 BF16 与 FP8 两种 K/V 格式。当前的 FP8 是带标注的事后转换（饱和 cast 到 E4M3，scale=1.0，不回灌模型）；P3 再用真实 serving 引擎导出原生 FP8 K/V 对照。

无损始终相对实际被编码的 buffer 定义；FP8 转换本身是有损的，单独标注，不和 BF16 结果混用。

## 项目阶段

### P1 — 单请求可信闭环（已完成，2026-10-04）

真实前向 → 持久状态提取 → ANS 编解码 → 逐字节恢复 → cache 重建续算，均已打通（`results/20261004_m*`）。复盘后修正的结论：块大小差异来自元数据而非数据；“完整状态压缩率”依赖长度；headline 不应使用 raw-byte ANS。见 `reports/meeting_2026-10-06.md` 的勘误。

### P2 — Agentic 按轮测量（首轮完成，2026-10-05）

在完整 AgentX 会话上按轮做增量 prefill，只存储和编码新块；混合模型每轮存储一份 state snapshot。输出每轮新增 token、R_dedup、各 codec 的 R_codec（K/V 与 state 分开）、BF16/FP8 两种格式、16/64/256-token 单元，以及共享概率表与块内概率表的对照。

验收：至少一个模型、两个以上完整（或按轮截断并标注）的会话，全部单元 bit-exact，归档重建检查通过；R_dedup 与 R_codec 分开报告，headline 使用最佳实用 codec；各项局限写清。

### P3 — 代表性

按上表顺序逐个扩展模型；加入真实内容轨迹对照与普通多轮对话对照；覆盖 subagent、更长的会话、原生 FP8 serving 导出。每一步只变一个因素，不一开始铺开笛卡尔积。

验收：说明 P2 结论在哪些条件下成立、在哪些条件下失效。

### P4 — 编码器与元数据设计

概率表共享方式（每模型离线校准 / 跨会话迁移代价 / 每块按需选择）、紧凑 header、按 token 对齐的单元粒度与服务引擎块大小（vLLM 16、AgentX 64、LMCache/Mooncake 256 级别）的关系；建模上限（按位拆分 sign/exponent/mantissa、按 head/channel 建模、上下文建模）；CPU/GPU 编解码吞吐。

验收：给出“压缩率 × 速度 × 元数据”三者权衡下的推荐设计及证据。

### P5 — 冷缓存、跨用户与系统路径（探索，暂不执行）

用 trace 时间戳和显式策略定义“冷”（例如超过 X 秒未访问的块），统计冷数据量与复用距离；跨用户方面分为完全相同块去重（需要全局 hash 的数据）与统计共享（共享概率表的跨用户迁移）；在 DRAM offload / Mooncake 路径上建立 `T_raw = S/B` 与 `T_cmp = T_enc + S/(R·B) + T_dec + T_extra` 的 break-even 分析。压缩放在哪一端必须明确：CPU 收到原始 KV 后再压缩只能省 DRAM，不能省已经发生的 GPU→CPU 传输。

### P6 — 原型与硬件（有证据再做）

只选一个 KV backend 和一条迁移/恢复路径做最小原型；依据实测瓶颈决定 GPU kernel、AMD 或硬件方向。

## 可复现与数据纪律

记录模型/tokenizer/data/harness/codec 的 revision、实际输入 token IDs 的指纹、会话与轮次身份、随机种子、环境、命令、原始结果与校验。保留错误、负结果和参数变更。每次运行使用新的 `results/<run_id>/`；大码流放在不提交 Git 的 `artifacts/raw/<run_id>/`。样本选择只依据结构（会话序号、轮数、长度），必须先于看到任何压缩结果。

## 复现入口

```bash
/home/zhouziang/.local/share/miniforge3/envs/cs5242-a4/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-pilot.txt
.venv/bin/python -m pytest -q

# P1（单请求闭环，已完成）
env CUDA_VISIBLE_DEVICES=1 HF_HOME="$PWD/models/hf" USE_HUB_KERNELS=NO .venv/bin/python scripts/run_pilot.py m1 --run-id <new_id>

# P2（按轮）：命令见 plans/weekly/2026-10-06.md 的 v2 执行节点；长上下文用 --config configs/turns_long.yaml
.venv/bin/python scripts/analyze_traces.py --run-id <id> --limit 50
env CUDA_VISIBLE_DEVICES=1 HF_HOME="$PWD/models/hf" USE_HUB_KERNELS=NO .venv/bin/python scripts/run_turns.py calibrate --model qwen35_9b --run-id <id>
env CUDA_VISIBLE_DEVICES=1 HF_HOME="$PWD/models/hf" USE_HUB_KERNELS=NO .venv/bin/python scripts/run_turns.py run --model qwen35_9b --trace-index 0 --tables results/<calib_id>/shared_tables.npz --run-id <id>
.venv/bin/python scripts/summarize_turns.py --run-id <id> <run_id> [<run_id> ...]
.venv/bin/python scripts/report_tables.py --run-id <id> --summary <summary_id> --trace-structure <n0_id>
```

固定配置：`configs/pilot.yaml`（P1）、`configs/turns.yaml` 与 `configs/turns_long.yaml`（P2）。

## 资料

[S1] AgentX methodology: https://inferencex.semianalysis.com/agentx
[S2] AgentX harness FAQ: https://github.com/SemiAnalysisAI/agentx-harness/blob/master/docs/benchmark-modes/semianalysis-agentx-faq.md
[S3] Qwen3.5-9B model/config: https://huggingface.co/Qwen/Qwen3.5-9B
[S4] AIPerf 0.13.0 `CodingContentGenerator`（hash-id 合成逻辑，已对照源码核实）

公开资料核对日期：2026-10-03/04。候选模型与数据集名称尚未在服务器上逐一核实，使用前以实际 revision 与审计结果为准。
