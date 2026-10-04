# Agentic Inference Cache Lossless Compression

本项目研究 agentic workload 中持久推理缓存的无损压缩空间，包括 full-attention K/V 以及混合架构实际保存的 recurrent/convolution state；在有实验依据时，再进一步研究冷 prefix cache 的存储、迁移和恢复。ANS 是首先评估的编码方法，不是预设一定优于其他方案的结论。

当前状态：P1 首轮实验已在授权的 A6000 单卡完成并冻结。真实缓存结构、bit-exact 闭环、小样本 codec 对照及组会报告均有可审计证据；具体进度与局限见 STATE、当前周计划和 `reports/meeting_2026-10-06.md`。

## 两层计划的职责

本 README 是项目级路线：研究问题、阶段、阶段验收和非目标。它不承载每天的待办清单。

`plans/weekly/YYYY-MM-DD.md` 是以组会日期命名的周计划：本周范围、执行节点、产物、验收和降级策略。每周滚动制定；周会后保留原计划并追加复盘，不覆盖历史承诺。下一周从 `plans/weekly/TEMPLATE.md` 创建。

`STATE.md` 只记录当前事实、最近结果、阻塞与下一步，并指向当前周计划，不构成第三份计划。`AGENTS.md` 是持续适用的执行规则。`CODEX_TASK.md` 是本次初始化与首轮实验的技术任务书；初始化后，项目范围以 README 为准，本周排程以当周计划为准，二者变化不需要反复改写启动任务书。

当前周计划：[2026-10-06](plans/weekly/2026-10-06.md)。

## 研究问题

我们首先要知道：真实模型产生并保存的持久推理状态，在固定模型、dtype、输入构造和编码粒度后，有多少可复现的无损压缩空间？空间来自符号分布、字节位置、layer、state kind、数据布局，还是输入构造与重复快照造成的偏差？KV-only 与完整持久状态必须分别报告。

随后要知道：这种空间能否跨真实内容、模型和缓存表示保持？哪些 KV 会实际进入冷缓存并再次使用？压缩容量收益与传输收益能否超过编码、解码、数据搬运、调度和随机访问开销？

这两个问题不能混为一谈。agentic KV 不一定需要“每字节比普通对话更好压”才有系统价值；缓存驻留、复用和迁移需求也可能形成机会。但 prefix 重复、跨快照去重与熵编码收益必须分别核算。

## 数据与模型定位

AgentX 是面向 agentic serving 的轨迹数据、重建/回放方法及评测工具，不是一个模型。公开 trace 保留长度、前缀关系、分支和时序，移除了原始 payload；回放使用合成内容，目标模型由执行者指定。原始 trace 的 Claude 模型标签不等于本实验运行的模型。[S1][S2]

因此本项目的第一批结果必须称为“所选模型在 AgentX 重建输入上的 KV 压缩结果”，不能称为原始 Claude KV 或真实业务内容的压缩率。师兄提供的聊天记录要求收集代表性 model、datatype、dataset，并未指定某个模型。

首轮主模型为 `Qwen/Qwen3.5-9B`，固定 revision 后在当前 A6000 单卡上运行。官方配置含 32 个文本层，其中 8 层为 full attention、24 层为 linear attention；前者保存逐 token K/V，后者保存 recurrent 与 convolution state。配置只用于 M0 前的容量估算，实际层类型、shape、dtype 与字节数必须从真实前向得到的缓存逐项核实，不能假设所有层都有 K/V，也不能把 FP32 状态转换成 BF16。[S3]

正式模型矩阵应分别考虑 agentic coding 使用场景、持久状态结构、实际 dtype、上下文长度和资源要求，而不能仅按发布日期排序。本周 Qwen3.5-9B 是单模型主实验，不构成最终代表性模型选型，也不自动扩展成多模型扫描。

会议提出的低比特权重压缩是独立的待确认分支。只有研究问题、模型范围、精度约束、基线与资源获得确认后才进入项目计划；它不属于当前 P1，不触发本周 20+ 模型扫描、权重量化实验或 AMD 优化任务。

## 项目阶段

### P1 — 建立可信闭环，获得初步压缩结果

完成 AgentX 输入来源审计、真实模型前向、有效持久状态提取、实际 ANS 编解码、元数据计费与逐字节恢复验证。覆盖全部实际缓存层和 state kind，并用少量块大小获得可重算结果；KV-only 与完整持久状态分别汇总。

验收：至少一个真实请求的完整闭环可复现，所有进入有效结果的块都 bit-exact，压缩字节数可审计，输入身份和样本范围清楚。不能把“压缩比必须超过某值”设为验收条件；膨胀也是有效测量。

### P2 — 代表性与机制分析

补真实文本/代码/工具内容对照，检查合成输入偏差。逐项扩展模型、真实缓存 dtype、context、layout 与 block size，不一开始铺开笛卡尔积。区分原生低精度 cache 与对已采集状态的事后转换；无损始终相对明确的输入表示定义。

分析 layer、state kind、字节位置与经验熵；拆分独立编码、共享概率模型和联合编码收益。只有测量目标状态后才能对不同架构推广；经验熵只作诊断，不能写成实测 ANS 压缩率。

验收：说明首轮结论在哪些条件下成立、在哪些条件下失效，并有足够来源证据支持下一步场景选择。若代表性条件下没有值得利用的空间，形成负结果或调整方向，而非强行推进实现。

### P3 — 冷缓存与系统机会

结合 trace 时序、前缀关系和显式缓存策略分析复用距离、驻留、迁移量及潜在容量压力。没有策略或系统实测时，不把“旧 prefix”自动标为冷数据，也不把理论复用比例叫做实际 cache hit rate。

测选定实现的编码/解码吞吐、目标路径有效带宽与额外复制开销。压缩放在哪一端必须明确：只有在瓶颈链路发送前已经压缩，才会减少该次链路字节数。在 CPU 收到原始 KV 后再压缩，可以节省 DRAM，但不能声称节省了刚刚完成的 GPU→CPU 传输。

串行初步模型为 `T_raw = S/B`、`T_cmp = T_enc + S/(R*B) + T_dec + T_extra`；这是明确无重叠假设下的模型，不能替代有流水并行、排队和 GPU 争用的系统测量。容量收益、首次 offload 延迟和再次恢复延迟分开报告。

验收：至少一个目标路径的 break-even 区域及不确定性清楚，收益与成本使用同一计量口径。硬件加速是否必要在此之后判断。

### P4 — 最小系统原型

在 P2/P3 支持的场景中，只选一个 KV backend 和一条迁移/恢复路径；实现冷 KV 压缩、索引和恢复。关注块粒度、额外元数据、随机取回、尾延迟和同其他推理任务的资源争用。

验收：与等资源、等 workload、等缓存策略的未压缩基线对照，至少证明一项真实系统收益，并完整报告其他指标的代价。未测的 TTFT、吞吐或命中率不得宣称改善。

### P5 — 有证据再做性能实现或硬件研究

依据实测瓶颈决定 CPU/GPU kernel、AMD 移植、融合、批处理或硬件加速方向。不得因为存在可移植代码，就把移植本身当作本项目研究问题。

验收：优化针对已识别瓶颈，有正确性与端到端收益证据。是否进入本阶段，由实验和导师/师兄讨论决定。

## 周会节奏

首个会议节点是 2026-10-06，实际实验窗口为 2026-10-04 与 2026-10-05。先完成 P1，而非在两天内覆盖 P1–P5。

2026-10-13 可优先做内容代表性验证和新模型选型审计；2026-10-20 可根据已有证据转向 codec/传输分析。以上仅是候选方向，不是固定承诺。每次会后，以本周实测结果生成下一周计划，再将其设为 active。

## 可复现与数据纪律

压缩比统一使用 `sum(raw_bytes) / sum(compressed_total_bytes)`；解码需要的概率表、头部和对齐必须计费。存储有效 tensor 字节而非序列化包大小；任何 dtype 都不作数值转换后冒充原表示。全部有效块逐字节往返验证。

记录模型/tokenizer/data/harness/codec revision、实际输入 token IDs 或可复现引用、采样身份、随机种子、实际环境、运行命令、原始结果与校验。全量快照、唯一前缀和新增 suffix 分开统计。保留错误、负结果和参数变更；禁止凭示意图编造数据。

大模型和 KV dumps 不提交 Git；元数据、配置、小规模结果表与报告保留在版本管理中。原始数据路径按实验室授权和配额选择，不能预设 `/shared/ssd` 可用。

## 仓库与复现入口

代码与结果只按当前节点建立。每个实验使用独立 `results/<run_id>/`，大码流位于忽略提交的 `artifacts/raw/<run_id>/`；周计划和报告引用具体 run，历史结果不覆盖。

已验证的最小复现入口如下。现有 Torch/Transformers 来自授权环境；项目专用依赖安装在 `.venv`。每次实验必须使用新的 run_id。

```bash
/home/zhouziang/.local/share/miniforge3/envs/cs5242-a4/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-pilot.txt
env HF_HOME="$PWD/models/hf" USE_HUB_KERNELS=NO .venv/bin/python scripts/run_pilot.py prepare
env CUDA_VISIBLE_DEVICES=1 HF_HOME="$PWD/models/hf" USE_HUB_KERNELS=NO .venv/bin/python scripts/run_pilot.py m0 --run-id <new_m0_run_id>
env CUDA_VISIBLE_DEVICES=1 HF_HOME="$PWD/models/hf" USE_HUB_KERNELS=NO .venv/bin/python scripts/run_pilot.py m1 --run-id <new_m1_run_id>
.venv/bin/python -m pytest -q
```

固定配置见 `configs/pilot.yaml`。本周报告：[meeting_2026-10-06](reports/meeting_2026-10-06.md)。

## 资料

[S1] AgentX methodology: https://inferencex.semianalysis.com/agentx
[S2] AgentX harness FAQ: https://github.com/SemiAnalysisAI/agentx-harness/blob/master/docs/benchmark-modes/semianalysis-agentx-faq.md
[S3] Qwen3.5-9B model/config: https://huggingface.co/Qwen/Qwen3.5-9B

公开资料核对日期：2026-10-03。执行实验时还需固定实际使用 revision；日期核对不等于服务器环境已验证。
