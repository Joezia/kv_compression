# 给 Codex 的任务书 v2：空仓库初始化、两层计划与 AgentX KV 压缩首轮实验

> **2026-10-03 修订说明**：本文件是启动任务书，不作为第三份动态计划。用户已确认以 README 和 active 周计划中的修订范围取代下列旧默认：不再使用 Qwen2.5-Coder-7B-Instruct，不再假设所有层都有 BF16 K/V，也不再统一使用 BF16 两字节通道和旧容量公式。本周主模型为 `Qwen/Qwen3.5-9B`；真实前向必须分别审计 full-attention K/V、recurrent state 与 convolution state，保留各自实际 dtype，并分别报告 KV-only 与完整持久状态。低比特权重压缩是待确认分支，不在本周执行。若本文件后文与 README 或 `plans/weekly/2026-10-06.md` 冲突，以后二者为准。

## 0. 空仓库初始化与两层计划

当前仓库是空的，尚未创建实验目录；如果已经复制了配套文档包，也只是规划文档，没有代码或实测结果。不要假设 AGENTS.md、README 或可复用 ANS 代码已经存在。先确认当前目录与已有文件；存在时阅读并保留，不存在时按下面的设计初始化，不因此阻塞。

首次只做轻量、只读的环境和资料核实，提出一次执行计划。Plan mode 中不要创建文件、安装依赖、修改代码或启动 GPU 作业。用户确认并切回执行模式后，先落盘必要文档，再实施 M0–M3；不要一直停留在计划模式，也不要每个小步骤反复请求确认。

### 文件职责

README.md：项目级计划，写研究问题、阶段 P1–P5、每阶段验收、代表性边界和已验证的复现入口。不把整个项目缩成周二前的 pilot，也不写成每天的待办。

plans/weekly/2026-10-06.md：第一周计划，写两天的 M0–M3、范围、资源预算、交付、验收和降级。之后每周用组会日期命名，保留历史并追加复盘。

plans/weekly/TEMPLATE.md：后续周计划模板。至少包含本周研究问题、所处项目阶段、范围与非目标、节点/依赖/产物/验收、预算、风险、最低交付、冻结节点和会后复盘。

STATE.md：仅记录已完成、正在进行、下一步、阻塞、实际命令和结果路径，并指向 active 周计划。最初写“实验未开始”，不能把准备文档写成已经验证硬件或 codec。

AGENTS.md：持续执行规则，要求每次读 README、STATE 和 active 周计划；遵守资源权限、结果真实性、位级无损和每节点回写规则。

CODEX_TASK.md：本启动任务书，保留详细实验方法。初始化后，README 是项目范围的权威位置，当周计划是本周排程的权威位置；任务书不作为不断复制进度的第三份计划。

.gitignore：忽略凭据、环境、权重和原始 KV；保留可审计的配置、摘要、报告和小结果表。

不要再创建与上述职责重叠的 PLAN.md 或 TODO.md。代码目录按需要逐个建立，不预生成大型空工程，不生成假数据或假图。结果按 results/<run_id>/ 隔离。

### 项目级路线（落盘至 README）

P1：真实 KV 采集 → ANS 编解码 → 元数据计费 → bit-exact → 初步压缩结果。验收看证据完整，不要求压缩率好看。

P2：检验 AgentX 合成内容偏差，补真实内容对照，逐项扩展代表性模型、dtype、上下文与布局；分析 layer/K/V/符号分布和编码粒度，明确适用边界。

P3：结合缓存策略和时序分析冷缓存与复用，再测 codec/链路成本并建 break-even。压缩比不等于系统加速；CPU 端在收到原始 KV 后压缩不能减少已经完成的 GPU→CPU 传输。

P4：只有机会成立后，在一个 backend 和一条迁移/恢复路径做最小原型；在等资源条件下测容量、流量、延迟及代价。

P5：依据瓶颈再决定 GPU/AMD kernel、融合或硬件方向。移植本身不替代研究问题，未获安排的未来阶段不自动实施。

这些阶段是长期研究路线，不是本周必须实现的清单。下一周节点由每次组会结果决定，不能预先保证每周都进入下一阶段。

## 1. 背景与本轮目标

我在 NUS Xtra Computing 实验室研究 KV cache 的无损压缩。师兄建议先研究 agentic workload 的 KV 是否有 ANS 压缩空间，之后才考虑 cold-prefix cache、DRAM offload、Mooncake，以及 GPU/AMD/硬件实现。

本次是这个研究方向的首轮 characterization，不是系统集成，也不是 AMD 移植。

今天是 2026-10-03（星期六）；下次组会是 2026-10-06（星期二）。按两个实际工作日规划：
- D1：2026-10-04，必须跑通真实模型 KV 的压缩/解压闭环。
- D2：2026-10-05，扩大少量样本、整理结果并形成组会报告。
- 2026-10-06 只做汇报与必要核查，不把主要实验留到当天。

本轮必须回答：
“在明确记录模型、KV dtype、AgentX 输入重建/采样方式的前提下，真实前向计算得到的 KV，用 ANS 能压缩多少？结果是否逐字节无损？K/V、layer、block size 有何初步差异？”

不能提前假定压缩率一定很好。低压缩率或压缩后膨胀，只要方法正确，同样是有效结果。

## 2. 资源边界与机器选择

当前工作环境在已获授权的 RTX A6000 48GB 服务器上。默认只使用该服务器上获准使用的一块 GPU，batch size = 1。

组内规则：
- 每人默认最多同时使用两块 GPU，但本实验默认一块，不因“看起来空闲”就占用第二块。
- 只能访问管理员明确授权的机器；禁止扫描其他服务器、擅自迁移或越权。
- 不杀其他用户进程，不改系统驱动/CUDA，不使用 sudo，不覆盖共享环境。
- 不输出或保存 token、密码、完整敏感环境变量。
- home 配额不是可随意用满的预算；大型数据使用本人有写权限且符合组内规定的目录。不要假设 /shared/ssd 一定存在。
- 重要代码、配置和结果摘要保留独立副本；禁止自动上传未经许可的数据。

只读检查当前 hostname、GPU 型号/空闲显存/占用情况、CPU 可用内存、磁盘可用空间与权限、Python/PyTorch/Transformers/CUDA 版本，以及当前用户明确提供的可复用模型/ANS 路径（没有时按从零接入规划）。不要递归扫描无关用户目录。

显式绑定选定 GPU。没有获准可用的 GPU 时，继续 CPU 侧的数据审计和 codec 单元测试，并报告阻塞；不要抢卡。

依赖优先沿用已有可用环境；确有需要再建立隔离的项目环境并固定版本，不盲目升级整套 PyTorch/CUDA。

## 3. 默认实验配置与非目标

首选模型：Qwen/Qwen2.5-Coder-7B-Instruct。
该模型属于 2024 年发布的系列，只作为两天内跑通 pipeline 的 coding/GQA 基线，不代表当前前沿 agent 模型。[S9]
师兄的聊天记录没有指定模型；不要把本默认选择写成师兄推荐。更新模型的代表性验证属于 P2，例如先审计 Qwen3-Coder-30B-A3B-Instruct 等候选的资源和实际 cache 结构，而不是在首轮自动切换模型。[S10]
权重 dtype：BF16。
KV dtype：运行时实际缓存的 BF16，必须验证，不能仅根据模型名称推断。
推理框架：优先 Hugging Face Transformers 直接前向，便于读取真实 cache。
批大小：1。
先用约 2K tokens 做 smoke test；主体样本优先 8K–16K 内，32K 仅作为实测可行后的补充。
模型、tokenizer、数据、harness 和 codec 全部记录 revision/commit 或版本。

该模型配置可作为容量检查的参照：28 层、4 个 KV heads、head_dim = 128；batch=1、BF16 时，K+V 合计为 56 KiB/token，因此 16,384 tokens 的 KV 约为 896 MiB。重新读取实际加载模型的 config 并核对，不能对其他模型套用该数值。[S4]

若模型下载受限或已有兼容的 7B/8B 模型更容易使用，先在计划中说明替代模型及 KV 容量，再确定；不要默默切换模型。

本轮不做：
- 完整 AgentX serving benchmark、官方榜单提交或满并发回放。
- 70B/MoE、大规模多模型或大规模多 dtype 扫描。
- FP8/INT4 KV、KV quantization、权重量化、RoPE/YaRN 参数改动。
- vLLM/SGLang 内核改造、Mooncake/LMCache 集成。
- 新的 GPU ANS kernel、AMD port、硬件设计。
- 为了展示更漂亮的结果而混入随机 KV、零填充或重复副本。

## 4. AgentX 数据与输入构造：先确认它到底是什么

起点是 AgentX 官方方法说明、官方 harness 和固定数据集：
- semianalysisai/cc-traces-weka-062126。
- SemiAnalysisAI/agentx-harness。
资料地址在本文末尾；实际使用时固定 revision，核实代码与文档是否一致。[S1–S3]

AgentX 不绑定某个推理模型：原始 trace 可含 Claude 模型标签，但回放可通过 --model 指定本地目标模型；我们在离线采集时同样明确指定模型和 tokenizer。[S2]

关键事实：公开 AgentX trace 没有原始 prompt/code/tool payload，也没有 KV；它提供请求长度、cache-block identities、时序和会话结构。官方通过确定性合成内容重建输入。因此我们采集的是“指定开源模型对 AgentX 合成输入计算出的 KV”，不是原始 Claude 会话的 KV。[S1]

先做 CPU 数据审计，确认真实 schema，而不是猜字段。核对 conversation/request/branch 标识、input/output token counts、hash_ids、block_size 和 hash_id_scope。尤其注意 local-scope 的 hash ID：不同会话中的相同数字不代表同一块。[S3]

优先复用官方 harness 的输入重建逻辑，获得适配目标 tokenizer 的完整输入 token IDs 或最终渲染后的输入。可以做薄适配层，不要求启动完整服务端，也不要求跑一小时官方压测。
不要直接假设 pip 上任意版本 aiperf 就包含目标 fork 的全部行为。
记录最终实际输入 token 数、chat template、special tokens、seed；trace 中的长度与重新 tokenize 后的长度可能不同。

采样目标不是硬性承诺：
- 先找到 1 个会话中至少 2–3 个相关请求，完成闭环。
- 扩展目标为 4–6 个不同会话、每个约 3–5 个相关轮次，以可获得的独立数据和资源预算为准。
- 优先选择真实可关联的主干多轮请求，不为凑短输入而只选短小辅助调用。
- 先按长度、prefix growth 和会话结构选择，不能先看压缩率再挑样本。
- 主体待压缩的唯一 KV 数据量控制在约 2–6 GiB，原始中间数据默认硬上限 10 GiB；超过前先缩减采样，不自动扩盘或删旧文件。

必须区分三种数据身份：
A. AgentX-native-subset：保留所选请求的完整重建输入，仅选择原生能容纳的短请求/会话片段。
B. AgentX-prefix-sampled：完整重建后，仅保留从位置 0 开始的前 N 个 tokens；明确记录原长度、实际长度和采样比例。这不是完整 AgentX replay。
C. AgentX-derived-proxy：官方重建接口无法及时接通时，根据 trace 结构自行构造的代理输入。必须独立标注，不能冒充 A/B。

优先 A；短请求不足时可用 B，不要无限等待。B 的限制必须醒目：如果每轮都截相同的前 16K，可能根本没有新增 prefix，不能把这些副本当成多轮新样本或用来画虚假的 turn 趋势。
不从长序列中截取中间片段、重新从 position 0 前向，再声称得到原位置 KV。
不通过重复一句话或重复短代码填满长 context；这会引入输入构造偏差。
C 只能作为保底管线验证，报告标题和结论都必须标出 proxy；继续记录真正 AgentX 接入的阻塞。

这是离线 KV characterization，可以不等待 inter-turn delay、不执行真实工具，也不重放并发 DAG；但保留来源和分支标签，不能声称测量了真实调度、冷热状态或 prefix hit rate。

## 5. 如何得到真实 KV

使用 model.eval() 和 inference_mode，不训练，不开梯度。
优先使用当前版本支持的 memory-efficient SDPA/Flash attention 路径；确认实际后端，避免长序列静默退化到巨大的 attention matrix。[S6]
不返回所有 attention weights 或 hidden states。

本轮主要采集请求 prefill 结束时的 prefix KV。完整输入中的历史 assistant/tool 内容也参与模型真实前向；不需要为每个请求额外生成几百个 tokens。
官方重建的历史 assistant 内容不等于本地模型刚生成的输出，不能自行替换后还声称保持了 trace 的原有前缀关系。[S2]

可直接调用底层 decoder 得到 cache，或使用当前版本支持的方式限制 logits 计算，避免为整个长序列计算和保存词表 logits。不要猜模型不存在的参数。

读取实际 past_key_values/Cache；根据本地 Transformers 版本适配接口，不硬编码未经检查的旧 API。[S5]
应采集 attention 真正保存并将复用的 K/V，而不是 hidden states、未经缓存处理的 K projection，或 GQA repeat 后人为复制出的 query-head 数量。
核对层数、KV head 数、seq_len、dtype、shape、有效 token 数和实际字节数。排除 padding、未使用的静态 cache 空间和 allocator 空洞。
对该默认模型，所有层、全部 K/V 都应有覆盖，不只选“好压缩”的层。

优先采用分块 prefill 或逐段增长来降低峰值内存；例如先尝试 512–1024 个新 tokens 一段。attention mask、position/cache position 和历史长度必须正确。[S5]
若 chunking 接口不能迅速验证，先用更短的整段前向跑通，不强行追求长 context。
OOM 时按顺序排查 logits/attention 暂存、chunk size、重复 GPU 副本，再缩短输入；不要第一反应切 dtype、量化权重或申请更多 GPU。

对相邻请求，计算最终实际 token IDs 的 longest common prefix，而不是用 token 长度差猜新增量。
- 只有真正兼容的前缀才允许复用 cache。
- 请求发生分叉或改写时，正确回退/重建，不能盲目 append。
- 以 turn 组织记录，但把旧 prefix 和本轮新增 KV 区分开。
- 同一个实际保留的旧 prefix 优先保存一次，后续通过引用建立快照。
- 全量快照统计与唯一块统计分开，不能把重复前缀压缩当作新的 entropy coding 收益。

复制到 CPU、重排或保存时保持原始位模式，禁止为了兼容 numpy 而把 BF16 数值转为 FP32 再压缩。允许可逆的字节重排，但必须记录并在解码时逆变换。
主压缩对象是实际 tensor 的有效原始字节，不是 pickle/.pt 文件的序列化开销。

## 6. 压缩器：先获得可信的 ANS 实测

仓库初始没有 ANS 代码；只有当前用户已明确提供或允许复用的本地实现才优先采用，不扫描无关目录查找；如果 GPU 实现需要较多编译/迁移，第一轮使用成熟 CPU ANS 实现，例如 constriction 的 AnsCoder。[S7]
GPU 负责产生 KV，CPU 可以负责离线压缩率测量。CPU codec 速度不能冒充 GPU codec 吞吐或在线 offload 性能。
不要从零写一个逐符号 Python ANS 当主数据处理路径。

至少比较以下配置：
1. 原始未压缩字节，作为容量基线。
2. 对原始 byte stream 使用 ANS。
3. BF16 按每个值的两个字节位置拆成两路，分别建模并做 ANS；字节序必须明确。
4. zstd，作为通用压缩器对照；记录版本和压缩级别。

配置 3 的关键是“两路独立概率模型”，不是单纯把字节顺序换一下。零阶 ANS 如果继续使用同一个 histogram，重排不会改变符号计数。
所有变换都必须完全可逆；不做截位、数值舍入、近似 delta、丢弃低位或其他有损操作。
不要把 BF16 的高字节简单称为“完整 exponent”。

先分别对每层 K 和 V 的连续有效数据做分块。记录原 tensor layout 和编码 layout；这是离线逻辑分块，不是已经实现了某个 serving engine 的物理 paged-cache layout。
默认块大小先测 64 KiB 与 256 KiB；有余力再加 1 MiB。单位是字节，不是 tokens。
不同 block-size 配置尽量使用相同的源数据；尾块有效长度单独记录。
这轮不做跨用户联合编码，不把很多重复快照拼成一个大文件压缩。

压缩大小必须可审计：
compressed_total_bytes =
payload_bytes + probability_table_bytes + per_block_header_bytes + required_padding_bytes。
必须包括解码必需的表、长度、分流信息、ANS 状态/对齐等，明确哪些公共元数据对所有方案同等排除。
例如库返回 uint32 数组时，记录实际 nbytes，而不是数组元素个数或理想有效 bit 数。[S7]
如使用数据自适应 histogram，解码所需统计表必须存入压缩包或以明确计费的共享方式提供；不能从原始数据偷偷重建。
检验解码器仅依靠压缩包及已声明的外部固定配置就能恢复，不依赖编码过程残留的未计费状态。
不要把 entropy estimate 当成真正 ANS bitstream 的结果。
压缩膨胀必须如实报告；若使用 raw fallback，单独报告启用前后的大小和 fallback 比例。

正确性要求：
- 对每个计入结果的 block，完成压缩、解压、逆变换，并做完整 uint8/bytes equality 检查。
- 100% 通过才计入有效结果；不能用 allclose 代替 bit-exact equality。
- 测试空/短/尾块、零字节、随机字节，以及包含特殊浮点位模式的字节输入。
- 对实际 KV 留存 checksum、原始长度和恢复长度。
- 数值计算在不同 attention backend/chunking 下的可能舍入差异，不等于 codec 允许有误差；codec 总是相对同一原始 buffer 逐位无损。

## 7. 输出、统计与验收

最少保存：
- configs/pilot.yaml：全部实验参数、资源限制和 seed。
- results/<run_id>/env_summary.json：非敏感环境、GPU、依赖版本。
- results/<run_id>/manifest.jsonl：可追溯的 trace/request/branch、长度、prefix/suffix、采样身份和数据引用。
- results/<run_id>/blocks.csv：块级实际数值。
- results/<run_id>/summary.csv：可从同一 run 的 blocks.csv 重算的汇总。
- results/<run_id>/figures/：三类必要图。
- reports/meeting_2026-10-06.md：组会报告。
- README.md：项目级路线，以及闭环跑通后追加的真实环境与复现命令。
- plans/weekly/2026-10-06.md：节点状态、证据路径与会后复盘。
- STATE.md、测试与运行日志；不要额外维护根目录 PLAN.md。

manifest / blocks 至少能关联：
run_id、model/tokenizer revision、trace_id、request/turn_id、branch、
source_kind、original/rendered/used input length、prefix length、
layer、K/V、dtype、layout、token span、block size、有效 raw bytes、
codec、payload/header/table/total bytes、bit_exact、checksum。

总体压缩比：
R = sum(raw_bytes) / sum(compressed_total_bytes)。
不要直接平均每块的 ratio。
同时报告空间节省比例 1 - 1/R、raw/compressed 总量、样本数、独立会话数、唯一 KV 字节数。
全量快照、唯一 prefix 块、以及新增 suffix 块的口径不能混为一谈。
不同会话/层的分布可以展示，但不要把同一会话的大量 blocks 当成大量独立工作负载。

三类必要图：
- 不同 codec 的总体 K/V 压缩比，包含元数据。
- 随 layer 变化的 K/V 压缩比。
- block size 对压缩比的影响。
图标题必须包含模型、KV dtype 和数据身份，尤其 sampled/proxy 标签。
只有真实存在足够不同轮次/上下文时再画 turn/context 趋势；否则明确缺失。
可以增加 byte-position entropy 解释，但只能称为所选零阶模型的经验分析，不能当作通用压缩极限。

codec 耗时为次要项：若测量，注明 CPU/GPU、线程数、计时范围、重复次数，区分编码、解码、数据复制、磁盘 I/O。没有做在线路径就不报告 TTFT 改善或系统加速。

报告结构：
研究问题；实验配置与来源；采样/变换；bit-exact 证据；结果；局限；下一周动作。
所有图表数字来自实际结果文件。禁止生成“示意结果”充当实测。
明确不能据此证明：原始 Claude KV 的压缩率、真实内容的一般性、完整 AgentX 性能、冷缓存比例、跨用户收益或生产 offload 加速。
agentic 场景是否值得压缩，不要求其每字节一定比普通 workload 更易压；长期保留和迁移需求是另一个独立问题。

## 8. 首周计划的初始化内容（落盘至 plans/weekly/2026-10-06.md）

M0 — 数据/环境可行性（D1 前半段）：
实际环境信息、确定的单卡与模型、trace schema 和输入重建方式、样本名单、内存/磁盘预算。
验收不是“读了文档”，而是能展示一个真实 trace request 到最终模型输入的映射。

M1 — 首个闭环（D1 结束前）：
至少一个真实模型请求的所有层 K/V，至少一个真正 ANS 配置，实际压缩后大小和逐字节恢复证据。
优先补齐一个会话的相关轮次；先保证闭环，不追求样本量。

M2 — 小样本结果（D2 前半段）：
扩展不同会话，完成 K/V、layer、两个 block sizes 及主要 codec 对照，保留完整 manifest。
样本不足或某个对照未完成时如实注明，不推迟所有结果整理。

M3 — 冻结并汇报（D2 后半段）：
停止增加新依赖/新模型/新功能，补测试、重算 summary、出图并写报告。
报告中给出真实范围、负结果和明确的下一步，不为了赶会而隐藏失败。

降级顺序：
先少做会话/轮次，再缩小上下文或采样字节数，再去掉第三种 block size/次要图。
保留“真实 KV + 实测 ANS + 解码所需元数据计费 + bit-exact”这个核心。
官方合成接口没接通就明确降为 proxy；只有 zstd 跑通就明确写“ANS 尚未完成”，不能把它改名为 ANS。
若只有 D1 的一个完整会话，也交付可信闭环和结果，而不是一个未运行的大框架。

## 9. 后续周节点的候选方向（不是承诺，不自动扩展）

2026-10-13：补真实文本/代码/工具轨迹的内容对照，检查 AgentX 合成输入偏差；按首轮结果决定第二个模型或 dtype。
2026-10-20：测实际 codec 吞吐与目标传输路径，建立压缩成本和传输/容量收益的 break-even 分析。
之后：只有证据支持时，再做一个 cold-KV/offload 最小原型，进一步评估 GPU/AMD/硬件需求。

不要因为这份后续路线而在两天内提前实现系统。

## 10. 首次回复必须给出的内容

请先给出：
对空仓库文档结构及两层计划职责的确认（Plan mode 中先展示，不创建）；
README 中 P1–P5 的项目路线与首周计划对应关系；
当前已确认的环境事实及仍未知事项；
一个明确的单卡默认方案；
M0–M3 的具体工作与验收；
最可能的两个阻塞及降级路径。
只问真正无法通过当前仓库、只读检查和资料核实解决的权限/资源问题。
等待一次计划确认后开始执行。

## 参考资料

以下用于核实接口与事实，不表示这些页面当前版本已经与本地环境匹配。

[S1] AgentX 官方方法说明。
[S2] 官方 harness 与其 AgentX FAQ。
[S3] 固定日期的官方数据集。
[S4] 默认模型说明与配置；默认 config 为 32,768 context，超出需要另外处理，不属于本轮默认范围。
[S5] Transformers cache 文档；按安装版本核对。
[S6] PyTorch SDPA 文档；按安装版本核对。
[S7] constriction ANS 文档。
[S8] Codex 官方 Plan mode 命令说明；按本地客户端版本核对。
[S9] Qwen2.5-Coder 官方首发说明，2024-09-19。
[S10] 后续模型候选的官方说明；不是首轮实施要求。

```text
S1 https://inferencex.semianalysis.com/agentx
S2 https://github.com/SemiAnalysisAI/agentx-harness
S2 https://github.com/SemiAnalysisAI/agentx-harness/blob/master/docs/benchmark-modes/semianalysis-agentx-faq.md
S3 https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126
S4 https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
S4 https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/main/config.json
S5 https://huggingface.co/docs/transformers/en/cache_explanation
S6 https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
S7 https://bamler-lab.github.io/constriction/apidoc/python/stream/stack.html
S8 https://developers.openai.com/codex/cli/slash-commands
S9 https://qwenlm.github.io/blog/qwen2.5-coder/
S10 https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
```
