# 当前状态

- 更新时间：2026-10-05（Asia/Singapore）
- 当前阶段：P2 agentic 按轮测量（P1 已完成）
- 当前周计划：`plans/weekly/2026-10-06.md`
- 状态：v2 N0–N4 全部完成（含 N1b/N2b/N3b 长上下文与多轨迹补跑），报告 v2 已生成，范围冻结，等待 2026-10-06 组会反馈

## 已完成事实

- 主线保持为 AgentX 推理持久状态的无损压缩；低比特权重压缩仍是待确认分支，未纳入本周执行。
- 主模型已切换为 `Qwen/Qwen3.5-9B`，固定 revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`；实验只使用当前 A6000 单卡。
- AgentX 数据固定为 `semianalysisai/cc-traces-weka-062126` revision `23f152f6f0f9399a85901b89a6458def0ef16729`，通过 AIPerf schema 解析并直接使用 trace token IDs。
- M0 已核实模型包含 8 个 full-attention 层和 24 个 linear-attention 层。真实持久状态包括：full-attention key/value（BF16）、convolution state（BF16）和 recurrent state（FP32）。代码未假设每层都有 K/V，也未把 FP32 状态强制转成 BF16。
- M1 已完成真实前向、raw-byte ANS 编码、完整元数据计费、全部块逐字节恢复、缓存重建和 continuation 一致性验证。
- M2 已完成 8K 单样本多 codec/块大小对照，以及另外两个独立 AgentX 请求的 2K raw-byte ANS 验证。
- M3 已生成可复查的聚合表、分层表、图和会议报告；实验范围现已冻结。

## v2 已完成事实（2026-10-04）

- 复盘确认 v1 的局限：每个会话只取 1 个请求的位置 0 前缀（3%–13%），没有多轮信息；三个会话的 K/V 压缩比几乎相同（1.2671/1.2673/1.2675）。
- 勘误（由 v1 的 blocks.csv 重算）：byte-lane K/V 的 payload-only 压缩比在 64 KiB 与 256 KiB 下为 1.4857 / 1.4862，总压缩比的差异来自元数据（概率表占比 4.4% vs 1.1%）；“完整持久状态”比值依赖长度；headline 应改用 byte-lane。
- 新增按轮实验代码：`src/agentic_inputs.py`（AgentX 会话逐轮合成、真实轨迹的 chat template 渲染、前缀链 key、hash 级 trace 统计）、`src/turn_runner.py`（增量 prefill、混合模型 snapshot/回滚、新块提取、并行编码、归档重建检查）、`src/kv_codec.py` 的 AKV2（紧凑 header、16-bit 块内表、共享表）、`scripts/analyze_traces.py`、`run_turns.py`、`summarize_turns.py`、`prepare_trajectories.py`，配置为 `configs/turns.yaml`。
- 测试：96 passed（CPU 容器，torch 2.11.0 / transformers 5.17.0 / constriction 0.5.0 / zstandard 0.25.0）；按轮流程用随机初始化的微型 Qwen3.5/Qwen3 验证（仅为测试夹具，不是结果）。

## v2 实验事实（2026-10-04 至 10-05，服务器实测）

- N0：AgentX 前 50 个会话、7,291 轮。整体前缀去重比 90.0×（单会话 p50 28.1×）；每轮前缀命中率 p50 99.5%、p10 97.0%；每轮新增 token p50 1,792、p90 4,416；会话最长输入 p50 215K token；相邻请求间隔 p50 17 秒。
- 模型审计：Qwen3-4B-Instruct-2507（revision `cdbee75f17c01a7cc42f958dc650907174af0554`）为 36 层全注意力 GQA，K/V BF16 `[1,8,T,128]`，每 token 147,456 bytes。
- 共享概率表：在留出的 trace 3（前 3 轮）上校准；表集合大小 Qwen3.5 98,304 B、Qwen3-4B 110,592 B。
- 最终结果集（14 个会话，`results/20261005_n4b_summary/`）：Qwen3.5-9B 与 Qwen3-4B 各 3 个 AgentX 会话 + 4 条 SWE-agent 真实轨迹（`nebius/SWE-agent-trajectories@68195a1450865274106246d0d0296a1d6807b88e`，前 4 个 ≥15 轮且 instance_id 不同的样本）。完整会话：AgentX t1（两模型）、Qwen3.5 t2、全部轨迹；其余按上下文上限截断（Qwen3.5 262K、Qwen3-4B 131K）。
- 主编码器 ans_lane_shared@64 的 BF16 K/V 压缩比：Qwen3.5 1.4824（会话间 1.4814–1.4827）、Qwen3-4B 1.4783（1.4774–1.4784）；order-0 熵上限 1.4851 / 1.4811；zstd 1.2693 / 1.2756；ans_raw 1.2690 / 1.2672。FP8-E4M3（事后转换，0 个元素截断）：1.2047 / 1.2027。
- 合成内容 vs 真实内容：Qwen3.5 1.4825 vs 1.4817；Qwen3-4B 1.4784 vs 1.4777。每轮压缩比在约 1.4818–1.4835 之间。
- 块内表 vs 共享表（Qwen3.5 BF16 K/V）：16 token 1.4170 vs 1.4806；64 token 1.4661 vs 1.4824；256 token 1.4787 vs 1.4828。共享表相对“每块最优表、表不计费”的损失：AgentX 0.07%–0.08%，SWE-agent 0.10%–0.13%。
- Qwen3.5 state 快照每份 49.5 MiB，压缩比 1.1776（recurrent 1.1702、conv 1.4737）；占存储 21%–29%（AgentX）/ 59%–70%（SWE-agent）；32%–45%（AgentX）与 84%–100%（SWE-agent）的轮次中快照大于该轮新增 K/V。
- 正确性：最终 14 个会话 16,590,152 个编码单元全部 bit-exact；13 个会话归档重建检查 ok，1 个跳过（Qwen3.5 t1，最后一轮无新块）；Qwen3.5 有 4 个会话因分叉后重算出现 16–45,260 个与原存块比特不同的块（续算 hidden 最大差 0.25），不是编码错误。
- 资源偏差：N1 与首轮 N2/N3 前的运行未设置 `CUDA_VISIBLE_DEVICES`，实际在 GPU 0 上运行（计划为 GPU 1）；N3 起及全部补跑在 GPU 1。最终结果集中只有 `20261005_n1_qwen35_t1` 来自 GPU 0 批次。
- 磁盘清理：8 个已通过重建检查且被取代或已核对的 run 的压缩归档已删除，`units.csv.gz` 保留（`logs/cleanup_20261005.log`）。
- v2 按轮实验共 22 次运行，GPU 墙钟约 2.3 小时。

## v1 关键结果（P1，保留）

- M0 256 tokens：完整持久状态 60,293,120 bytes；KV-only 8,388,608 bytes。
- M1 2K raw-byte ANS（256 KiB blocks）：KV-only 压缩比 1.267268；完整持久状态压缩比 1.172099；472/472 块逐字节一致。
- M2a 8K：byte-lane ANS（256 KiB blocks）KV-only 压缩比 1.464953、完整持久状态 1.404215；raw-byte ANS 分别为 1.267499、1.230233；zstd 分别为 1.267825、1.230731。
- M2 聚合（3 个不同请求/会话，raw-byte ANS，256 KiB blocks）：KV-only 压缩比 1.267395，节省 21.098%；完整持久状态压缩比 1.204734，节省 16.994%。
- M2 全部 19,328 个 block-codec 记录逐字节一致；缓存重建后 continuation hidden state 逐字节一致，最大差值为 0。

这些数值是实际编码结果，不是理论熵。当前样本属于 AgentX prefix-sampled 输入，尚不足以代表完整工作负载分布。

## 证据位置

- v2 报告：`reports/meeting_2026-10-06_v2.md`
- 报告引用数字：`results/20261005_n5_report_tables/`（由 `scripts/report_tables.py` 生成）
- 汇总表与图：`results/20261005_n4b_summary/`
- N0：`results/20261004_n0_trace_structure/`；校准：`results/20261004_n1_qwen35_calib/`、`results/20261005_n2_qwen3_4b_calib/`；审计：`results/20261005_n2_qwen3_4b_audit/`
- 按轮运行：`results/20261005_n1_qwen35_t1/`、`results/20261005_n1b_*`、`results/20261005_n2b_*`、`results/20261005_n3b_*`（被取代的短版运行仍保留：`20261004_n1_qwen35_t0_smoke`、`20261005_n1_qwen35_t{0,2}`、`20261005_n2_qwen3_4b_t*`、`20261005_n3_qwen3_4b_traj*`）
- 日志：`logs/`
- v1：M0 `results/20261004_m0_qwen35_256_retry1/`（首次下载中断记录 `results/20261004_m0_qwen35_256/failure.json`）、M1 `results/20261004_m1_qwen35_2k_ansraw/`、M2 `results/20261004_m2{a,b,c}_*`、M3 `results/20261004_m3_summary/`、报告 `reports/meeting_2026-10-06.md`

## 环境与验证

- 服务器：RTX A6000 48 GB，Python 3.11.16，PyTorch 2.11.0+cu128，Transformers 5.17.0，AIPerf 0.13.0，constriction 0.5.0，zstandard 0.25.0。
- 未安装 FLA/causal-conv1d；Transformers 参考实现在 262K（Qwen3.5）/131K（Qwen3-4B）内显存峰值 ≤34.3 GiB，可运行。
- 单元测试：96 passed。

## 下一步

- 2026-10-06 组会审阅报告 v2；会后归档本周计划并按反馈建立 2026-10-13 周计划。
- 报告第 6 节列出的候选方向（编解码吞吐与 offload break-even、冷 KV 量化、扩展模型、更强建模、原生 FP8 导出）需组会确认后才进入计划。

## 阻塞

- 无。
