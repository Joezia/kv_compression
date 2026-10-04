# 当前状态

- 更新时间：2026-10-04（Asia/Singapore）
- 当前阶段：P2 agentic 按轮测量（P1 已完成）
- 当前周计划：`plans/weekly/2026-10-06.md`
- 状态：v1（M0–M3）完成；2026-10-04 用户复盘后改为 v2 按轮测量，代码已就绪，N0–N4 尚未在服务器上运行

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
- 测试：94 passed（CPU 容器，torch 2.11.0 / transformers 5.17.0 / constriction 0.5.0 / zstandard 0.25.0）；其中按轮流程用随机初始化的微型 Qwen3.5/Qwen3 验证（仅为测试夹具，不是结果）。真实模型与数据尚未运行。

## v1 关键结果（P1，保留）

- M0 256 tokens：完整持久状态 60,293,120 bytes；KV-only 8,388,608 bytes。
- M1 2K raw-byte ANS（256 KiB blocks）：KV-only 压缩比 1.267268；完整持久状态压缩比 1.172099；472/472 块逐字节一致。
- M2a 8K：byte-lane ANS（256 KiB blocks）KV-only 压缩比 1.464953、完整持久状态 1.404215；raw-byte ANS 分别为 1.267499、1.230233；zstd 分别为 1.267825、1.230731。
- M2 聚合（3 个不同请求/会话，raw-byte ANS，256 KiB blocks）：KV-only 压缩比 1.267395，节省 21.098%；完整持久状态压缩比 1.204734，节省 16.994%。
- M2 全部 19,328 个 block-codec 记录逐字节一致；缓存重建后 continuation hidden state 逐字节一致，最大差值为 0。

这些数值是实际编码结果，不是理论熵。当前样本属于 AgentX prefix-sampled 输入，尚不足以代表完整工作负载分布。

## 证据位置

- M0：`results/20261004_m0_qwen35_256_retry1/`
- M1：`results/20261004_m1_qwen35_2k_ansraw/`
- M2a：`results/20261004_m2a_qwen35_8k_allcodecs/`
- M2b：`results/20261004_m2b_qwen35_trace1_2k_ansraw/`
- M2c：`results/20261004_m2c_qwen35_trace2_2k_ansraw/`
- M3：`results/20261004_m3_summary/`
- 会议报告：`reports/meeting_2026-10-06.md`
- 首次 M0 下载中断记录：`results/20261004_m0_qwen35_256/failure.json`

## 环境与验证

- Python 3.11.16
- PyTorch 2.11.0+cu128
- Transformers 5.17.0
- AIPerf 0.13.0
- constriction 0.5.0
- zstandard 0.25.0
- 单元测试：37 passed
- 未安装 FLA/causal-conv1d；Transformers fallback 在本轮资源预算内可运行，因此未追加依赖。

## 下一步

- 在服务器上按周计划 v2 执行 N0 → N1 → N2（→ N3 stretch）→ N4；命令见 `plans/weekly/2026-10-06.md`。
- N1 先用 `--max-turns 5` 冒烟，确认 64K 首轮的耗时与显存，再跑完整会话。

## 阻塞

- 无硬阻塞。待核实：Qwen3-4B-Instruct-2507 的 revision 与结构；真实轨迹数据集的名称与 schema；64K 首轮在参考 PyTorch 路径上的资源占用。
