# PCIe / 并发传输饱和 Microbenchmark 计划

## 结论

这组实验能做，而且值得做。

但以当前 `vllm_speculative` 的实现，**还不能直接支撑** Reviewer #2-3 想要的结论，原因是现有代码只做了“触发异步迁移”，没有把以下几个量测清楚：

- 单次 draft model reload 的真实完成时间
- 并发 reload / offload 时的带宽退化
- reload 是否被 decode / verify 计算重叠隐藏
- 用户可见的 stall 到底是多少

所以建议分成两层实验：

1. **独立 PCIe microbenchmark**
   直接测 host<->device 传输，回答“PCIe 会不会饱和”。
2. **Nightjar 集成态 benchmark**
   在真实 `offload/reload` 路径上埋点，回答“reload latency 是否能被 overlap 隐藏”。

## 为什么现状不够

### 1. 当前 offload/reload 计时不是真正的传输完成时间

在 [vllm/spec_decode/spec_decode_worker.py](/root/autodl-tmp/nightjar/vllm_speculative/vllm/spec_decode/spec_decode_worker.py:1466)：

- `offload_proposer_worker()` 用线程池提交 `self.model.to("cpu", non_blocking=True)`。
- `load_neural_model_async()` 用线程池提交 `self.model.to("cuda", non_blocking=True)`。
- 外层日志记录的是“提交线程任务”前后的时间，不是 DMA 真正完成时间。

关键位置：

- [spec_decode_worker.py:1481](/root/autodl-tmp/nightjar/vllm_speculative/vllm/spec_decode/spec_decode_worker.py:1481)
- [spec_decode_worker.py:1494](/root/autodl-tmp/nightjar/vllm_speculative/vllm/spec_decode/spec_decode_worker.py:1494)
- [spec_decode_worker.py:1510](/root/autodl-tmp/nightjar/vllm_speculative/vllm/spec_decode/spec_decode_worker.py:1510)
- [spec_decode_worker.py:1515](/root/autodl-tmp/nightjar/vllm_speculative/vllm/spec_decode/spec_decode_worker.py:1515)

### 2. `LLMEngine` 过早翻转状态位

在 [vllm/engine/llm_engine.py:2230](/root/autodl-tmp/nightjar/vllm_speculative/vllm/engine/llm_engine.py:2230)：

- `load_neural_model_async()` 一调用就把 `self.proposer_worker_to_cpu = False`
- scheduler 也同步改成 `False`

这会让系统状态看起来像“draft 已经回到 GPU”，但真实传输可能还没结束。这样无法准确区分：

- reload 开始时间
- reload 完成时间
- reload 期间是否发生 stall

### 3. 现有 event log 只有 KV migration，没有 draft reload/offload 的完整事件

目前已有：

- `speculative_step`
- `kv_block_migration`

例如 [vllm/core/block_manager.py:655](/root/autodl-tmp/nightjar/vllm_speculative/vllm/core/block_manager.py:655) 已记录 KV migration duration。

但还缺：

- `draft_reload_start`
- `draft_reload_done`
- `draft_offload_start`
- `draft_offload_done`
- reload 的 bytes、stream 数、并发数、raw latency、visible stall

## 总体实验设计

### A. Standalone PCIe microbenchmark

目的：回答审稿人“并发 CPU-GPU 传输是否会饱和 PCIe，导致 reload latency 明显增长”。

测量对象：

- H2D reload latency
- D2H offload latency
- aggregate bandwidth
- 并发度增加时的 tail latency

建议实验矩阵：

- 方向：`H2D`、`D2H`
- payload size：
  - 64 MB
  - 256 MB
  - 512 MB
  - 1 GB
  - `draft_model_bytes` 附近的真实大小
- 并发数：
  - 1
  - 2
  - 4
  - 8
- 执行模式：
  - 单 GPU 多 stream
  - 多 GPU 同时传输（如果机器有多卡）
- host buffer：
  - pinned memory
  - pageable memory（可选，只作对照）

输出指标：

- `raw_latency_ms`
- `p50/p95/p99 latency`
- `effective_bandwidth_GBps`
- `aggregate_bandwidth_GBps`
- `saturation_point`

### B. Nightjar 集成态 overlap benchmark

目的：回答“reload latency 能否被 overlap 隐藏”。

测量对象：

- reload 的原始完成时间
- reload 发起后，到系统真正因为 draft 不可用而停顿了多久
- reload 与 verify/decode 是否重叠

建议做三类场景：

1. **Isolated reload**
   没有活跃 decode，请求空闲时只测 reload。

2. **Contended reload**
   reload 同时叠加其他 host<->device 传输压力。
   可用一个后台 stressor 线程/进程不断做 pinned H2D / D2H copy。

3. **Overlapped reload**
   在 Nightjar 正常 decode 中触发：
   - 先 offload draft model
   - 之后在低负载时 reload
   - 记录 reload 期间 `speculative_step` 和 e2e latency 的变化

输出指标：

- `reload_raw_ms`
- `reload_exposed_stall_ms`
- `reload_hidden_ratio = 1 - exposed_stall / raw_latency`
- `next_spec_step_delay_ms`
- `e2e_latency_delta_ms`
- `ttft/tpot delta`

## 需要改的代码位置

### 1. 在 `SpecDecodeWorker` 增加 draft transfer instrumentation

文件：

- [vllm/spec_decode/spec_decode_worker.py](/root/autodl-tmp/nightjar/vllm_speculative/vllm/spec_decode/spec_decode_worker.py:1466)

建议改动：

- 给 offload/reload 建立独立的 transfer state
- 不再只靠一个 `self.event`
- 为每次 transfer 生成 `transfer_id`
- 记录：
  - `direction`
  - `start_ts`
  - `submit_ts`
  - `complete_ts`
  - `bytes`
  - `device_id`
  - `stream_id`
  - `concurrency_group`

建议新增成员：

```python
self.draft_transfer_state = {
    "active": False,
    "transfer_id": None,
    "direction": None,
    "submit_ts": None,
    "start_ts": None,
    "complete_ts": None,
    "bytes": None,
    "status": "idle",
}
```

建议新增方法：

- `_estimate_draft_model_bytes()`
- `_start_draft_transfer(direction: str, target_device: str, stream=None)`
- `_poll_draft_transfer()`
- `_finalize_draft_transfer_if_done()`

### 2. 用显式 CUDA stream/event 测真正完成时间

文件：

- [vllm/spec_decode/spec_decode_worker.py](/root/autodl-tmp/nightjar/vllm_speculative/vllm/spec_decode/spec_decode_worker.py:1500)

建议不要只记录“提交线程池”的 wall-clock 时间，改成：

- 创建专用 copy stream，例如 `self.draft_transfer_stream`
- 在该 stream 上记录 `start_event` / `end_event`
- 如果 `Module.to()` 无法稳定给出逐层异步 copy 完成语义，则进一步下沉到参数级 copy，或者先做 standalone benchmark 时用 tensor copy 替代验证

注意点：

- `nn.Module.to(..., non_blocking=True)` 对整个 module 的异步语义不够透明，论文里不要把它直接表述成“non-blocking DMA 已被严格验证”，除非 instrumentation 证明其完成时序。
- 如果要拿论文结果，最好最终测的是“真实 draft 权重集合的逐 tensor copy 总时间”。

### 3. 修正 engine 层状态机

文件：

- [vllm/engine/llm_engine.py](/root/autodl-tmp/nightjar/vllm_speculative/vllm/engine/llm_engine.py:2230)

建议改动：

- `load_neural_model_async()` 调用后，不要立刻把：
  - `self.proposer_worker_to_cpu = False`
  - `self.scheduler[virtual_engine].proposer_worker_to_cpu = False`
- 而是等 worker 明确报告 reload complete 后再翻转

建议增加状态：

- `draft_model_state in {"on_gpu", "offloading", "on_cpu", "reloading"}`

这样后续 event log 才能区分：

- speculation disabled because planner chose `gamma=0`
- speculation unavailable because draft model still reloading

### 4. 扩展 Nightjar event logger

文件：

- [vllm/engine/nightjar_event_logger.py](/root/autodl-tmp/nightjar/vllm_speculative/vllm/engine/nightjar_event_logger.py:1)

建议新增事件：

- `draft_transfer_start`
- `draft_transfer_done`
- `draft_transfer_poll`
- `draft_transfer_overlap`

字段建议：

```json
{
  "event_type": "draft_transfer_done",
  "transfer_id": "uuid",
  "direction": "cpu_to_gpu",
  "bytes": 536870912,
  "raw_latency_ms": 7.31,
  "wall_latency_ms": 7.58,
  "device_id": 0,
  "stream_label": "draft_transfer",
  "concurrency": 4,
  "draft_model_state_before": "on_cpu",
  "draft_model_state_after": "on_gpu"
}
```

### 5. 在 scheduler / decode path 增加 exposed stall 观测点

文件：

- [vllm/core/scheduler.py](/root/autodl-tmp/nightjar/vllm_speculative/vllm/core/scheduler.py:1596)

已有 `speculative_step` 日志，可以继续利用。建议再补：

- 当 planner 希望重新启用 speculation，但 draft 仍在 reload 中时，打事件：
  - `draft_reload_blocked_speculation`
- 记录：
  - 当前 batch size
  - queue length
  - wait time until draft ready

这样可以定义：

- `visible_stall_ms = speculation-ready time - planner-requested reload time`

### 6. 新增独立 microbenchmark 脚本

建议新增文件：

- `benchmarks/pcie_transfer_microbenchmark.py`

功能：

- 分配 pinned host tensor / device tensor
- 支持 H2D / D2H
- 支持多 stream 并发
- 支持多进程多 GPU 并发
- 输出 csv/json

CLI 建议：

```bash
python benchmarks/pcie_transfer_microbenchmark.py \
  --direction h2d \
  --size-mb 512 \
  --concurrency 4 \
  --iters 200 \
  --device 0 \
  --pinned true \
  --output benchmark_results/pcie_h2d_c4.json
```

### 7. 新增 Nightjar 集成实验脚本

建议新增：

- `run_pcie_reload_benchmark.sh`
- 或 `exps/run_pcie_reload_benchmark.py`

职责：

- 触发固定次数 offload/reload
- 选择是否挂载 background copy stressor
- 保存：
  - Nightjar event log
  - benchmark json
  - summary csv

### 8. 新增分析脚本

建议新增：

- `exps/plot_draft_transfer_overhead.py`

输入：

- `nightjar_events.jsonl`
- standalone microbenchmark csv/json

输出：

- 单路/并发 reload latency 图
- latency vs concurrency 图
- aggregate bandwidth 图
- hidden ratio 图

## 实验定义建议

### 实验 1：单路 reload latency

问题：

- 一次真实 draft reload 从 CPU 到 GPU 要多久？

做法：

- 先 offload 到 CPU
- 空闲状态 reload 回 GPU
- 重复 50 到 100 次

输出：

- `p50/p95 raw reload latency`
- `draft_model_bytes`
- `effective GB/s`

### 实验 2：并发传输饱和

问题：

- 当存在并发 CPU-GPU 传输时，reload latency 会增长多少？

做法：

- 保持 Nightjar reload 不变
- 同时启动 `N=1/2/4/8` 个 background copy workers
- workers 做相同方向或混合方向 host<->device copy

输出：

- `reload latency vs concurrency`
- `aggregate bandwidth vs concurrency`
- 是否出现明显 saturation knee

### 实验 3：overlap 是否隐藏 reload

问题：

- reload latency 是否会出现在端到端关键路径上？

做法：

- 在低负载阶段触发 reload
- 对比两组：
  - `reload with no overlap`
  - `reload during active decode/verify`

输出：

- `raw_reload_ms`
- `visible_stall_ms`
- `hidden_ratio`
- `e2e_latency_delta`

## 建议论文里最后报告的表/图

### 表 1：Standalone PCIe transfer benchmark

列建议：

- direction
- payload size
- concurrency
- p50 latency
- p95 latency
- effective GB/s

### 图 1：Reload latency vs concurrency

横轴：

- 并发数 `1/2/4/8`

纵轴：

- reload latency

### 图 2：Hidden ratio under overlap

横轴：

- 场景：isolated / contended / overlapped

纵轴：

- hidden ratio

### 表 2：Visible stall in integrated Nightjar runs

列建议：

- workload
- with stressor / without stressor
- raw reload ms
- visible stall ms
- TTFT delta
- TPOT delta

## 实现优先级

### P0：先做，决定实验能不能回答审稿人

1. 给 draft offload/reload 补完整 event log
2. 修正 `LLMEngine` 的 draft state 翻转时机
3. 新增 standalone PCIe microbenchmark 脚本

### P1：第二步做，得到可投稿图表

1. background PCIe stressor
2. integrated reload benchmark
3. hidden ratio / visible stall 统计脚本

### P2：如果时间够，再做增强

1. 多 GPU 同时 reload
2. 多租户模拟
3. 不同 payload size 曲线

## 风险点

### 1. 当前 `Module.to()` 不一定是你想象中的“纯 DMA”

如果最终发现 `self.model.to("cuda")` 包含：

- 参数重绑定
- allocator 行为
- lazy init

那就要把“standalone tensor copy microbenchmark”和“真实模型 reload benchmark”分开报，不能混成一个数字。

### 2. 21.9us 这个量级大概率需要重新核实

如果你们文中所谓 reload latency 是：

- 线程提交时间
- API 返回时间
- 还没等 DMA 完成

那这个数不能直接拿来回 Reviewer #2-3。

### 3. overlap 的定义必须统一

建议论文里明确：

- `raw reload latency`: transfer start 到 transfer complete
- `visible stall latency`: reload 导致 speculation 或 decode 真正等待的时间
- `hidden ratio`: `1 - visible_stall / raw_reload`

## 推荐落地顺序

1. 先做 `benchmarks/pcie_transfer_microbenchmark.py`
2. 再给 `SpecDecodeWorker` 和 `LLMEngine` 补 transfer state 与 event log
3. 再做 integrated reload benchmark
4. 最后用 `exps/plot_*` 生成论文图表

## 直接回答“这能做实验吗”

可以做。

但要想让结果足够硬，能正面回应 Reviewer #2-3，需要至少补以下代码：

- `vllm/spec_decode/spec_decode_worker.py`
- `vllm/engine/llm_engine.py`
- `vllm/engine/nightjar_event_logger.py`
- 新增 `benchmarks/pcie_transfer_microbenchmark.py`
- 新增一个 integrated reload benchmark 脚本
- 新增一个 draft transfer 分析脚本

如果只用现有代码直接跑，最多只能得到“Nightjar 有 offload/reload 行为”，**得不到**“PCIe 是否饱和、reload latency 是否被 overlap 隐藏”这两个审稿人真正要的结论。
