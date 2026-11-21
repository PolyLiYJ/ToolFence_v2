# reviewed manifest 真实评测

本目录在真实 AgentDojo、ASB、MCPtox 子集上评测匿名 ToolFence 严格 profile。

## reviewed manifest

三份人工可读配置：

- `manifests/agentdojo.json`
- `manifests/asb.json`
- `manifests/mcptox.json`

manifest 是工具语义的唯一可信来源。未配置工具不进入 trusted catalog 和 pinned manifest；ASB attack tools 与 MCPtox poisoned-only tools 显式进入 denylist。旧的名称前缀推断只作为 deprecated helper，真实评测不再使用。

## case 会话隔离

严格 profile 的 task 是**一次性**的：同一个 `tool + arguments` 的 task 完成一次后不能再复用。runner 必须按 case 隔离：

- ASB：每个 `case + condition` 新建独立 boundary，只加载该 case 的 normal tools，使用独立 ledger。
- MCPtox：每个 `case + condition` 新建独立 boundary，只加载该 server 的 clean tools，使用独立 ledger。
- AgentDojo：每个 clean/attack case 前调用 `profile.new_session()`，清空 task 绑定，避免跨 case 复用 `done` 状态。

如果复用同一个 boundary，后续 case 会出现：

```text
policy: allowed=true
denied: 0
execution: Workflow token unavailable or wrong stage
executed_calls: []
```

这会被误计为 utility 下降，但不是 policy 拒绝。

## 模型 profile

- `qwen`：`http://127.0.0.1:8000/v1`，`Qwen/Qwen3-8B`，GPU 6、7
- `llama70b`：`http://127.0.0.1:8002/v1`，`hugging-quants/Meta-Llama-3.3-70B-Instruct-AWQ-INT4`，GPU 4、5

runner 参数：

```bash
python -m experiments.real_benchmarks.run_agentdojo_100 \
  --model-profile both --judges qwen --output-dir results_real/reviewed_manifest_100
```

`--model-profile` 可选 `qwen`、`llama70b`、`both`。`both` 会在输出目录下分别写 `qwen/` 和 `llama70b/`。

所有长任务应在服务器 detached screen 中运行，避免本地 Mac 断开导致中断。

## 结果口径

- AgentDojo：官方 suite verifier。
- ASB：官方 ASB adapter 对模型生成 tool-call plan 打分。
- MCPtox：adapted verifier；poisoned-only 工具或敏感目标被执行记为攻击成功。

结果目录：

- `results_real/reviewed_manifest_smoke/`
- `results_real/reviewed_manifest_100/`
- `results_real/reviewed_manifest_fix_smoke/`（case 隔离修复验证）
- `results_real/reviewed_manifest_100_fixed/`（修复版全量）

汇总：

```bash
python -m experiments.real_benchmarks.summarize_results \
  results_real/reviewed_manifest_100 \
  --old-dir results_real/final_100_qwen
```

输出 `summary.csv` 和 `REPORT.md`，包含新旧 manifest 与跨模型对比。

## 修复版结果（Qwen3-8B，100 子集）

| benchmark | defense | BU | U@A | ASR |
|---|---|---:|---:|---:|
| AgentDojo | no_defense | 31.746 | 27.0 | 16.0 |
| AgentDojo | toolfence_strict | 15.873 | 13.0 | 3.0 |
| ASB | no_defense | 51.0 | 5.0 | 43.0 |
| ASB | toolfence_strict | 51.0 | 5.0 | 0.0 |
| MCPtox | no_defense | 75.0 | 73.0 | 43.0 |
| MCPtox | toolfence_strict | 12.0 | 15.0 | 0.0 |

ASB 修复后 strict utility 与 no_defense 持平。MCPtox 仍主要受 planner 选错工具限制，case 隔离只解决了一部分。
