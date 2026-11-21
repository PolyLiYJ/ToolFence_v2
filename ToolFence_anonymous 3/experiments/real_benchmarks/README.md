# reviewed manifest 真实评测

本目录在真实 AgentDojo、ASB、MCPtox 子集上评测匿名 ToolFence 严格 profile。

## reviewed manifest

三份人工可读配置：

- `manifests/agentdojo.json`
- `manifests/asb.json`
- `manifests/mcptox.json`

manifest 是工具语义的唯一可信来源。未配置工具不进入 trusted catalog 和 pinned manifest；ASB attack tools 与 MCPtox poisoned-only tools 显式进入 denylist。旧的名称前缀推断只作为 deprecated helper，真实评测不再使用。

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

汇总：

```bash
python -m experiments.real_benchmarks.summarize_results \
  results_real/reviewed_manifest_100 \
  --old-dir results_real/final_100_qwen
```

输出 `summary.csv` 和 `REPORT.md`，包含新旧 manifest 与跨模型对比。
