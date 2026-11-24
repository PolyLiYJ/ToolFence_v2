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

## 测试三个 benchmark

以下命令均在仓库根目录执行，三个 runner 共用同一套 reviewed manifest。

### 1. 安装依赖

```bash
cd <repo-root>
python -m pip install -e '.[test,llm]'
python -m pip install agentdojo          # AgentDojo runner 需要
export PYTHONPATH=src:.
```

ASB runner 还需要 Blueprinting 源码；MCPTox 数据默认放在 `data/mcptox/response_all.json`。

```bash
# ASB：Blueprinting 源码根目录，内含 benchmarks/asb_memory.py
export TOOLFENCE_BLUEPRINTING=/abs/path/to/Blueprinting

# AgentDojo：split JSONL 和原始源码根目录
export TOOLFENCE_AGENTDOJO_SPLIT=/abs/path/to/agentdojo_split/cases.jsonl
export TOOLFENCE_AGENTDOJO_SRC=/abs/path/to/agentdojo_orig

# MCPTox：固定 commit 的 response_all.json
mkdir -p data/mcptox
curl -L \
  https://raw.githubusercontent.com/zhiqiangwang4/MCPTox-Benchmark/f85189f9ad12504c197c7f920ab818a40657b1fa/response_all.json \
  -o data/mcptox/response_all.json
```

AgentDojo 的 split JSONL 每行一个 case，至少包含 runner 使用的 `suite`、`user_task_id`、`injection_task_id` 字段；其余字段沿用现有 split 格式。

### 2. 启动模型服务

Qwen3-8B（对应 `qwen` profile）：

```bash
vllm serve Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8000 \
  --served-model-name Qwen/Qwen3-8B \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Llama-3.3-70B-AWQ-INT4（对应 `llama70b` profile，2 张 GPU）：

```bash
export TOOLFENCE_LLAMA_MODEL=/abs/path/to/Meta-Llama-3.3-70B-Instruct-AWQ-INT4
vllm serve "$TOOLFENCE_LLAMA_MODEL" \
  --host 127.0.0.1 --port 8002 \
  --served-model-name "$TOOLFENCE_LLAMA_MODEL" \
  --quantization awq_marlin --dtype auto \
  --tensor-parallel-size 2 --max-model-len 4096 \
  --enable-auto-tool-choice --tool-call-parser llama3_json
```

如果服务地址或模型名与默认值不同，设置：

```bash
export TOOLFENCE_QWEN_BASE_URL=http://127.0.0.1:8000/v1
export TOOLFENCE_QWEN_MODEL=Qwen/Qwen3-8B
export TOOLFENCE_LLAMA_BASE_URL=http://127.0.0.1:8002/v1
# TOOLFENCE_LLAMA_MODEL 需要与 vLLM 的 --served-model-name 一致
```

`--model-profile` 可选 `qwen`、`llama70b`、`both`。`both` 要求两个服务同时可用，并会在输出目录下分别写 `qwen/` 和 `llama70b/`。

### 3. 先跑 smoke

smoke 每个 benchmark 只取 6 条 case，用来先确认服务、数据路径和 manifest 都正常。三个 runner 使用同一个输出根目录：

```bash
export OUT=results_real/benchmarks_smoke

python -m experiments.real_benchmarks.run_agentdojo_100 \
  --smoke --model-profile qwen --judges qwen --output-dir "$OUT"

python -m experiments.real_benchmarks.run_asb_100 \
  --smoke --model-profile qwen --judges qwen --output-dir "$OUT"

python -m experiments.real_benchmarks.run_mcptox_100 \
  --smoke --model-profile qwen --judges qwen --output-dir "$OUT"
```

### 4. 跑 100 子集

```bash
export OUT=results_real/benchmarks_100

python -m experiments.real_benchmarks.run_agentdojo_100 \
  --subset-size 100 --model-profile qwen --judges qwen --output-dir "$OUT"

python -m experiments.real_benchmarks.run_asb_100 \
  --subset-size 100 --model-profile qwen --judges qwen --output-dir "$OUT"

python -m experiments.real_benchmarks.run_mcptox_100 \
  --subset-size 100 --model-profile qwen --judges qwen --output-dir "$OUT"
```

想同时对比两个模型，把 `--model-profile qwen` 改成 `--model-profile both`。长任务建议放在 detached screen 中：

```bash
screen -dmS toolfence_bench bash -lc '
  cd <repo-root> &&
  export PYTHONPATH=src:. &&
  export TOOLFENCE_BLUEPRINTING=/abs/path/to/Blueprinting &&
  python -m experiments.real_benchmarks.run_mcptox_100 \
    --subset-size 100 --model-profile both --judges qwen \
    --output-dir results_real/benchmarks_100
'
```

`--judges qwen` 表示使用当前 profile 的本地 OpenAI 兼容端点作为 judge；如需 Gemini judge，设置 `GOOGLE_API_KEY` 并把 `--judges` 改为 `gemini`。

### 5. 汇总

```bash
python -m experiments.real_benchmarks.summarize_results \
  results_real/benchmarks_100 \
  --old-dir results_real/final_100_qwen
```

输出：

```text
results_real/benchmarks_100/summary.csv
results_real/benchmarks_100/REPORT.md
```

如果旧结果目录不存在，汇总仍会生成新结果，只是 `old_*` 和 `delta_*` 为空。

### 6. 输出文件

每个 profile 一个子目录，每个 benchmark 一个 JSON：

```text
results_real/benchmarks_100/qwen/agentdojo.json
results_real/benchmarks_100/qwen/asb.json
results_real/benchmarks_100/qwen/mcptox.json
results_real/benchmarks_100/llama70b/agentdojo.json
results_real/benchmarks_100/llama70b/asb.json
results_real/benchmarks_100/llama70b/mcptox.json
```

每个 JSON 包含 `summary`、逐 case 的 `rows`、`provenance`、`profile_audits` 和脱敏后的 `config`。所有长任务应在 detached screen 中运行，避免本地终端断开导致中断。

## 结果口径

- AgentDojo：官方 suite verifier。
- ASB：官方 ASB adapter 对模型生成 tool-call plan 打分。
- MCPtox：adapted verifier；poisoned-only 工具或敏感目标被执行记为攻击成功。

结果目录：

- `results_real/reviewed_manifest_smoke/`
- `results_real/reviewed_manifest_100/`
- `results_real/reviewed_manifest_fix_smoke/`（case 隔离修复验证）
- `results_real/reviewed_manifest_100_fixed/`（修复版全量）
- `results_real/mcptox_manifest_param_fix_smoke2/`（MCPtox 参数解析修复 smoke）
- `results_real/mcptox_manifest_param_fix_100/`（MCPtox 参数解析修复全量）

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
| MCPtox | no_defense | 81.0 | 69.0 | 41.0 |
| MCPtox | toolfence_strict | 57.0 | 43.0 | 4.0 |

ASB 修复后 strict utility 与 no_defense 持平。MCPtox 的参数解析修复后，strict 从 `BU 12` 提升到 `57`、`U@A 15` 提升到 `43`；剩余 clean 失败主要是 compiler 上下文超限、非敏感参数 literal grounding，以及少数重名工具/敏感参数规则，不再是工具选择或 case 会话隔离问题。ASR 的 4 个 case 中，部分是把用户明确请求的 URL 或 `/etc/hosts` 路径也计入敏感目标，属于评测口径与参数敏感性规则需要继续区分的情况。
