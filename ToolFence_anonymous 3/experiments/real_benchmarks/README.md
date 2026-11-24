# Reviewed-manifest benchmark evaluation

This directory evaluates the anonymous ToolFence strict profile on real AgentDojo, ASB, and MCPtox subsets.

## Reviewed manifests

Three human-readable configurations:

- `manifests/agentdojo.json`
- `manifests/asb.json`
- `manifests/mcptox.json`

The manifest is the single trusted source of tool semantics. Tools that are not configured do not enter the trusted catalog or the pinned manifest; ASB attack tools and MCPtox poisoned-only tools are explicitly placed in the denylist. The old name-prefix inference remains only as a deprecated helper and is no longer used by real evaluations.

## Per-case session isolation

Strict-profile tasks are **one-shot**: the same `tool + arguments` task cannot be reused after it completes once. Runners must isolate sessions per case:

- ASB: create a separate boundary for each `case + condition`, load only that case's normal tools, and use a separate ledger.
- MCPtox: create a separate boundary for each `case + condition`, load only that server's clean tools, and use a separate ledger.
- AgentDojo: call `profile.new_session()` before each clean/attack case to clear task bindings and avoid reusing `done` state across cases.

If the same boundary is reused, subsequent cases can show:

```text
policy: allowed=true
denied: 0
execution: Workflow token unavailable or wrong stage
executed_calls: []
```

This is sometimes miscounted as a utility drop, but it is not a policy denial.

## Testing the three benchmarks

Run the following commands from the repository root. All three runners share the same reviewed manifests.

### 1. Install dependencies

```bash
cd <repo-root>
python -m pip install -e '.[test,llm]'
python -m pip install agentdojo          # required by the AgentDojo runner
export PYTHONPATH=src:.
```

The ASB runner also needs the Blueprinting source tree. MCPTox data defaults to `data/mcptox/response_all.json`.

```bash
# ASB: Blueprinting source root containing benchmarks/asb_memory.py
export TOOLFENCE_BLUEPRINTING=/abs/path/to/Blueprinting

# AgentDojo: split JSONL and original source root
export TOOLFENCE_AGENTDOJO_SPLIT=/abs/path/to/agentdojo_split/cases.jsonl
export TOOLFENCE_AGENTDOJO_SRC=/abs/path/to/agentdojo_orig

# MCPTox: response_all.json from the pinned commit
mkdir -p data/mcptox
curl -L \
  https://raw.githubusercontent.com/zhiqiangwang4/MCPTox-Benchmark/f85189f9ad12504c197c7f920ab818a40657b1fa/response_all.json \
  -o data/mcptox/response_all.json
```

The AgentDojo split JSONL contains one case per line and must include at least the `suite`, `user_task_id`, and `injection_task_id` fields used by the runner; other fields follow the existing split format.

### 2. Start model servers

Qwen3-8B (the `qwen` profile):

```bash
vllm serve Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8000 \
  --served-model-name Qwen/Qwen3-8B \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Llama-3.3-70B-AWQ-INT4 (the `llama70b` profile, 2 GPUs):

```bash
export TOOLFENCE_LLAMA_MODEL=/abs/path/to/Meta-Llama-3.3-70B-Instruct-AWQ-INT4
vllm serve "$TOOLFENCE_LLAMA_MODEL" \
  --host 127.0.0.1 --port 8002 \
  --served-model-name "$TOOLFENCE_LLAMA_MODEL" \
  --quantization awq_marlin --dtype auto \
  --tensor-parallel-size 2 --max-model-len 4096 \
  --enable-auto-tool-choice --tool-call-parser llama3_json
```

If the service address or model name differs from the defaults, set:

```bash
export TOOLFENCE_QWEN_BASE_URL=http://127.0.0.1:8000/v1
export TOOLFENCE_QWEN_MODEL=Qwen/Qwen3-8B
export TOOLFENCE_LLAMA_BASE_URL=http://127.0.0.1:8002/v1
# TOOLFENCE_LLAMA_MODEL must match vLLM's --served-model-name
```

`--model-profile` accepts `qwen`, `llama70b`, or `both`. `both` requires both services and writes results under separate `qwen/` and `llama70b/` subdirectories.

### 3. Run smoke tests

Smoke mode uses six cases per benchmark to verify the services, data paths, and manifests before a full run. The three runners share one output root:

```bash
export OUT=results_real/benchmarks_smoke

python -m experiments.real_benchmarks.run_agentdojo_100 \
  --smoke --model-profile qwen --judges qwen --output-dir "$OUT"

python -m experiments.real_benchmarks.run_asb_100 \
  --smoke --model-profile qwen --judges qwen --output-dir "$OUT"

python -m experiments.real_benchmarks.run_mcptox_100 \
  --smoke --model-profile qwen --judges qwen --output-dir "$OUT"
```

### 4. Run the 100-case subset

```bash
export OUT=results_real/benchmarks_100

python -m experiments.real_benchmarks.run_agentdojo_100 \
  --subset-size 100 --model-profile qwen --judges qwen --output-dir "$OUT"

python -m experiments.real_benchmarks.run_asb_100 \
  --subset-size 100 --model-profile qwen --judges qwen --output-dir "$OUT"

python -m experiments.real_benchmarks.run_mcptox_100 \
  --subset-size 100 --model-profile qwen --judges qwen --output-dir "$OUT"
```

To compare both models, change `--model-profile qwen` to `--model-profile both`. Long jobs should run in a detached screen:

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

`--judges qwen` uses the current profile's local OpenAI-compatible endpoint as the judge. To use a Gemini judge, set `GOOGLE_API_KEY` and change `--judges` to `gemini`.

### 5. Aggregate results

```bash
python -m experiments.real_benchmarks.summarize_results \
  results_real/benchmarks_100 \
  --old-dir results_real/final_100_qwen
```

Outputs:

```text
results_real/benchmarks_100/summary.csv
results_real/benchmarks_100/REPORT.md
```

If the old result directory does not exist, the summary still includes the new results; only `old_*` and `delta_*` fields are empty.

### 6. Output files

Each profile has its own subdirectory, and each benchmark produces one JSON file:

```text
results_real/benchmarks_100/qwen/agentdojo.json
results_real/benchmarks_100/qwen/asb.json
results_real/benchmarks_100/qwen/mcptox.json
results_real/benchmarks_100/llama70b/agentdojo.json
results_real/benchmarks_100/llama70b/asb.json
results_real/benchmarks_100/llama70b/mcptox.json
```

Each JSON file contains `summary`, per-case `rows`, `provenance`, `profile_audits`, and a redacted `config`. All long jobs should run in a detached screen so that a local terminal disconnect does not interrupt them.

## Metric definitions

- AgentDojo: official suite verifier.
- ASB: official ASB adapter scores model-generated tool-call plans.
- MCPtox: adapted verifier; execution of poisoned-only tools or sensitive targets is counted as attack success.

Result directories:

- `results_real/reviewed_manifest_smoke/`
- `results_real/reviewed_manifest_100/`
- `results_real/reviewed_manifest_fix_smoke/` (case-isolation fix verification)
- `results_real/reviewed_manifest_100_fixed/` (full run after the fix)
- `results_real/mcptox_manifest_param_fix_smoke2/` (MCPTox argument-parsing fix smoke test)
- `results_real/mcptox_manifest_param_fix_100/` (full run after the MCPTox argument-parsing fix)

Summary:

```bash
python -m experiments.real_benchmarks.summarize_results \
  results_real/reviewed_manifest_100 \
  --old-dir results_real/final_100_qwen
```

This writes `summary.csv` and `REPORT.md` with old/new manifest and cross-model comparisons.
