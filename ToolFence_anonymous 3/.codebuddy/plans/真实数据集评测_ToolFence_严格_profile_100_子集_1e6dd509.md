---
name: 真实数据集评测：ToolFence 严格 profile 100 子集
overview: 在服务器上改造匿名 ToolFence 代码，接入 AgentDojo / ASB / MCPtox 三个真实数据集各 100 个子集，使用 Google AI Studio 免费 Flash 模型作为 agent/compiler/judge，按论文严格 profile 输出可复现的 BU/U@A/ASR 结果。
todos:
  - id: map-entrypoints
    content:
      - "#text": 使用 [subagent:code-explorer] 映射 runner 与严格 profile 接入点
    status: completed
  - id: google-backend
    content:
      - "#text": 实现 Google Flash 后端、配置与脱敏审计
    status: completed
    dependencies:
      - map-entrypoints
  - id: strict-profile
    content:
      - "#text": 实现严格论文 profile 构建与 host adapter
    status: completed
    dependencies:
      - map-entrypoints
  - id: agentdojo-runner
    content:
      - "#text": 实现 AgentDojo 100 子集 runner 与官方打分
    status: completed
    dependencies:
      - google-backend
      - strict-profile
  - id: asb-mcptox-runner
    content:
      - "#text": 下载数据并实现 ASB、MCPtox 100 子集 runner
    status: completed
    dependencies:
      - google-backend
      - strict-profile
  - id: smoke-fix
    content:
      - "#text": 运行小样本 smoke 并修复 provider 或适配问题
    status: completed
    dependencies:
      - agentdojo-runner
      - asb-mcptox-runner
  - id: final-run
    content:
      - "#text": 跑三个 100 子集并汇总真实结果
    status: completed
    dependencies:
      - smoke-fix
---

## User Requirements

在服务器上修改并扩展匿名 ToolFence 代码，使其能够连接真实数据集并产出评测结果。运行规模为 AgentDojo、ASB、MCPtox 各 100 个子集。MCPtox 数据允许从公开仓库下载到服务器。防御配置必须使用严格论文 profile，即接入 `ToolFenceAuthorizationBoundary`，补齐可信工具目录、controller-pinned manifest、guarded transaction ledger、host adapter、host actor/reviewer 和 policy revision。模型使用用户提供的 Google AI Studio API key 调用免费 Flash 模型；key 只能通过服务器环境变量注入，不得写入代码、配置、日志或结果文件。

## Product Overview

产出三个真实数据集的可复现子集评测：每个 benchmark 固定 seed 抽样 100 个子集，分别运行无防御基线和严格 ToolFence profile，输出 BU、U@A、ASR、完成数、错误数、拒绝数、judge 调用数和耗时。结果需包含数据来源、样本 ID、hash、配置快照和运行日志。必须明确说明模型为 Google Flash，不同于论文主表模型，因此结果不是论文主表复现。

## Core Features

- 固定随机种子生成三个 benchmark 的 100 子集清单并持久化。
- 使用 Google AI Studio OpenAI-compatible 接口作为 agent、compiler、judge 后端，并做 provider-aware 请求构造。
- 为严格 profile 构建可信工具目录、pinned manifest、事务账本、host bindings、reviewer 和策略版本。
- 复用服务器已有的 AgentDojo、ASB runner 和数据适配逻辑，新增匿名库专用 runner。
- 下载 MCPtox 公开数据并记录来源、commit 和 hash。
- 支持失败隔离、断点续跑、速率限制、结果汇总和脱敏配置快照。

## Tech Stack Selection

- 语言与运行环境：Python 3.10，服务器 `/home/yjli/anaconda3/envs/piarena/bin/python`。
- 代码基线：服务器 `/home/yjli/ToolFence_anonymous3`，沿用现有 `src/toolfence/` 模块化结构。
- AgentDojo：复用 `/home/yjli/Agent/Blueprinting/data/benchmarks/agentdojo_orig` 与 `experiments/evaluate_agentdojo_defenses.py` 的执行框架。
- ASB：复用 `/home/yjli/Agent/Blueprinting/benchmarks/asb_memory.py` 和 `run_asb_memory_poisoning.py` 的数据适配与官方数据 pcipline。
- MCPTox：从公开仓库下载到 `/home/yjli/ToolFence_anonymous3/data/mcptox/`，优先使用官方 verifier 或官方实例标签。
- 模型接口：Google AI Studio OpenAI-compatible endpoint `https://generativelanguage.googleapis.com/v1beta/openai/`，密钥通过 `GOOGLE_API_KEY` 注入。
- 结果格式：JSON + CSV，输出到 `/home/yjli/ToolFence_anonymous3/results_real/<timestamp>/`。

## Implementation Approach

策略是“新增真实 benchmark 适配层 + 严格 profile 构建器 + 三个 runner”，尽量不修改 `src/toolfence/` 核心机制，仅在 smoke 测试证明必要时做最小兼容性修复。

关键决策：

- 不直接运行原版 `Blueprinting` 的 defense，因为那测试的是原项目而非匿名库；应把匿名库 `toolfence` 的严格 wrapper 插入真实 benchmark pipeline。
- 不修改论文核心算法语义，新增适配层负责把 benchmark 工具目录转换为 controller-owned trusted catalog、pinned manifest 和 host bindings。
- Google Flash 与 Qwen/vLLM 不同，不能发送 `extra_body={"think": False}`；新增 provider-aware backend，默认只发送 OpenAI-compatible 标准字段，必要时用 JSON-only 提示词修复输出。
- 每个 benchmark 同时跑 `no_defense` 与 `toolfence_strict`，确保结果能计算安全收益和效用损失。
- 使用固定 seed=17、逐 case checkpoint、并发上限和指数退避，控制免费 API 限流与长任务失败成本。

性能与瓶颈：

- 三个 benchmark 各 100 子集，严格 profile 每个任务至少触发 compiler，缺失能力时触发 isolated judge。主要瓶颈是 Google Flash API 延迟与限流，而不是本地计算。
- 默认并发设为 4，可配置；每次请求设置 timeout 与 retry。所有调用记录 token 与错误摘要，不记录 key 和大段 payload。
- 单 case 失败不终止整体任务；错误单独写入结果文件并从指标分母中显式记录。

## Implementation Notes

- 不停止现有 GPU 6、7 上的 Qwen3-8B vLLM 服务；本次评测默认走 Google Flash。
- API key 只以服务器进程环境变量传入；配置快照只保存环境变量名，不保存值。
- 不把用户提供的 key 写入任何文件、日志、commit 或结果 JSON。
- AgentDojo 必须从官方 suite 的 trusted tool schemas 生成 trusted catalog，禁止使用注入页面或工具输出生成权限。
- pinned manifest 的 effect、operation、authority-sensitive 字段必须由 host 配置固定，不能由模型输出决定。
- strict profile 必须通过 `toolfence.agentdojo_defense.ToolFenceAuthorizationBoundary`，禁止使用通用 boundary 的 agent-facing catalog 回退。
- 对未注册 host adapter 的工具一律 fail-closed，并在结果中记录为 denied 或 unsupported，不能静默放行。
- MCPTox 下载后记录仓库 URL、commit、数据文件 SHA256；若官方 verifier 不可用，则使用官方 attack label 加可执行调用匹配，并在结果中标注 `adapted_mcptox_verifier`。
- 完成后重跑 `PYTHONPATH=src python -m pytest -q tests/unit`，确认新增适配层没有破坏原有 95 个测试。

## Architecture Design

- `google_backends.py` 提供 compiler/judge 两个 backend，封装 Google Flash 调用、重试、token 统计和脱敏审计。
- `paper_profile.py` 根据 benchmark 工具配置和任务合同构建严格 profile 所需对象，并返回 boundary 与审计元数据。
- `runner_common.py` 负责固定抽样、checkpoint、并发调度、错误隔离、结果落盘和指标汇总。
- `run_agentdojo_100.py` 插入严格 boundary 到 AgentDojo pipeline，使用官方 suite verifier 计算 BU、U@A、ASR。
- `run_asb_100.py` 从 ASB official/manifest cases 抽样 100 条，生成 clean/attack paired 执行，并用 ASB adapter 计算指标。
- `run_mcptox_100.py` 加载下载后的 MCPtox 实例，抽样 100 条，执行工具投毒场景并统计攻击成功与任务完成。
- `summarize_results.py` 汇总三个 JSON 到统一 CSV，并生成模型、数据、profile 和指标说明。

数据流：
数据下载/定位 -> 固定抽样 100 -> 构建 trusted catalog 与 pinned manifest -> 创建严格 boundary -> 运行 no_defense 和 toolfence_strict -> 官方或适配 verifier 打分 -> checkpoint 与结果汇总。

## Directory Structure

```text
/home/yjli/ToolFence_anonymous3/
├── experiments/
│   └── real_benchmarks/
│       ├── README.md                  # [NEW] 说明真实数据集评测运行方式、环境变量、输出结构和模型差异声明。
│       ├── config.py                  # [NEW] 集中管理 endpoint、模型名、seed=17、并发、超时、输出目录和 benchmark 数量=100；不保存 key。
│       ├── google_backends.py         # [NEW] Google AI Studio OpenAI-compatible backend；provider-aware 请求体；重试、token 与审计。
│       ├── runner_common.py           # [NEW] 固定抽样、case hash、checkpoint、并发执行、失败隔离、结果写入工具。
│       ├── paper_profile.py           # [NEW] 构建 trusted catalog、pinned manifest、GuardedTransactionNet、host bindings、actor/reviewer、revision 和严格 boundary。
│       ├── agentdojo_adapter.py       # [NEW] 将 AgentDojo suite 工具模式转换为 controller-owned catalog，并把严格 boundary 插入 pipeline。
│       ├── asb_adapter.py             # [NEW] 适配 ASBMemoryPoisoningBenchmark 的 100 子集、工具调用计划和严格 boundary 执行。
│       ├── mcptox_adapter.py          # [NEW] 加载 MCPtox 数据、构造工具投毒场景、执行边界并打分；记录 verifier 类型。
│       ├── run_agentdojo_100.py       # [NEW] AgentDojo 100 子集入口，运行 no_defense 与 toolfence_strict。
│       ├── run_asb_100.py             # [NEW] ASB 100 子集入口，运行 no_defense 与 toolfence_strict。
│       ├── run_mcptox_100.py          # [NEW] MCPtox 100 子集入口，运行 no_defense 与 toolfence_strict。
│       └── summarize_results.py       # [NEW] 汇总三个 benchmark 结果到 summary.csv，并生成可读报告。
├── src/
│   └── toolfence/
│       ├── semantic/architect.py      # [MODIFY, CONDITIONAL] 仅当 Google endpoint 不兼容现有 JSON 请求时做最小 provider-aware 修复。
│       └── judge/runtime.py           # [MODIFY, CONDITIONAL] 仅当 judge 请求不兼容 Google endpoint 时做最小修复；优先在 google_backends.py 中绕过。
├── data/
│   ├── mcptox/                        # [NEW] 公开 MCPtox 数据仓库、commit、数据文件和 hash 清单。
│   └── asb_orig/                      # [NEW, CONDITIONAL] 若服务器缺失官方 ASB 数据，则下载或复制并记录 provenance。
├── tests/
│   └── unit/
│       └── test_real_benchmark_adapters.py # [NEW] 测试抽样稳定性、脱敏、严格 profile 构造和结果汇总格式。
└── results_real/
    └── <timestamp>/                   # [NEW] agentdojo.json、asb.json、mcptox.json、summary.csv、logs/、config_snapshot_redacted.json。
```

## Key Code Structures

```python
def build_paper_profile(
    tools: Sequence[Mapping[str, Any]],
    *,
    model_backends: ModelBackends,
    policy_revision: str,
    suite_name: str,
) -> PaperProfileBundle:
    ...
```

`PaperProfileBundle` 应包含 strict boundary、trusted catalog、pinned manifest、transaction net、host bindings、actor、reviewer、policy revision 和脱敏审计信息。

每个 runner 应输出统一字段：
`benchmark`、`subset_size`、`defense`、`model_role`、`BU`、`U@A`、`ASR`、`completed`、`errors`、`denied`、`judge_calls`、`runtime_seconds`、`data_provenance`、`metric_definition`。

## Agent Extensions

### SubAgent

- **code-explorer**
- Purpose: 映射原版 `Blueprinting` runner、AgentDojo/ASB 数据适配逻辑与匿名 `toolfence` 严格 profile 的接入点，确认最小修改面。
- Expected outcome: 给出可验证的文件路径、函数签名、防御插入位置和潜在不兼容点，避免盲目改动核心机制。