---
name: reviewed manifest 生成接入与重跑
overview: 为 AgentDojo、ASB、MCPtox 生成人工可读的 JSON reviewed manifest，替换按工具名猜测的 pinned manifest 逻辑；先 smoke 验证效用恢复，再在服务器 tmux/screen 中重跑三个 100 子集并给出新旧对比。
todos:
  - id: inventory-manifest-targets
    content: 使用 [subagent:code-explorer] 盘点三个 benchmark 的工具集合和 manifest 接入点
    status: pending
  - id: create-json-manifests
    content: 生成 AgentDojo、ASB、MCPtox 三份 JSON reviewed manifest
    status: pending
    dependencies:
      - inventory-manifest-targets
  - id: add-loader-validator
    content: 实现 manifest loader、schema 校验和 fail-closed 规则
    status: pending
    dependencies:
      - inventory-manifest-targets
  - id: integrate-profile-runners
    content: 改造 paper_profile 和三个 runner 使用 reviewed manifest
    status: pending
    dependencies:
      - create-json-manifests
      - add-loader-validator
  - id: add-tests-readme
    content: 补充 manifest 单测、旧新对比汇总和运行文档
    status: pending
    dependencies:
      - integrate-profile-runners
  - id: smoke-in-screen
    content: 在服务器 detached screen 中跑 smoke 并修复效用问题
    status: pending
    dependencies:
      - add-tests-readme
  - id: full-rerun-compare
    content: 在 screen 中重跑三个 100 子集并生成新旧对比
    status: pending
    dependencies:
      - smoke-in-screen
---

## 用户需求

基于上一轮真实数据集评测中严格 profile 过度拒绝的问题，生成新的 reviewed manifest。manifest 必须是人工可读、可审计的 JSON 配置文件，每个 benchmark 一份；代码只读取配置，不再按工具名前缀猜测工具语义。生成并接入后，先在服务器 screen 或 tmux 中跑小样本 smoke，确认严格 profile 效用明显恢复，再重跑 AgentDojo、ASB、MCPtox 三个 100 子集，并给出新旧结果对比。

## 产品概述

新增三份 reviewed manifest：AgentDojo、ASB、MCPtox。每份配置明确列出可信工具、工具语义、参数敏感性、参数来源约束、允许使用的上下文和拒绝规则。未配置工具一律 fail-closed。三个 runner 改为按 benchmark 与 suite/server 上下文加载 manifest，严格 profile 使用 manifest 构造 pinned manifest 和 trusted catalog。

## 核心功能

- 提供 JSON manifest 格式和校验逻辑。
- 为 AgentDojo 四个 suite 配置 reviewed 工具清单。
- 为 ASB normal tools 配置 reviewed 规则，攻击工具明确 deny 或不进入 trusted catalog。
- 为 MCPtox 按 server 配置 clean tools，poisoned-only 工具明确 deny。
- 替换当前按工具名前缀推断 effect 和参数敏感性的逻辑。
- 保持 ToolFence 核心机制和严格 wrapper 不变。
- 先 smoke 验证效用恢复，再在 detached screen 或 tmux 中跑全量 100 子集。
- 输出新结果目录和旧结果对比。

## Tech Stack Selection

- 运行环境：Python 3.10，服务器 `/home/yjli/anaconda3/envs/piarena/bin/python`。
- 代码基线：`/home/yjli/ToolFence_anonymous3`。
- 配置格式：JSON，放在 `experiments/real_benchmarks/manifests/`。
- 评测模型：默认继续使用本地 Qwen3-8B，因为 Google Flash 免费额度已耗尽；Gemini 仅保留已有 smoke 对比结果。
- 服务：Qwen3-8B vLLM 端口 8000，已开启 tool-call 解析。
- 后台运行：服务器 screen 或 tmux，保证本地 Mac 断开后仍可继续。

## Implementation Approach

策略是把工具语义从运行时猜测迁移到 host 侧 JSON 配置。manifest 是唯一可信来源，未命中的工具不进入 trusted catalog，也不进入 pinned manifest。compiler 仍只负责生成计划，不能定义工具 effect、operation、参数敏感性或来源规则。

关键决策：

- 不再默认调用 `paper_profile.py::classify_tool`。该函数可保留为 deprecated legacy helper，但真实评测必须要求 reviewed manifest。
- `build_paper_profile` 增加 manifest 上下文参数，例如 benchmark、suite、server；runner 必须传入。
- trusted catalog 由 manifest 的工具集合生成，而不是 benchmark 暴露的全部工具。agent 仍可看到原 benchmark 工具，但严格 profile 下未配置工具不可执行。
- ASB attack tool 不进入 reviewed catalog；MCPtox poisoned-only tool 不进入 reviewed catalog。
- 参数级 `grounding_rule` 写入 JSON，作为审计和可解释性元数据；映射到 ToolFence 现有 binding 语义，不修改核心 Pydantic schema。
- 保留现有通用 `benchmark.sandbox.invoke` adapter，避免改动核心事务机制；通过 manifest 限制其授权范围。

性能：

- manifest 加载和校验为 O(tools)，开销可忽略。
- 主要瓶颈仍是本地 Qwen 推理；全量运行继续使用 checkpoint、小并发和 detached session。
- smoke 先验证 strict BU/U@A 是否从 0 恢复，避免全量跑完才发现配置错误。

## Implementation Notes

- 执行第一步先确认 SSH 可用。此前出现过 `Operation timed out`，若仍不可用，等待并重试，不绕过。
- 不把 API key 写入任何 manifest、代码、日志或结果文件。
- 保留旧结果目录 `results_real/final_100_qwen/`，新结果写入 `results_real/reviewed_manifest_100/`。
- 新增 manifest 必须记录版本、benchmark、policy revision、数据来源和生成依据。
- 每个 JSON manifest 必须包含 `fail_closed: true` 和 `unknown_tool_policy: deny`。
- manifest 加载器必须校验：
- 工具名唯一；
- effect 和 operation 不是 unknown；
- parameters 与工具 schema 顶层 properties 对齐；
- grounding_rule 属于允许集合；
- denylist 中的工具不会进入 trusted catalog。
- AgentDojo manifest 按 suite 隔离；ASB manifest 按 agent 或全局 normal/attack 规则隔离；MCPtox manifest 按 server 隔离。
- 如果 smoke 后 strict BU 仍为 0，先定位是 manifest 覆盖不足、参数 grounding 过严、还是 judge 过严，不直接进入全量重跑。
- 全量运行必须用 detached session，例如 `screen -dmS reviewed_full_ad`、`screen -dmS reviewed_full_asb`、`screen -dmS reviewed_full_mcp`。

## Architecture Design

数据流：

JSON manifest 配置
到 manifest loader 和 validator
到 reviewed tool catalog
到 pinned ToolManifest
到 ToolFenceAuthorizationBoundary
到 benchmark runner
到 smoke 或全量结果
到新旧结果对比报告

组件职责：

- `manifest_loader.py`：读取 JSON，校验 schema，按 benchmark 与上下文选择工具规则。
- `paper_profile.py`：用 reviewed manifest 构造 ToolManifest 和 trusted catalog，不再使用名称启发式作为默认。
- 三个 runner：传入 suite/server/agent 上下文，使用 manifest 过滤 trusted tools。
- `summarize_results.py`：汇总新结果，并加入旧结果对比字段。

## Directory Structure

```text
/home/yjli/ToolFence_anonymous3/
├── experiments/
│   └── real_benchmarks/
│       ├── manifests/
│       │   ├── agentdojo.json                  # [NEW] AgentDojo reviewed manifest。按 workspace、slack、banking、travel 列出可信工具、effect、operation、参数敏感性和 grounding_rule；未知工具拒绝。
│       │   ├── asb.json                        # [NEW] ASB reviewed manifest。normal tools 显式规则化，attack tools 在 denylist 或不进入 trusted catalog。
│       │   └── mcptox.json                     # [NEW] MCPtox reviewed manifest。按 server 列出 clean tools，poisoned-only tools 明确 deny。
│       ├── manifest_loader.py                  # [NEW] 加载和校验 JSON manifest，按 benchmark、suite、server、agent 上下文返回 reviewed tools 和审计信息。
│       ├── manifest_schema.py                  # [NEW] 定义允许字段、effect/operation、grounding_rule、context 白名单和校验函数。
│       ├── paper_profile.py                    # [MODIFY] 接入 reviewed manifest，移除默认 classify_tool 路径；build_paper_profile 增加 manifest 上下文参数；审计中记录 manifest_hash。
│       ├── run_agentdojo_100.py                # [MODIFY] 传入 suite_name 加载 AgentDojo manifest；trusted catalog 使用 reviewed tools。
│       ├── run_asb_100.py                      # [MODIFY] 传入 ASB agent 或全局上下文；attack tool 不进入 trusted catalog；记录 manifest_hash。
│       ├── run_mcptox_100.py                   # [MODIFY] 传入 server 上下文；trusted catalog 只使用 clean reviewed tools；poisoned-only tools deny。
│       ├── summarize_results.py                # [MODIFY] 支持新结果与旧结果目录对比，输出 BU/U@A/ASR/judge_calls 差异。
│       ├── README.md                           # [MODIFY] 说明 manifest 格式、加载方式、screen/tmux 跑法和新旧结果口径。
│       └── deprecated_heuristic_manifest.py    # [NEW, OPTIONAL] 保留旧 classify_tool 逻辑作为显式 legacy 对照，真实评测不得默认使用。
├── tests/
│   └── unit/
│       └── test_real_benchmark_adapters.py     # [MODIFY] 增加 manifest schema、fail-closed、覆盖度、benchmark 上下文加载和 paper_profile 接入测试。
└── results_real/
    ├── reviewed_manifest_smoke/                # [NEW] smoke 结果、checkpoint 和日志。
    └── reviewed_manifest_100/                  # [NEW] 全量 100 子集结果、summary.csv、REPORT.md、新旧对比。
```

## Key Code Structures

`manifest_loader.py` 应提供以下接口：

```python
def load_reviewed_manifest(benchmark: str) -> dict:
    ...

def reviewed_tools_for_context(
    benchmark: str,
    *,
    suite: str | None = None,
    server: str | None = None,
    agent: str | None = None,
) -> tuple[list[dict], dict]:
    ...
```

`paper_profile.py` 的关键函数签名应调整为：

```python
def build_paper_profile(
    backends: object,
    tools: list[dict],
    *,
    ledger_path: str,
    benchmark: str,
    manifest_context: dict,
    policy_revision: str | None = None,
    host_actor: str | None = None,
) -> PaperProfileBundle:
    ...
```

Manifest 单工具字段至少包含：

- `name`
- `effect`
- `operation`
- `resource`
- `reversible`
- `reviewed_local_read`
- `external_disclosure`
- `mutates_state`
- `parameters`
- `host_adapter`
- `allowed_in_attack_context`
- `grounding_rule`

## Agent Extensions

### SubAgent

- **code-explorer**
- 用途：只读探索服务器上三个 runner、现有 `paper_profile.py`、AgentDojo suite tools、ASB normal/attack tools、MCPtox clean/poisoned tools，生成 manifest 覆盖清单并验证所有接入点。
- 预期结果：给出每个 benchmark 需要写入 manifest 的工具集合、上下文键、runner 修改点和潜在不兼容项。