# toolfence：匿名源码库

本代码库基于 ToolFence 中的现有实现整理、修改而成，包含策略编译器、参数来源检查、动态能力授权、持久化事务账本、针对性测试。代码库不包含非公开数据、凭据、模型权重。

## 代码结构与论文方法

| 位置 | 对应机制 |
|---|---|
| `src/toolfence/guarded_net.py`、`transaction_net.py` | 工作流阶段、能力和证据的事务账本；原子化预留、可信回执、动态授权提案与任务内授权 |
| `src/toolfence/semantic/` | 隔离的策略编译器、类型化工具清单和能力、具体参数及来源检查 |
| `src/toolfence/hybrid/` | 确定性快速路径、动态能力授权和跨会话能力形状缓存 |
| `src/toolfence/runtime.py`、`colored_runtime.py`、`agentdojo_defense.py` | 智能体运行时及最终 guarded ledger 执行边界 |
| `src/toolfence/judge/`、`core/`、`foundation/` | 隔离评审器、共享监控逻辑和基础授权结构 |
| `experiments/` | 不依赖模型的宿主沙箱机制示例 |
| `tests/unit/` | 事务、参数来源、动态授权和论文模式回归测试 |

所有项目内部模块均收拢在 `src/toolfence/` 下；`agentdojo` 仅指外部基准测试框架。

### 与论文对应的修改

- `ToolFenceAuthorizationBoundary` 要求控制器提供固定的工具目录、`ToolManifest`、策略版本，以及包含宿主适配器、身份和评审器的 `GuardedTransactionNet`。智能体可见的工具描述不能提供授权元数据。
- `SemanticPolicyCompiler(pinned_manifest=...)` 检查固定清单与可信工具目录的名称、参数是否完全对应，并跳过由模型生成工具效果分类的步骤。模型给出的操作、效果、复用性和来源标签，由控制器清单覆盖。
- 在 `strict_capabilities=True` 的论文模式下，能力授权失败会拒绝调用，不会退回到逐次调用的 LLM 放行。批准的新能力先加入监控器，具体调用随后重新接受确定性检查。
- 自动读取只适用于宿主已审核、无外部披露和状态修改的本地读取，并要求所有具体参数都由用户请求证明。读取会先获得有界能力，再经过常规的预留与提交路径；其他读取不会自动绕过监控器。
- 论文模式下的 `DERIVED` 绑定必须指定一个来源工具的一个输出字段。来源含糊或只有非结构化文本时，不能据此签发敏感参数的能力；已声明字段也不能退回匹配其他字段。
- 跨会话缓存只保存不含具体值的批准签名，并按控制器策略版本隔离。新任务的参数值仍需重新绑定和核验。

## 安装与验证

需要 Python 3.10 或更新版本。在代码库根目录执行：

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests/unit
```

安装后使用 `import toolfence` 导入；源码采用 `src/toolfence/` 布局。

完整测试还会导入上游 **AgentDojo** 的执行边界接口，因此需要在环境中安装该框架，或通过 `PYTHONPATH` 提供它。事务账本和动态授权提案本身不依赖 AgentDojo 或在线模型服务。

在新的输出目录运行宿主沙箱示例：

```bash
python experiments/run_guarded_net_cases.py --output output/guarded-example
python experiments/run_refinement_cases.py --output output/refinement-example
```

这些示例不会调用模型或真实外部工具。输出只验证机制行为，不代表提示注入攻击成功率。

## 接入可信宿主

1. 根据宿主维护的工具模式建立可信工具目录和 `ToolManifest`。清单必须明确每个工具的操作、效果，以及每个参数是否影响权限。只有经审核的本地读取才能设置 `reviewed_local_read=True`，同时必须保持 `external_disclosure=False` 和 `mutates_state=False`。
2. 使用固定清单创建 `SemanticPolicyCompiler(architect_backend, pinned_manifest=manifest)`。隔离的策略编译器只能看到经过认证的用户请求和可信工具目录，不能接收检索页面或工具输出。为运行时能力授权另建隔离的 `RuntimeJudge`。
3. 对每个经过认证的任务，由宿主签发精确的 `TaskContract`，在 `GuardedTransactionNet` 中配置 `TransitionRule` 和所需的 `EvidenceStep`，并注册 `HostToolBinding`。阶段规则、参数编码、身份、预算、有效期和回执证明均由宿主控制。
4. 使用 `trusted_catalog`、`transaction_net`、`host_bindings`、`host_actor`、`host_reviewer` 和 `policy_revision` 创建 `ToolFenceAuthorizationBoundary`。其默认运行时为 `TransactionalColoredToolFenceRuntime`，因此具体调用既要通过语义监控器，也要通过最终的事务账本。
5. 如果缺少精确的任务合同，`GuardedBoundary(refine_missing=True)` 只会记录**待审**的 `RefinementProposal`，不会授予权限。宿主可调用 `review_refinement`，由隔离评审器审核，并提供宿主决定的预算及新任务参数验证。批准后，智能体必须重新提交调用。执行使用可信适配器的回执；结果不明的操作保持占用，直至完成对账。

`issue`、`configure`、`ingest`、`activate`、`review_refinement`、`reconcile` 等控制器接口不能暴露为智能体工具。模型输出或工具结果本身不构成可信证明。

## 匿名性

发布包不包含版本历史、作者或机构信息、本机路径、密钥、本地账本、缓存及非公开实验材料。测试中的邮箱和对象名称均为虚构示例。
