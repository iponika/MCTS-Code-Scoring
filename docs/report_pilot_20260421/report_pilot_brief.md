# 代码评分项目阶段性实验汇报

## 基于 AXIOM 统一评分、MCTS 路径监督与 value-guided 推理的代码评分稳定性实验

## 当前结论

本项目目标是让通用大模型在代码评审任务中输出更稳定、更可靠的代码质量分数，文本评论只作为评分的辅助解释。当前内部统一采用 AXIOM 0-5 ordinal grade：0-2 表示功能不正确或不匹配，3-5 表示功能正确但修复/重构成本不同。

目前已跑通三类实验：

- 有监督 MCTS 数据生成在中等规模样本上明显优于直接询问，说明用标签监督搜索路径可以产生更稳定的评分数据。
- 基于 AXIOM + CodeCritic + MCTS value-only 数据的 480 step LoRA/value-head pilot，在 CodeCritic held-out 上提升了直接评分质量。
- value-guided MCTS 的选择指标很关键：从 `last_value` 改成 `response_mean_value` 后，CodeCritic held-out 上的 3 次重复结果更稳定；但 AXIOM 跨数据集测试没有取得正向结果，说明当前模型/提示/推理流程的跨数据集泛化还不足。

因此，当前可客观汇报的结论是：**在同分布 CodeCritic held-out 上，训练和 value-guided MCTS 有明确收益；在 AXIOM 跨数据集 held-out 上，当前流程失败，主要表现为格式稳定性下降和对正确代码过度低估。**

## 评分口径

AXIOM 0-5 用作内部统一评分标准：

| AXIOM 分 | 语义 |
|---:|---|
| 5 | Production-ready，无需修改 |
| 4 | 功能正确，仅需小幅代码质量调整 |
| 3 | 功能正确，但需要较大质量重构 |
| 2 | 功能有缺陷，小修可恢复 |
| 1 | 功能有缺陷，需要较大修复 |
| 0 | 严重不匹配或根本错误，重写比修复更划算 |

主要指标：

| 指标 | 含义 |
|---|---|
| valid_rate | 输出能否解析出合法 `<review>` JSON 与 AXIOM 分数 |
| MAE | 预测 AXIOM grade 与参考 grade 的平均绝对误差 |
| boundary_acc | 是否正确判断功能边界，即 0-2 vs 3-5 |
| low-FP | 参考分 >=3 但模型给 <3，表示把正确代码误判为错误 |
| high-FN | 参考分 <3 但模型给 >=3，表示把错误代码误判为正确 |
| unsupported | 评论证据被 verifier 判定为无依据的比例 |

## 实验 1：有监督 MCTS 数据生成，中等规模样本

这组实验使用标签监督的 MCTS 路径探索，与直接询问基础模型进行对比。样本量为 378，目标 grade 均衡分布，每档 63 条。

| 方法 | count | valid_rate | MAE | median_err | boundary_acc | Pearson proxy |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 378 | 0.622 | 1.136 | 1.000 | 0.715 | 0.581 |
| Supervised MCTS | 378 | 1.000 | 0.746 | 0.000 | 0.754 | 0.737 |

结论：有监督 MCTS 明显提升了格式稳定性、MAE 和相关性。这说明如果有客观标签或较强 reward，MCTS 路径探索可以生成质量更高的评分数据。

限制：这组是有监督评测，不能直接证明“无标签自监督 MCTS”有效。

结果来源：[supervised_medium summary](/data1/xianzhiwei/mcts-code-review/data_collection/review_mcts_runs/supervised_medium_20260418/summary.json)

## 实验 2：verifier value-only 数据质量对照

我们测试了 verifier correction 的训练方式。早期让模型模仿“修正文本”会伤害策略模型，因此后来改为 **value-only correction**：verifier 发现 unsupported evidence 时，只把该路径作为 value 负样本，不强迫 policy 模仿突变式修正文本。

单次 40 条 held-out CodeCritic 对照：

| 方法 | valid_rate | MAE | boundary_acc | low-FP | high-FN | unsupported |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1.000 | 1.300 | 0.675 | 0.200 | 0.125 | 0.025 |
| Value-only verifier | 0.975 | 1.154 | 0.821 | 0.103 | 0.077 | 0.000 |

3 次 repeat 均值：

| 方法 | valid_rate | MAE | boundary_acc | low-FP | high-FN | unsupported |
|---|---:|---:|---:|---:|---:|---:|
| Baseline mean | 0.942 | 1.327 | 0.727 | 0.194 | 0.079 | 0.034 |
| Value-only mean | 0.983 | 1.356 | 0.754 | 0.153 | 0.093 | 0.017 |

结论：value-only verifier 平均能改善 valid_rate、boundary_acc、low-FP 和 unsupported evidence，但 MAE 不稳定，不应宣称全面提升。它更像是数据质量和证据可靠性的约束，而不是直接提高全部评分准确率。

结果来源：[value-only ablation](/data1/xianzhiwei/mcts-code-review/model_training/src/output/review-eval-verifier_valueonly_ablation_20260420/comparison.json)，[value-only repeat3](/data1/xianzhiwei/mcts-code-review/model_training/src/output/review-eval-verifier_valueonly_repeat3_20260420/comparison.json)

## 实验 3：报告版 480 step pilot 训练，CodeCritic held-out

训练数据包含：

| 数据组成 | 数量 | 说明 |
|---|---:|---|
| AXIOM + CodeCritic 静态样本 | 1600 | 经过 1152 token 预算过滤 |
| MCTS value-only 路径样本 | 1317 | 主要用于 value head 监督 |
| 合计 | 2917 | policy 样本少，value-only 样本多 |

训练方式：

| 项 | 配置 |
|---|---|
| Base model | Qwen3.5-9B |
| 训练方式 | LoRA + value head |
| 训练步数 | 480 steps |
| value loss weight | 0.05 |
| 序列长度 | 1152 |
| 主要脚本 | `model_training/src/magicoder/train_multi.py` |

40 条 CodeCritic held-out 确定性评测：

| 方法 | valid_rate | MAE | median_err | boundary_acc | low-FP | high-FN | unsupported |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Direct | 1.000 | 1.450 | 1.000 | 0.775 | 0.025 | 0.200 | 0.100 |
| Trained Direct | 0.950 | 1.342 | 1.000 | 0.789 | 0.053 | 0.158 | 0.053 |
| Value-Guided MCTS, old `last_value` | 0.950 | 1.316 | 1.000 | 0.737 | 0.158 | 0.105 | 0.026 |

结论：训练后的 direct 模式相对 base direct 有稳定提升。旧版 value-guided MCTS 能降低 MAE、high-FN 和 unsupported evidence，但 boundary_acc 下降，low-FP 上升，说明它更保守。

结果来源：[report pilot comparison](/data1/xianzhiwei/mcts-code-review/model_training/src/output/review-eval-report_pilot_training_20260420/comparison.json)

## 实验 4：value-guided MCTS 选择指标消融

我们发现旧版 `last_value` 只取 response 最后一个 token 的 value，容易忽略同一段思考中间的低置信信号。因此测试了两种替代指标：

| score_key | valid_rate | MAE | boundary_acc | low-FP | high-FN | unsupported |
|---|---:|---:|---:|---:|---:|---:|
| `last_value` baseline | 0.950 | 1.316 | 0.737 | 0.158 | 0.105 | 0.026 |
| `response_mean_value` | 0.925 | 1.081 | 0.757 | 0.162 | 0.081 | 0.027 |
| `response_conservative_value` | 0.975 | 1.410 | 0.667 | 0.231 | 0.103 | 0.000 |

结论：`response_mean_value` 明显优于 `last_value`，因此已把默认 value-guided selection 改为 `response_mean_value`。`response_conservative_value` 虽然消除了 unsupported evidence，但过度惩罚低置信 token，导致 MAE 和 boundary_acc 变差。

结果来源：[score-key ablation](/data1/xianzhiwei/mcts-code-review/model_training/src/output/review-eval-value_score_key_ablation_20260421/comparison.json)

## 实验 5：CodeCritic held-out 上的 3 次 repeat

为了避免只报告单次好结果，我们对 `response_mean_value` 的 value-guided MCTS 做了 3 次不同 seed 的重复评测。每次仍为同一批 40 条 CodeCritic held-out。

| 方法 | valid_rate | MAE | boundary_acc | low-FP | high-FN | unsupported |
|---|---:|---:|---:|---:|---:|---:|
| Base Direct | 1.000 | 1.450 | 0.775 | 0.025 | 0.200 | 0.100 |
| Trained Direct | 0.950 | 1.342 | 0.789 | 0.053 | 0.158 | 0.053 |
| Value-Guided MCTS mean, 3-run avg | 0.942 | 1.089 | 0.804 | 0.125 | 0.071 | 0.018 |

3 次 repeat 标准差：

| 指标 | std |
|---|---:|
| valid_rate | 0.024 |
| MAE | 0.035 |
| boundary_acc | 0.049 |
| low-FP | 0.036 |
| high-FN | 0.014 |
| unsupported | 0.013 |

结论：在 CodeCritic held-out 上，`response_mean_value` 的 value-guided MCTS 平均表现最好，尤其 MAE、boundary_acc、high-FN、unsupported 都优于 direct baselines。但 low-FP 仍高于 Base Direct，说明仍有把正确代码过度打低的问题。

结果来源：[mean MCTS repeat comparison](/data1/xianzhiwei/mcts-code-review/model_training/src/output/review-eval-report_mean_mcts_repeat_20260421/comparison.json)

## 实验 6：AXIOM 跨数据集 held-out，负结果

为了测试跨数据集泛化，我们从 AXIOMBench 中构造了 30 条 held-out 样本，每个 AXIOM 分数档 5 条，并排除了训练混合数据中出现过的 231 个 AXIOM 样本。

严格 `<review>` JSON 解析结果：

| 方法 | valid_rate | MAE | boundary_acc | low-FP | high-FN | unsupported |
|---|---:|---:|---:|---:|---:|---:|
| Base Direct | 0.633 | 1.421 | 0.579 | 0.368 | 0.053 | 0.000 |
| Trained Direct | 0.467 | 1.429 | 0.571 | 0.357 | 0.071 | 0.000 |
| Trained Value-Guided MCTS | 0.267 | 1.625 | 0.500 | 0.375 | 0.125 | 0.000 |

宽松解析结果，即即使缺失 `</review>`，只要输出里出现 `"axiom_grade": N` 就抽取分数：

| 方法 | parsed | MAE | boundary_acc | low-FP | high-FN |
|---|---:|---:|---:|---:|---:|
| Base Direct | 28/30 | 1.321 | 0.607 | 0.357 | 0.036 |
| Trained Direct | 26/30 | 1.615 | 0.577 | 0.385 | 0.038 |
| Trained Value-Guided MCTS | 29/30 | 1.966 | 0.448 | 0.483 | 0.069 |

结论：AXIOM 跨数据集结果失败，不能作为正向效果展示。主要问题有两个：

- 格式稳定性差，特别是 value-guided MCTS 严格 valid_rate 只有 0.267。
- 评分风格迁移失败，模型在 AXIOM 上明显过度低估正确代码，low-FP 很高。

这说明当前训练虽然使用了部分 AXIOM 静态数据，但模型并没有稳定学会 AXIOM 的跨数据集评分语义。后续必须单独做 AXIOM held-out 的 prompt、训练配比和格式约束优化。

结果来源：[AXIOM v3 comparison](/data1/xianzhiwei/mcts-code-review/model_training/src/output/review-eval-axiom_report_eval_v3_20260421/comparison.json)

## 总体判断

目前最可靠的正向证据来自 CodeCritic held-out：

- Trained Direct 相对 Base Direct 有稳定提升。
- `response_mean_value` 的 Value-Guided MCTS 在 3 次重复中进一步提升 MAE、boundary_acc、high-FN 和 unsupported evidence。
- value-only verifier 对证据可靠性和功能边界有帮助，但对 MAE 不稳定。

目前最重要的负结果来自 AXIOM held-out：

- 跨数据集泛化没有跑通。
- 当前 value-guided MCTS 在 AXIOM 上比 direct 更差。
- 这说明 MCTS/value 机制不是天然提升，必须依赖良好的评分语义对齐、prompt 格式约束和训练数据配比。

## 组内汇报建议

建议如实汇报为：

> 我们已经在 CodeCritic 同分布 held-out 上看到训练和 value-guided MCTS 的稳定收益，尤其是 `response_mean_value` 替代 `last_value` 后，3 次重复平均 MAE 从 Base Direct 的 1.450 降到 1.089，boundary_acc 从 0.775 提升到 0.804，unsupported evidence 从 0.100 降到 0.018。但 AXIOM 跨数据集 held-out 上没有取得正向结果，说明当前方法还没有解决跨数据集评分语义泛化问题。下一步应重点优化 AXIOM-style 评分训练、格式约束和 value-guided MCTS 的保守偏置。

## 下一步计划

1. 先解决输出格式稳定性：对 AXIOM 和 CodeCritic 都统一使用更短 final-only JSON 模板，减少 invalid review。
2. 调整训练配比：增加 AXIOM held-out 风格但不泄漏测试样本的静态 value supervision，让模型更明确学习 0-5 ordinal semantics。
3. 拆分训练目标：policy 负责稳定输出 JSON 和评分解释，value head 主要学习 grade correctness，不让 value-guided MCTS 过度惩罚正确代码。
4. 在 AXIOM、CodeCritic、Code-DiTing 上分别建立固定 held-out 套件，后续每次训练都同时报告同分布和跨分布结果。

## 图表说明

已有图表仍对应最早的 report pilot：

- 图 1：[figure_main_metrics.svg](/data1/xianzhiwei/mcts-code-review/docs/report_pilot_20260421/figure_main_metrics.svg)
- 图 2：[figure_risk_metrics.svg](/data1/xianzhiwei/mcts-code-review/docs/report_pilot_20260421/figure_risk_metrics.svg)

建议后续新增一张“CodeCritic 正结果 vs AXIOM 负结果”的对比图，避免只展示单一数据集上的好结果。
