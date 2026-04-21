# 代码评分项目阶段性结果

## 建议标题

基于 MCTS 监督与价值建模的代码评分稳定性提升初步结果

## 一页结论版

本项目的目标是让通用大模型在代码评审任务中输出更稳定、更可靠的分数。为此，我们采用 AXIOM 0-5 作为统一内部评分标准，并在同一 backbone 上引入价值建模，使模型不仅输出分数，还能对推理路径进行打分和约束。

在当前阶段，我们先完成了一个“报告版 pilot”闭环：使用 AXIOM + CodeCritic 的静态监督数据，以及已有 MCTS 路径数据中的 value-only 标注，对 Qwen3.5-9B 进行了小规模微调，并在 CodeCriticBench 的 40 个 held-out 样本上进行确定性评测。结果表明，微调后的模型在直接评分模式下，平均绝对误差、功能边界判断和证据可靠性均优于基础模型；在 value-guided MCTS 推理模式下，模型进一步降低了 unsupported evidence 和高风险高分误判，但目前存在一定的保守偏置。

因此，当前最稳妥的阶段性结论是：训练后的模型已经能稳定改善直接代码评分；MCTS / value-guided 推理展示出进一步提升可信度的潜力，但其评分风格仍需要继续校准。

## 推荐正文表述

为了在代码评分任务上获得稳定、可比较的分数，我们将不同来源的数据统一映射到 AXIOM 0-5 评分语义，并基于这一评分语义训练模型的 value head。随后，我们构造了一个报告版 pilot 训练集：其中包含 1600 条经过 token 预算过滤的 AXIOM + CodeCritic 静态监督样本，以及 1317 条来自 MCTS 过程的 value-only 路径监督样本，总计 2917 条训练样本。基于这些样本，我们对 Qwen3.5-9B 进行了 480 step 的 LoRA + value head 微调。

在 40 条 held-out CodeCriticBench 样本上的确定性评测结果显示，微调后的直接评分相较于基础模型，平均绝对误差从 1.450 降低到 1.342，功能边界判断准确率从 0.775 提高到 0.789，unsupported evidence 比例从 0.100 降低到 0.053。这说明，训练后的模型已经能够更稳定地输出接近目标语义的分数，并减少无依据的评审证据。

进一步地，我们使用训练后的 policy + value 模型进行 value-guided MCTS 推理。该模式下，平均绝对误差进一步下降到 1.316，unsupported evidence 比例进一步下降到 0.026，高分漏报率也从 0.158 降低到 0.105。但与此同时，低分误判率上升到 0.158，说明当前的 value-guided 推理会更倾向于保守评分。也就是说，MCTS 推理当前更擅长提升“可信度”和“风险控制”，但还未在整体分数准确性上全面超过训练后的直接评分。

## 结果表

| 方法 | valid_rate | MAE | median_err | boundary_acc | low-FP | high-FN | unsupported |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Direct | 1.000 | 1.450 | 1.000 | 0.775 | 0.025 | 0.200 | 0.100 |
| Trained Direct | 0.950 | 1.342 | 1.000 | 0.789 | 0.053 | 0.158 | 0.053 |
| Trained Value-Guided MCTS | 0.950 | 1.316 | 1.000 | 0.737 | 0.158 | 0.105 | 0.026 |

## 口径建议

- 如果强调“当前最稳的提升”，主讲 `Base Direct -> Trained Direct`。
- 如果强调“MCTS 机制的潜力”，主讲 `Trained Direct -> Trained Value-Guided MCTS` 在 `MAE`、`high-FN`、`unsupported` 上的改善。
- 不建议把当前 MCTS 结果表述为“全面优于直接评分”，因为 `boundary_acc` 和 `low-FP` 仍有退化。

## 可直接口述的结论

现阶段我们已经证明两点。第一，基于统一 AXIOM 语义和小规模监督数据，模型微调后可以稳定提升直接代码评分质量。第二，value-guided MCTS 推理可以进一步减少 unsupported evidence，并更积极地压低高风险样本分数，说明“价值模型监督推理”的路线是成立的。下一步的工作重点不是重新证明这条路线是否有效，而是继续校准 MCTS 的评分风格，降低其保守偏置。

## 图表说明

- 图 1：[figure_main_metrics.svg](/data1/xianzhiwei/mcts-code-review/docs/report_pilot_20260421/figure_main_metrics.svg)
- 图 2：[figure_risk_metrics.svg](/data1/xianzhiwei/mcts-code-review/docs/report_pilot_20260421/figure_risk_metrics.svg)

## 数据与结果来源

- 训练结果汇总：[comparison.json](/data1/xianzhiwei/mcts-code-review/model_training/src/output/review-eval-report_pilot_training_20260420/comparison.json)
- 训练日志：[train_report.log](/data1/xianzhiwei/mcts-code-review/data_collection/review_mcts_runs/report_pilot_training_20260420/logs/train_report.log)
- 运行日志：[pipeline_20260420_200029.log](/data1/xianzhiwei/mcts-code-review/data_collection/review_mcts_runs/report_pilot_training_20260420/logs/pipeline_20260420_200029.log)
