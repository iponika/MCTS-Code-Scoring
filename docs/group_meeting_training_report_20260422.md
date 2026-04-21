# 代码评分项目组会汇报提纲：模型训练进展与评测设计

## 1. 汇报主线

本项目的目标不是生成代码评审文本，而是让模型对代码质量给出更稳定、更可信的分数；评论文本主要作为分数的证据和可解释性补充。

当前阶段已经完成从数据生成、训练到 value-guided 推理的最小闭环。比较明确的进展是：在 CodeCritic 同分布 held-out 上，LoRA + value head 训练和 value-guided MCTS 推理能带来可观提升；但在 AXIOM 跨数据集 held-out 上，当前方法还没有泛化成功，主要问题是格式稳定性和对正确代码过度低估。

建议明天汇报时把结论控制在这个范围内：项目已经跑通训练闭环，并在同分布评测上有正向结果；跨数据集泛化仍是主要技术难点，需要讨论更有说服力的实验设计。

## 2. 当前训练路线

### 2.1 统一评分标准

项目内部采用 AXIOM 0-5 ordinal grade 作为统一评分语义。它不是普通线性分数，而是以“把代码修复至可部署状态所需工作量”为核心：

| AXIOM 分 | 语义 |
|---:|---|
| 5 | Production-ready，无需修改 |
| 4 | 功能正确，仅需小幅代码质量调整 |
| 3 | 功能正确，但需要较大质量重构 |
| 2 | 功能有缺陷，小修可恢复 |
| 1 | 功能有缺陷，需要较大修复 |
| 0 | 严重不匹配或根本错误，重写比修复更划算 |

关键设计是把 correctness 作为主轴：0-2 表示功能不正确或不匹配，3-5 表示功能正确；代码质量和修复工作量只在两侧内部细分。

### 2.2 数据来源和标签对齐

目前主要使用：

| 数据源 | 当前定位 | 标签处理 |
|---|---|---|
| CodeCriticBench | 主训练/评测数据之一 | 根据 correctness + 原始 score 映射到 AXIOM 0-5 |
| AXIOMBench | 统一评分语义锚点 | 原始 0-5 直接作为 AXIOM grade |
| Code-DiTing | 弱正确性监督 | 只作为 0-2 / 3-5 区间标签，不强行映射成精确分 |
| CodeJudgeBench | 相对排序监督候选 | 当前受显存和 batch-local pairwise 限制，短期低权重使用 |

当前阶段最可靠的是 CodeCritic + AXIOM 两类精确或近似精确标签。Code-DiTing 和 CodeJudgeBench 更适合作为辅助正确性/排序监督，不能直接当作绝对评分标签。

### 2.3 训练方式

已跑通的训练方式是 Qwen3.5-9B 上的 LoRA + value head：

| 项 | 当前实现 |
|---|---|
| Base model | Qwen3.5-9B |
| Policy 训练 | 对 `<review>` JSON 输出做 supervised fine-tuning |
| Value 训练 | 对样本或 MCTS 路径节点的 AXIOM value 做回归/辅助监督 |
| 参数高效微调 | LoRA |
| value-guided 推理 | policy 生成候选，value head 给候选评分并选择 |
| 主要训练脚本 | `model_training/src/magicoder/train_multi.py` |

训练命令由 workflow 脚本封装，主要入口包括：

```bash
data_collection/scripts/run_report_pilot_training.sh
data_collection/scripts/run_cross_dataset_review_eval.sh
data_collection/scripts/run_principle_generalization_eval.sh
```

其中 `run_report_pilot_training.sh` 产出了当前最可展示的 CodeCritic held-out 正结果；`run_cross_dataset_review_eval.sh` 暴露了跨数据集负结果；`run_principle_generalization_eval.sh` 是正在进行的下一版对齐实验。

## 3. 已采取且有效的技术选择

### 3.1 用 AXIOM ordinal grade 替代简单 0-100 回归

简单把不同数据集分数线性映射到 0-100 会混淆不同数据集的评分语义。AXIOM 的优点是先确定功能正确性边界，再讨论修复成本，更适合代码评分任务。

目前这项选择带来的直接收益是：可以统一 CodeCritic、AXIOM、Code-DiTing、CodeJudgeBench 的标签语义，并把评测指标拆成 MAE、boundary accuracy、low-FP、high-FN，而不是只看一个平均分误差。

### 3.2 去掉模型自评分，改为客观标签和 verifier 约束

早期方案考虑让模型给自己的思考节点打分，但这会把基础模型的偏见直接写进训练标签。当前已经去掉 self-judge reward，改为使用数据集标签、测试/正确性字段、verifier 检查作为监督来源。

这更符合训练数据生成的要求：节点 value 不是“模型觉得自己好不好”，而是“这条推理是否更接近客观 AXIOM 标签和可验证证据”。

### 3.3 verifier correction 改为 value-only

我们尝试过让模型模仿 verifier 修正后的文本，但这种“突变式修正”容易伤害 policy 的自然输出。后来改成 value-only：如果 verifier 发现 unsupported evidence，不强迫 policy 学一段修正文案，只把该路径作为 value 负样本。

实验上，value-only verifier 对证据可靠性和功能边界有帮助，但对 MAE 不稳定。因此它适合作为质量约束，不应被包装成全面提升评分准确率的模块。

### 3.4 value-guided 选择从 `last_value` 改为 `response_mean_value`

旧版只取最后 token 的 value，容易让同一步思考前半段的低置信信号被最后 token 掩盖。改成 response token 平均 value 后，CodeCritic held-out 上效果明显更好。

CodeCritic 40 条 held-out 的消融结果：

| score_key | valid_rate | MAE | boundary_acc | low-FP | high-FN | unsupported |
|---|---:|---:|---:|---:|---:|---:|
| `last_value` baseline | 0.950 | 1.316 | 0.737 | 0.158 | 0.105 | 0.026 |
| `response_mean_value` | 0.925 | 1.081 | 0.757 | 0.162 | 0.081 | 0.027 |
| `response_conservative_value` | 0.975 | 1.410 | 0.667 | 0.231 | 0.103 | 0.000 |

结论：`response_mean_value` 是目前最合理的默认选择；过度保守的 value 聚合会降低 MAE 和 boundary accuracy。

### 3.5 训练资源上的保守选择

两张 4090 不能通过普通 DDP 自动合并单样本显存。为了避免再次 OOM，目前训练采用单卡、batch size 1、gradient accumulation 的保守方案。双卡更适合并行跑评测或不同实验，而不是直接解决单个长样本显存问题。

最新改动是增加短训练 prompt：训练时不再重复完整 MCTS step prompt，而使用更紧凑的评分 prompt。它把训练样本保留量从上一次的 130/1360 提高到 640/1360，仍有改进空间。

## 4. 当前主要实验结果

### 4.1 有监督 MCTS 数据生成

中等规模 378 条、六个分数档均衡的监督 MCTS 数据生成实验：

| 方法 | count | valid_rate | MAE | median_err | boundary_acc | Pearson proxy |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 378 | 0.622 | 1.136 | 1.000 | 0.715 | 0.581 |
| Supervised MCTS | 378 | 1.000 | 0.746 | 0.000 | 0.754 | 0.737 |

这说明：如果有较可靠的客观 reward，MCTS 能生成更稳定的评分路径和训练数据。但这不是最终部署形态，因为它使用了标签监督。

### 4.2 CodeCritic held-out 上的训练和 value-guided 推理

40 条 CodeCritic held-out 确定性评测：

| 方法 | valid_rate | MAE | median_err | boundary_acc | low-FP | high-FN | unsupported |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Direct | 1.000 | 1.450 | 1.000 | 0.775 | 0.025 | 0.200 | 0.100 |
| Trained Direct | 0.950 | 1.342 | 1.000 | 0.789 | 0.053 | 0.158 | 0.053 |
| Value-Guided MCTS, old `last_value` | 0.950 | 1.316 | 1.000 | 0.737 | 0.158 | 0.105 | 0.026 |

3 次 repeat 后，使用 `response_mean_value` 的 value-guided MCTS 平均结果：

| 方法 | valid_rate | MAE | boundary_acc | low-FP | high-FN | unsupported |
|---|---:|---:|---:|---:|---:|---:|
| Base Direct | 1.000 | 1.450 | 0.775 | 0.025 | 0.200 | 0.100 |
| Trained Direct | 0.950 | 1.342 | 0.789 | 0.053 | 0.158 | 0.053 |
| Value-Guided MCTS mean, 3-run avg | 0.942 | 1.089 | 0.804 | 0.125 | 0.071 | 0.018 |

可汇报结论：在 CodeCritic 同分布 held-out 上，训练后的 direct 模式优于 base direct；value-guided MCTS 进一步提升 MAE、boundary accuracy、high-FN 和 unsupported evidence。但 low-FP 变高，说明模型有把正确代码过度打低的倾向。

### 4.3 AXIOM 跨数据集 held-out 负结果

AXIOM 30 条 held-out，六个分数档均衡，并排除了训练混合数据中出现过的 AXIOM 样本。

严格 `<review>` JSON 解析：

| 方法 | valid_rate | MAE | boundary_acc | low-FP | high-FN | unsupported |
|---|---:|---:|---:|---:|---:|---:|
| Base Direct | 0.633 | 1.421 | 0.579 | 0.368 | 0.053 | 0.000 |
| Trained Direct | 0.467 | 1.429 | 0.571 | 0.357 | 0.071 | 0.000 |
| Trained Value-Guided MCTS | 0.267 | 1.625 | 0.500 | 0.375 | 0.125 | 0.000 |

宽松解析，即只要输出中能抽到 `"axiom_grade": N` 就计入：

| 方法 | parsed | MAE | boundary_acc | low-FP | high-FN |
|---|---:|---:|---:|---:|---:|
| Base Direct | 28/30 | 1.321 | 0.607 | 0.357 | 0.036 |
| Trained Direct | 26/30 | 1.615 | 0.577 | 0.385 | 0.038 |
| Trained Value-Guided MCTS | 29/30 | 1.966 | 0.448 | 0.483 | 0.069 |

可汇报结论：跨数据集泛化失败，不能作为正向结果展示。当前训练虽然接入了 AXIOM，但模型没有稳定学会 AXIOM 的跨数据集评分语义，特别是容易把正确代码打到 0-2。

## 5. 当前技术难点

### 5.1 跨数据集评分语义未对齐

CodeCritic 和 AXIOM 都能映射到 AXIOM 0-5，但它们的样本形态和评分来源不同。模型在 CodeCritic held-out 上有效，不代表学会了通用代码评分标准。

需要请教的问题：怎样设计更可信的跨数据集 held-out，证明模型学到的是评分原则，而不是某个数据集的风格？

### 5.2 value-guided MCTS 有保守偏置

value-guided MCTS 能降低 high-FN，也就是减少把错误代码评高；但 low-FP 上升，说明它更容易把正确代码评低。这与代码评分中的风险偏好有关：安全场景可能更接受保守，但评分任务需要校准。

需要请教的问题：代码质量评分应该优化 MAE、boundary accuracy，还是更重视 high-FN/low-FP 的非对称风险？

### 5.3 长样本训练和显存限制

代码评审样本天然长，包括需求、代码、评分标准、证据。4090 单卡在 Qwen3.5-9B + value head + LoRA 下，对 1152 token 已经很紧张。普通双卡 DDP 不能合并单样本显存，因此不能简单通过“两张卡一起跑”解决。

需要请教的问题：后续是否应投入时间做 FSDP、DeepSpeed ZeRO、QLoRA，还是先通过样本压缩、短 prompt 和小模型建立可靠趋势？

### 5.4 pairwise ranking 还没有稳定纳入

CodeJudgeBench 天然适合 pairwise loss，但当前实现是 batch-local pairwise：正负样本必须同 batch 出现，batch size 又受显存限制。因此短期为了保证绝对分数训练稳定，暂时降低或关闭 pairwise loss。

需要请教的问题：后续是否应单独实现 pairwise data collator，或把 pairwise 训练拆成独立阶段？

### 5.5 输出格式稳定性影响评测

AXIOM 跨数据集上 strict valid_rate 明显下降，说明当前 prompt 和训练还不足以保证 `<review>` JSON 稳定输出。评分任务里，如果分数解析失败，实际系统不可用。

需要请教的问题：最终评测应采用严格 JSON 解析，还是允许宽松抽取分数？我倾向主指标用严格解析，宽松解析只用于诊断模型“其实有没有给出分数”。

## 6. 当前评测方案

### 6.1 Baseline 和对照组

建议固定以下对照：

| 方法 | 作用 |
|---|---|
| Base Direct | 基础模型直接评分，是最重要 baseline |
| Trained Direct | 检验训练本身是否有效 |
| Base MCTS / Self-MCTS | 检验不训练时搜索是否有效，可作为辅助 |
| Trained Value-Guided MCTS | 检验 value model 是否能改善 policy 输出 |
| Oracle/Supervised MCTS 数据生成 | 只证明数据生成机制上限，不作为部署对照 |

为避免不公平，Direct 和 MCTS 应尽量控制总 token 预算或总推理次数；如果 MCTS 多采样，就需要增加 direct 多采样 rerank baseline。

### 6.2 数据集拆分

建议建立固定评测套件：

| 测试集 | 用途 |
|---|---|
| CodeCritic held-out | 同分布效果 |
| AXIOM held-out | 跨数据集、AXIOM 语义泛化 |
| Code-DiTing held-out | 功能正确性二分类边界 |
| CodeJudgeBench held-out | 正负答案相对排序 |

每次训练都报告同一套指标，避免只展示有利数据集。

### 6.3 指标

| 指标 | 目的 |
|---|---|
| valid_rate | 格式可用性 |
| MAE | AXIOM grade 平均误差 |
| median_err | 抗极端值的误差 |
| boundary_acc | 是否判断对功能正确性边界 |
| low-FP | 正确代码被打到 0-2 的比例 |
| high-FN | 错误代码被打到 3-5 的比例 |
| unsupported | 评论证据被 verifier 判定无依据的比例 |
| pairwise_acc | 正样本分数是否高于负样本 |
| repeat std | 多 seed 稳定性 |

其中 boundary_acc、low-FP、high-FN 对代码评分比纯 MAE 更重要，因为 AXIOM 的核心是 correctness boundary。

## 7. 希望组会上请教的问题

1. 什么样的对照实验最能证明项目有效：同分布提升是否足够，还是必须跨数据集也提升？
2. MCTS 方法是否必须与 direct 多采样进行同预算比较？如果 MCTS 用更多 token，应该如何公平控制推理成本？
3. AXIOM 0-5 是否适合作为最终统一评分标准，还是应该拆成 correctness classifier + quality scorer 两阶段模型？
4. 对代码评分任务，high-FN 和 low-FP 应该如何加权？是否需要根据应用场景定义非对称风险指标？
5. pairwise ranking 是否应作为后续重点？它可能比绝对分数更稳健，但需要额外训练实现。
6. 在两张 4090 的限制下，应该优先做工程上的 FSDP/QLoRA，还是先扩大小规模但严格控制的评测套件？
7. 最终报告是否应该强调 value-guided MCTS 的推理收益，还是强调 MCTS 生成数据 + value head 训练的完整路线？

## 8. 明天建议汇报话术

可以这样概括：

> 当前已经跑通了从 MCTS 数据生成、LoRA + value head 训练，到 value-guided MCTS 推理的完整闭环。我们把代码评分统一到 AXIOM 0-5，避免不同数据集分数线性混合。现阶段最明确的正结果是在 CodeCritic 同分布 held-out 上，trained direct 相对 base direct 有提升，使用 response_mean_value 的 value-guided MCTS 在 3 次重复中进一步把 MAE 从 1.450 降到 1.089，boundary accuracy 从 0.775 提升到 0.804，unsupported evidence 从 0.100 降到 0.018。
>
> 但跨数据集 AXIOM held-out 上结果是负的，trained value-guided MCTS 反而更差，主要问题是格式稳定性下降和对正确代码过度低估。这说明当前方法还没有证明通用评分能力，只证明了同分布训练和 value-guided 推理有潜力。下一步需要讨论更公平、更可信的评测设计，尤其是同预算对照、多 seed、跨数据集 held-out 和 pairwise ranking 指标。

## 9. 当前正在进行的改进实验

当前后台任务：`principle_generalization_shortprompt_20260421`。

目的：

- 增加更紧凑的 review training prompt，减少长样本被过滤。
- 以 AXIOM + CodeCritic 为主，Code-DiTing/CodeJudgeBench 低权重辅助。
- 暂时关闭 batch-local pairwise loss，先稳定绝对 AXIOM 分数。
- 使用 batch size 1 + gradient accumulation，降低单卡 OOM 风险。

目前状态：

- 训练数据构造：1360 条候选训练样本。
- 实际通过 1152 token 过滤：640 条。
- 相比上一版 130/1360 已明显改善，但仍说明长样本处理是瓶颈。
- 最终评测还未完成，不能作为明天报告的结论，只能作为“正在解决跨数据集泛化”的工作进展。

