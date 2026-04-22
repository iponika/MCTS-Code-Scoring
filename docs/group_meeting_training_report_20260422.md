# 代码评分项目组会汇报提纲：模型训练进展与评测设计

## 1. 概况

目前的目标是让模型对代码质量给出更稳定、更可信的分数；评论文本仅作为分数的证据和可解释性补充。

当前阶段已经完成从数据生成、训练到 value-guided 推理的最小闭环。

现阶段的正向结果主要是在 CodeCritic 同分布保留测试集上，trained direct 相对 base direct 有提升。使用 response_mean_value 的 value-guided MCTS 在 3 次重复中把 MAE 从 1.450 降到 1.089，boundary accuracy 从 0.775 提升到 0.804。

但在 AXIOM 保留测试集上结果是负面的，trained value-guided MCTS 反而更差，主要问题是格式稳定性下降和对正确代码过度低估。这说明当前方法还没有证明 MCTS 的通用评分能力，只证明了同分布训练和 value-guided 推理有潜力。

接下来需要着重提升泛化能力，并设计更公平、更可信的评测。计划在解决泛化问题后进行规模化训练，得到初步的最终结果。

## 2. 当前训练方式

### 2.1 统一评分标准

项目内部采用 AXIOM 0-5 ordinal grade 作为统一评分语义。它以“把代码修复至可部署状态所需工作量”为核心：

| AXIOM 分 | 语义 |
|---:|---|
| 5 | Production-ready，无需修改 |
| 4 | 功能正确，仅需小幅代码质量调整 |
| 3 | 功能正确，但需要较大质量重构 |
| 2 | 功能有缺陷，小修可恢复 |
| 1 | 功能有缺陷，需要较大修复 |
| 0 | 严重不匹配或根本错误，重写比修复更划算 |

关键设计是把 correctness 作为主轴：0-2 表示功能不正确或不匹配，3-5 表示功能正确；代码质量和修复工作量只在两侧内部细分。

### 2.2 数据来源

目前主要使用：

| 数据源 | 当前定位 | 标签处理 |
|---|---|---|
| CodeCriticBench | 主训练/评测数据之一 | 根据 correctness + 原始 score 映射到 AXIOM 0-5 |
| AXIOMBench | 统一评分语义锚点 | 原始 0-5 直接作为 AXIOM grade |
| Code-DiTing | 弱正确性监督 | 只作为 0-2 / 3-5 区间标签，不强行映射成精确分 |
| CodeJudgeBench | 相对排序监督候选 | 当前受显存和 batch-local pairwise 限制，短期低权重使用 |

当前阶段最可靠的是 CodeCritic + AXIOM 两类精确或近似精确标签。Code-DiTing 和 CodeJudgeBench 更适合作为辅助正确性/排序监督，不能直接当作绝对评分标签。之后需要进一步增加数据集和对齐标签。

## 3. 已采取且有效的技术选择

### 3.1 用 AXIOM ordinal grade 替代简单 0-100 回归

简单把不同数据集分数线性映射到 0-100 会混淆不同数据集的评分语义。AXIOM 的优点是先确定功能正确性边界，再讨论修复成本，更适合代码评分任务。

目前这项选择带来的直接收益是：可以统一 CodeCritic、AXIOM、Code-DiTing、CodeJudgeBench 的标签语义，并把评测指标拆成 MAE、boundary accuracy、low-FP、high-FN，而不是只看一个平均分误差。

### 3.2 去掉模型自评分，改为客观标签和 verifier 约束

早期方案考虑让模型给自己的思考节点打分，但这会把基础模型的偏见直接写进训练标签。当前已经去掉 self-judge reward，改为使用数据集标签、测试/正确性字段、verifier 检查作为监督来源。

这更符合训练数据生成的要求：节点 value 不是“模型觉得自己好不好”，而是“这条推理是否更接近客观 AXIOM 标签和可验证证据”。

## 4. 当前主要实验结果

### 4.1 有监督 MCTS 数据生成

中等规模 378 条、六个分数档均衡的监督 MCTS 数据生成实验：

| 方法 | count | valid_rate | MAE | median_err | boundary_acc | Pearson proxy |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 378 | 0.622 | 1.136 | 1.000 | 0.715 | 0.581 |
| Supervised MCTS | 378 | 1.000 | 0.746 | 0.000 | 0.754 | 0.737 |

可见如果有较可靠的客观 reward，MCTS 能生成更稳定的评分路径和训练数据。

### 4.2 CodeCritic held-out 上的训练和 value-guided 推理

3 次 repeat 后，使用 `response_mean_value` 的 value-guided MCTS 平均结果：

| 方法 | valid_rate | MAE | boundary_acc | low-FP | high-FN | unsupported |
|---|---:|---:|---:|---:|---:|---:|
| Base Direct | 1.000 | 1.450 | 0.775 | 0.025 | 0.200 | 0.100 |
| Trained Direct | 0.950 | 1.342 | 0.789 | 0.053 | 0.158 | 0.053 |
| Value-Guided MCTS mean, 3-run avg | 0.942 | 1.089 | 0.804 | 0.125 | 0.071 | 0.018 |

在 CodeCritic 同分布 held-out 上，训练后的 direct 模式优于 base direct；value-guided MCTS 进一步提升 MAE、boundary accuracy、high-FN 和 unsupported evidence。但 low-FP 变高，说明模型有把正确代码过度打低的倾向。

### 4.3 AXIOM 保留数据集迁移

AXIOM 30 条 held-out，六个分数档均衡，并排除了训练混合数据中出现过的 AXIOM 样本。

| 方法 | valid_rate | MAE | boundary_acc | low-FP | high-FN | unsupported |
|---|---:|---:|---:|---:|---:|---:|
| Base Direct | 0.633 | 1.421 | 0.579 | 0.368 | 0.053 | 0.000 |
| Trained Direct | 0.467 | 1.429 | 0.571 | 0.357 | 0.071 | 0.000 |
| Trained Value-Guided MCTS | 0.267 | 1.625 | 0.500 | 0.375 | 0.125 | 0.000 |

可见目前跨数据集泛化失败。当前训练虽然接入了 AXIOM，但模型没有稳定学会 AXIOM 的跨数据集评分语义，特别是容易把正确代码打到 0-2。

## 5. 当前技术难点

### 5.1 跨数据集评分语义未对齐

CodeCritic 和 AXIOM 都能映射到 AXIOM 0-5，但它们的样本形态和评分来源不同。模型在 CodeCritic held-out 上有效，不代表学会了通用代码评分标准。
最终模型需要的还是可泛化的能力，这需要找到更多的评分数据集进行训练，并设计更可信的跨数据集 held-out，证明模型学到的是评分原则，而不是某个数据集的风格。

### 5.2 value-guided MCTS 打分偏低

value-guided MCTS 能降低 high-FN，也就是减少把错误代码评高；但 low-FP 上升，说明它更容易把正确代码评低。

### 5.3 长样本训练和显存限制

代码评审样本包括需求、代码、评分标准、证据。4090 单卡在 Qwen3.5-9B + value head + LoRA 下，加载模型就要2G多，加载 1152 token 已经很紧张。而普通双卡 DDP 不能分担样本显存，因此不能简单通过“两张卡一起跑”解决。
后续应该换个显存更大的显卡，还是改用更小的模型？

## 6. 当前评测方案

### 6.1 Baseline 和对照组

同模型对照组如下：

| 方法 | 作用 |
|---|---|
| Base Direct | 基础模型直接评分 |
| Traditional Trained | 数据集直接进行传统训练，再直接评分 |
| Trained Direct | policy model 训练后直接评分 |
| Trained Value-Guided MCTS | value model 评分引导改善 policy 输出 |

为避免不公平，Direct 和 MCTS 应控制总 token 预算或总推理次数相等，进行对照。

### 6.2 测试数据集

当前计划用于测试的数据集：

| 测试集 | 用途 |
|---|---|
| CodeCritic held-out | 同分布效果 |
| AXIOM held-out | AXIOM 语义泛化 |
| Code-DiTing held-out | 功能正确性二分类边界 |
| CodeJudgeBench held-out | 正负答案相对排序 |

### 6.3 评测指标

用以下指标，评测不同模型在各种训练和推理方式下，在不同数据集测试中得到的如下指标，以确认MCTS框架的效果：

| 指标 | 含义 |
| MAE | AXIOM grade 平均误差 |
| median_err | 抗极端值的误差 |
| boundary_acc | 是否判断对功能正确性边界 |
| low-FP | 正确代码被打到 0-2 的比例 |
| high-FN | 错误代码被打到 3-5 的比例 |

## 7. 问题与展望

希望请教的问题：

1. 什么样的对照实验最能证明项目有效：同数据集 or 同类数据集提升是否足够，还是必须证明任意数据集都提升？
2. AXIOM 0-5 是否适合作为最终统一评分标准，还是应该拆成 correctness classifier + quality scorer 两阶段模型？
3. 最终报告是否应该强调 value-guided MCTS 的推理收益，还是强调 MCTS 生成数据 + value head 训练的完整路线？

接下来的计划：

统计推理成本，寻找压缩成本的方法
……
