# AXIOM 对齐训练路线（2026-04-23）

## 1. 当前判断

当前阶段不宜继续把多个数据集直接混合做统一训练主线。更稳妥的路线是先把评分语义严格对齐到 AXIOM，再逐步扩充数据源。

项目的主任务是代码打分，评论文本只作为分数证据和可解释性补充。因此训练和评测都应围绕 AXIOM 0-5 这一套 ordinal grade 展开，而不是围绕长评论文本本身展开。

## 2. 统一语义

AXIOM 0-5 的核心不是“好坏主观印象”，而是“把代码修到可部署状态需要多少 effort”。

| AXIOM 分 | 语义 |
|---:|---|
| 5 | Production-ready，无需修改 |
| 4 | 功能正确，仅需小幅代码质量调整 |
| 3 | 功能正确，但需要较大质量重构 |
| 2 | 功能有缺陷，小修可恢复 |
| 1 | 功能有缺陷，需要较大修复 |
| 0 | 严重不匹配或根本错误，重写比修复更划算 |

其中最重要的边界是：

- `0-2`：功能错误或不匹配
- `3-5`：功能正确

当前主要问题不是“模型不会输出一个数字”，而是“模型没有稳定学会这条边界以及边界内的分级语义”。

## 3. 数据使用原则

### 3.1 当前主线

- `CodeCriticBench`：训练主源，先映射到 AXIOM 语义。
- `AXIOMBench`：外部对齐锚点，不直接混入当前这一轮主训练。

这样做的目的不是追求最优分数，而是回答一个更基础的问题：

`CodeCriticBench -> AXIOM` 这一套映射是否足够可靠，能否让模型在未见过的 AXIOM 样本上保持同样的评分语义。

### 3.2 后续原则

以后新接入数据集时，先判断它提供的是哪一类监督：

- `exact`：可以直接映射为 AXIOM 精确等级
- `interval`：只能确定在 `0-2` 或 `3-5`
- `pairwise`：只能表示 A 比 B 更好

不要再直接把不同来源的分数线性归一化后混在一起训练。

## 4. 训练与评测分工

严格来说，AXIOM 不应整体被反复拿来调参。更规范的做法是：

1. `CodeCriticBench` 作为主训练源。
2. `AXIOM-dev` 用于映射策略开发。
3. `AXIOM-test` 作为最终保留测试集。

当前为了尽快看清对齐效果，先执行一个工程上更简单的版本：

- 训练：只用 CodeCritic 的 AXIOM 映射样本
- 外评：用 AXIOM clean heldout（去掉 0 分）比较 `base direct`、`trained direct`、`trained value-rerank`

明天如果这条线有效，再把 AXIOM 正式拆成 dev/test。

## 5. 今晚的过夜任务

过夜任务目标不是“最终最优结果”，而是得到一个更干净、可解释的对齐实验：

1. 使用 `Qwen/Qwen3-8B`
2. 使用完整 `4096` token review prompt，尽量不截断样本
3. 训练集只取 CodeCritic 映射后的 `AXIOM 1-5`
4. 每档平衡采样，避免训练集均值偏低
5. 训练后立即跑 AXIOM clean heldout 外评

这样明天看到的结果可以直接回答三件事：

1. CodeCritic 的 AXIOM 映射是否足够一致
2. 8B 全上下文训练是否比之前小模型/短上下文更稳
3. value-rerank 在外部 AXIOM 样本上究竟是帮助还是拖累

## 6. 结果判读方式

明天优先看以下指标：

- `grade_mae`
- `boundary_acc`
- `low_grade_false_positive_rate`
- `high_grade_false_negative_rate`

其中最重要的是：

- 是否继续低估正确代码
- 是否真的学稳了 `2/3` correctness boundary

如果 `trained direct` 已经比 `base direct` 稳定，而 `value-rerank` 仍然拉低外评结果，那么下一步就应优先修正 value head，而不是继续盲目扩数据。
