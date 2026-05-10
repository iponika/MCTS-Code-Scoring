# Methodology Figure Generation Prompt

Use the following prompt to generate a clean IEEE-style methodology figure. The figure must use English only.

```text
Draw a professional IEEE/ACM conference paper style methodology flowchart for a machine learning paper. The figure should be clean, compact, and easy to understand. Use a left-to-right layout. Arrows represent data flow. Use concise labels only. Do not include Chinese text.

The key visual emphasis should be on the MCTS-based data generation module. It should occupy more visual space than the other modules, because it is the central part of the method.

Overall pipeline:
Seed sample → MCTS-based review path exploration → Policy-quality training with LoRA → Quality-guided inference → Final structured assessment

Module 1: Seed Sample
Draw a small data-card icon, with a small code/document icon beside it.
Do not write a generic literal phrase such as “Problem + Code + Reference Grade”.
Instead, show a compact realistic example using short snippets. The text should be visibly truncated, not complete.

Use this example content:

Seed Sample
Task:
Generate a CSV file for 100 people, with Name, Age, Height, Weight, and append average Age/Height/Weight.

Candidate Code:
def task_func(filename):
    rows = [[f"Person_{i}", rand_age, rand_height, rand_weight] ...]
    write header, 100 rows, and average row

Reference Score:
4/5

The problem and code snippets should be short and schematic. Do not paste a full programming problem or full code listing into the figure.

Module 2: MCTS-based Review Path Exploration
This is the most important module. Draw it larger than the other modules.

Draw a search tree that starts from a black root node. Each parent node should have 2 or 3 children. Use circular nodes for intermediate reasoning states.

Use colors:
- Green nodes: high-quality reasoning states
- Red nodes: low-quality reasoning states
- Light gray nodes: neutral or unexplored states

Nodes on the same branch should have similar colors, because quality is propagated backward from leaf rewards.

Use square leaf nodes for final structured assessments. Each square leaf should contain a short example, not full JSON.

Example high-quality branch:
Intermediate node:
“Meets functional requirements; only minor style/complexity issues.”
Quality score: +0.82

Leaf:
Structured Assessment
Predicted Score: 4/5
Evidence: correct CSV generation; unnecessary lambda complexity
Quality Reward: +1.00

Example low-quality branch:
Intermediate node:
“Treats generated names like Person_i as a functional defect.”
Quality score: -0.30

Leaf:
Structured Assessment
Predicted Score: 2/5
Evidence: incorrectly claims name generation is defective
Quality Reward: -0.30

Optional bad terminal example:
Predicted Score: 5/5
Quality Reward: -1.00

Draw small backward arrows from square leaf nodes to their ancestor nodes. Label them:
Quality Backpropagation

The backward arrows should visually indicate that leaf rewards update upstream node quality estimates.

Do not overfill the tree with text. Use only 3 or 4 short node labels. The tree structure and colors should communicate the main idea.

Module 3: Policy-Quality Training with LoRA
Draw a large backbone model block with a simple LLM icon, labeled:
Backbone LLM

Attach two trainable modules:
Policy Head
Quality Head

Add a small blue label:
LoRA Fine-tuning

Show that the MCTS tree provides training data at the path level:
All explored paths → Quality Head training
Selected high-quality paths → Policy Head training

Use “quality” everywhere in the figure, not “value”. For example, write “Quality Head”, “Quality score”, and “Quality-guided inference”.

Module 4: Quality-Guided Inference
Draw an iterative loop with a small LLM/code-review icon.

At each iteration:
Policy Head proposes several candidates:
- reasoning step
- reasoning step
- final assessment

Quality Head scores candidates:
q = +0.65
q = -0.20
q = +0.91

Select the highest-quality candidate as the next step.

If the best candidate is a final structured assessment, stop.

Add a small optional branch:
Low confidence or high disagreement → Rethink

Final Output
Draw a final document/card icon labeled:
Final Structured Assessment
Predicted Score: 0–5/5
Supporting Evidence
Summary

Icon suggestions:
- Use a small code-file icon near Candidate Code.
- Use a neural network or chip icon near Backbone LLM.
- Use a magnifying glass or checklist icon near Structured Assessment.
- Use a branching-tree icon near MCTS.

Visual style:
- Use grayscale as the base style.
- Use green only for high-quality nodes.
- Use red only for low-quality nodes.
- Use blue only for trainable modules such as Policy Head, Quality Head, and LoRA.
- Use thin arrows and clear module boundaries.
- Keep labels short.
- Avoid decorative backgrounds.
- Avoid showing implementation details such as model names, GPU settings, file paths, exact JSON schemas, or hyperparameters.

Important wording constraints:
- Do not use the word “value” in the figure. Use “quality” instead.
- Do not write “Problem + Code + Reference Grade” as a literal label.
- Do not write “Review + Grade” as a literal label.
- Use “Reference Score: 4/5” for the seed example.
- Use quality scores in the range [-1, 1], such as +0.82, -0.30, and +1.00.
- Show grades as x/5, such as 4/5 or 2/5.
```

Notes for manual adjustment:

- The example is adapted from an AXIOM/BigCodeBench-style seed sample in which the task is to generate a CSV file for 100 people and append averages. The reference score is 4/5 because the code is functionally correct but has quality issues.
- The figure should not reproduce the full sample. The snippets are only anchors to prevent the image model from using abstract placeholder text.
- If the generated figure is too crowded, remove the optional bad terminal example first, then shorten the candidate code snippet.
