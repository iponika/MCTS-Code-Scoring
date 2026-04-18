from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type

from pydantic import field_validator

from mcts_math.constants import NO_VALID_CHILD, TOO_MANY_STEPS, WARNING_COLOR
from mcts_math.nodes import MCTSNode
from mcts_math.review_utils import compute_review_reward
from termcolor import colored

from .mcts import MCTS


class ReviewMCTS(MCTS):
    review_sample: Dict[str, Any]

    REVIEW_NODE_KEYS: List[str] = [
        "action",
        "action_input",
        "final_answer",
        "target_dimension",
        "reward_details",
        "force_final_review",
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        from .utils import review_prompt_wrap, review_step_result_unwrap

        self.prompt_wrap = review_prompt_wrap
        self.step_unwrap = review_step_result_unwrap
        self._bootstrap_dimension_children()
        self.select_next_step()

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        super().validate_config(cfg)
        if cfg.mode != "mcts":
            raise ValueError("ReviewMCTS only supports mcts mode.")
        if getattr(cfg, "self_review_value_func", False):
            raise ValueError(
                "self_review_value_func has been removed. Use objective labeled rewards for data generation "
                "or a trained value head with need_value_func=True."
            )
        return cfg

    def create_node(self, parent: Optional[Type[MCTSNode]] = None) -> Type[MCTSNode]:
        return MCTSNode(
            parent=parent,
            additional_state_keys=self.REVIEW_NODE_KEYS,
            c_puct=self.config.c_puct,
        )

    def _bootstrap_dimension_children(self) -> None:
        if self.root.children:
            return
        dimensions = self._selected_review_dimensions()
        if not dimensions:
            self.root.is_terminal = True
            return

        correctness_prior = 0.6 if "Correctness Verification" in dimensions and len(dimensions) > 1 else None
        for index, dimension in enumerate(dimensions):
            child = self.create_node(parent=self.root)
            child.tag = f"{self.root.tag}.{index}"
            child.depth = self.root.depth + 1
            if correctness_prior is not None:
                child.prior = correctness_prior if dimension == "Correctness Verification" else (1.0 - correctness_prior) / (len(dimensions) - 1)
            else:
                child.prior = 1.0 / len(dimensions)
            child.state["target_dimension"] = dimension
            self.root.children.append(child)

    def _selected_review_dimensions(self) -> List[str]:
        available = list(dict.fromkeys(self.review_sample["reference_scores"].keys()))
        preferred = [
            "Correctness Verification",
            "Robustness Validation",
            "Time Complexity Optimization",
            "Algorithm Optimization",
            "Output Format Compliance",
            "Space Complexity Control",
            "Maintainability",
            "Code Readability Enhancement",
        ]
        ordered = [dimension for dimension in preferred if dimension in available]
        ordered.extend(dimension for dimension in available if dimension not in ordered)
        return ordered[: self.config.max_review_dimensions]

    def _get_target_dimension(self, node: Type[MCTSNode]) -> str:
        cursor = node
        while cursor is not None:
            dimension = cursor.state.get("target_dimension")
            if dimension:
                return dimension
            cursor = cursor.parent
        raise ValueError("target_dimension is missing from review node lineage")

    def _visible_step_depth(self, node: Type[MCTSNode]) -> int:
        depth = 0
        cursor = node
        while cursor is not None:
            if cursor.state.get("text"):
                depth += 1
            cursor = cursor.parent
        return depth

    def is_ignored_node(self, node: Type[MCTSNode]) -> bool:
        return node.is_terminal or self._visible_step_depth(node) > self.config.max_depth

    def selection(self) -> Optional[Type[MCTSNode]]:
        while True:
            node = self.root
            while node.has_children() or node.is_terminal:
                next_node = self.select_child(node)
                if next_node is None:
                    node.is_terminal = True
                    break
                node = next_node

            if not node.is_terminal:
                return node
            if self.root.is_terminal:
                return None

    @staticmethod
    def _premature_review_to_step(step_result: str) -> str:
        draft = step_result.strip()
        if "<review>" in draft:
            draft = draft.split("<review>", 1)[1].strip()
        if "</review>" in draft:
            draft = draft.split("</review>", 1)[0].strip()
        return (
            "<step>\n"
            "Premature final review draft retained as a reasoning note, not as the final scored review:\n"
            f"{draft}\n"
            "</step>"
        )

    def create_prompt(self, is_value_only: bool = False) -> List[str]:
        if is_value_only and not self.config.need_value_func:
            return []
        prompts = []
        current_nodes = self.candidate_nodes if is_value_only else self.current_nodes
        for current_node in current_nodes:
            if not is_value_only and self.is_ignored_node(current_node):
                continue
            target_dimension = self._get_target_dimension(current_node)
            partial_solution = self.collect_partial_solution(current_node)
            review_context = {
                "target_dimension": target_dimension,
                "dimension_rubric": self.review_sample["dimension_rubrics"][target_dimension],
                "candidate_code": self.review_sample["candidate_code"],
                "tests_for_prompt": self.review_sample["tests_for_prompt"],
                "force_final_review": self._should_force_final_review(current_node),
            }
            prompt = self.prompt_wrap(
                self.question,
                partial_solution,
                self.config,
                review_context,
                is_value_only,
            )
            prompts.append(prompt.rstrip() + "\n")
        return prompts

    def _should_force_final_review(self, node: Type[MCTSNode]) -> bool:
        return bool(node.state.get("force_final_review")) or self._visible_step_depth(node) >= self.config.max_depth

    def create_child(
        self,
        step_result: str,
        parser_result: Dict[str, str],
        node: Type[MCTSNode],
        prior_prob: float,
        idx: int,
    ) -> None:
        new_node = self.create_node(parent=node)
        new_node.tag = f"{node.tag}.{idx}"
        new_node.depth = node.depth + 1
        new_node.prior = prior_prob
        new_node.state["target_dimension"] = self._get_target_dimension(node)

        if parser_result is None:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = NO_VALID_CHILD
            self.eval_final_answer(new_node)
        elif parser_result["final_answer"] and self._should_force_final_review(node):
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = parser_result["final_answer"]
            self.eval_final_answer(new_node)
        elif parser_result["final_answer"]:
            coerced_step = self._premature_review_to_step(step_result)
            new_node.state["text"] = coerced_step
            new_node.state["action"] = coerced_step
            new_node.state["action_input"] = ""
        elif self._should_force_final_review(node):
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = NO_VALID_CHILD
            self.eval_final_answer(new_node)
        elif parser_result["action"]:
            new_node.state["text"] = step_result.strip()
            new_node.state["action"] = parser_result["action"]
            new_node.state["action_input"] = parser_result["action_input"]
        else:
            print(colored(f"WARNING: '{step_result}' Cannot resolve\n", WARNING_COLOR))
            new_node.state["text"] = step_result

        if not new_node.is_terminal and self._visible_step_depth(new_node) > self.config.max_depth:
            new_node.is_terminal = True
            new_node.state["final_answer"] = TOO_MANY_STEPS
            self.eval_final_answer(new_node)

        node.children.append(new_node)

    def eval_final_answer(self, node: Type[MCTSNode]) -> None:
        if node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS]:
            node.state["reward_details"] = json.dumps({"error": node.state["final_answer"]}, ensure_ascii=False)
            node.update_recursive(self.config.negative_reward, self.root)
            return

        reward, details = compute_review_reward(
            target_dimension=self._get_target_dimension(node),
            final_answer=node.state["final_answer"],
            sample=self.review_sample,
        )
        node.state["reward_details"] = json.dumps(details, ensure_ascii=False)
        node.update_recursive(reward, self.root)
        if self.__class__.is_valid_final_answer_node(node):
            self.final_answer_nodes.append(node)

    def select_next_step(self, outputs=None) -> None:
        if outputs is not None:
            for candidate_node, output in zip(self.candidate_nodes, outputs):
                if candidate_node.is_terminal and candidate_node.state.get("reward_details"):
                    continue
                value_estimate = output.value_estimate if output and output.value_estimate is not None else self.config.negative_reward
                if output is None or output.value_estimate is None:
                    candidate_node.is_terminal = True
                candidate_node.update_recursive(value_estimate, self.root)
                if self.__class__.is_valid_final_answer_node(candidate_node):
                    self.final_answer_nodes.append(candidate_node)

        self.current_nodes = []
        selection_node = self.selection()
        if selection_node is not None:
            self.current_nodes.append(selection_node)

    def _collect_nodes(self) -> List[Type[MCTSNode]]:
        candidates = [self.root]
        nodes = []
        while candidates:
            node = candidates.pop(0)
            nodes.append(node)
            candidates.extend(node.children)
        return nodes

    def prepare_final_review_nodes(self) -> bool:
        all_nodes = self._collect_nodes()
        terminal_dimensions = {
            node.state.get("target_dimension")
            for node in all_nodes
            if self.__class__.is_valid_final_answer_node(node)
        }
        selected_nodes = []
        for dimension in self._selected_review_dimensions():
            if dimension in terminal_dimensions:
                continue
            leaf_candidates = [
                node
                for node in all_nodes
                if node.state.get("target_dimension") == dimension
                and not node.is_terminal
                and not node.children
            ]
            if not leaf_candidates:
                continue
            best_leaf = max(
                leaf_candidates,
                key=lambda node: (
                    self._visible_step_depth(node),
                    node.q_value(),
                    node.visit_count(),
                    node.value if node.value is not None else -100,
                ),
            )
            best_leaf.state["force_final_review"] = True
            selected_nodes.append(best_leaf)

        self.current_nodes = selected_nodes
        return bool(self.current_nodes)

    def generate_next_step(self, outputs) -> None:
        self.candidate_nodes = []
        for current_node, output in zip(self.current_nodes, outputs):
            self.expand_node(output.outputs, current_node)
            current_node.update_recursive(self.config.neutral_visit_reward, self.root)
            self.candidate_nodes.extend(current_node.children)
