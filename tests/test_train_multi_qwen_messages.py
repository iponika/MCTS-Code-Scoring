import unittest

from magicoder.train_multi import IGNORED_INDEX, qwen_messages_to_training_ids


class FakeQwenTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    bos_token = "<s>"
    eos_token = "</s>"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def _ids(self, text: str) -> list[int]:
        ids = []
        for char in text:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab) + 10
            ids.append(self.vocab[char])
        return ids

    def __call__(self, text_list, add_special_tokens=False, **_kwargs):
        if isinstance(text_list, str):
            text_list = [text_list]
        return {"input_ids": [self._ids(text) for text in text_list]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_kwargs):
        rendered = ""
        for message in messages:
            rendered += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        if tokenize:
            return self._ids(rendered)
        return rendered


class FakeDeepSeekTokenizer(FakeQwenTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_kwargs):
        rendered = "<bos>"
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "assistant" and content.startswith("<think>\n\n</think>\n\n"):
                content = content.split("</think>\n\n", 1)[1]
            rendered += f"<|{role}|>{content}"
        if add_generation_prompt:
            rendered += "<|assistant|><think>\n"
        if tokenize:
            return self._ids(rendered)
        return rendered


class FakeDeepSeekReviewTokenizer(FakeQwenTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_kwargs):
        rendered = "<bos>"
        for message in messages:
            role = message["role"]
            content = message["content"]
            rendered += f"<|{role}|>{content}"
        if add_generation_prompt:
            rendered += "<|assistant|><think>\n"
        elif messages and messages[-1]["role"] == "assistant" and messages[-1]["content"] == "":
            rendered += "<eos>"
        if tokenize:
            return self._ids(rendered)
        return rendered


class QwenMessageTrainingTest(unittest.TestCase):
    def test_qwen_messages_mask_user_tokens_and_label_assistant_parts(self) -> None:
        tokenizer = FakeQwenTokenizer()
        messages = [
            {"role": "user", "content": "score code"},
            {"role": "assistant", "content": "<think>\nA\n</think>\n\n<review>B</review>"},
        ]
        assistant_parts = [
            {"type": "reasoning", "text": "A", "q_value": 0.25},
            {"type": "review", "text": "<review>B</review>", "q_value": 0.75},
        ]

        result = qwen_messages_to_training_ids(
            tokenizer,
            messages,
            assistant_parts,
            train_lm=True,
        )

        self.assertEqual(len(result["input_ids"]), len(result["labels"]))
        first_labeled = next(index for index, label in enumerate(result["labels"]) if label != IGNORED_INDEX)
        self.assertTrue(all(label == IGNORED_INDEX for label in result["labels"][:first_labeled]))
        q_positions = [index for index, value in enumerate(result["Q"]) if value != IGNORED_INDEX]
        self.assertEqual(len(q_positions), 2)
        self.assertEqual(result["Q"][q_positions[0]], 0.25)
        self.assertEqual(result["Q"][q_positions[1]], 0.75)

    def test_qwen_messages_can_keep_value_only_without_lm_labels(self) -> None:
        tokenizer = FakeQwenTokenizer()
        messages = [
            {"role": "user", "content": "score code"},
            {"role": "assistant", "content": "<think>\n\n</think>\n\n<review>B</review>"},
        ]
        assistant_parts = [{"type": "review", "text": "<review>B</review>", "q_value": -0.5}]

        result = qwen_messages_to_training_ids(
            tokenizer,
            messages,
            assistant_parts,
            train_lm=False,
        )

        self.assertTrue(all(label == IGNORED_INDEX for label in result["labels"]))
        self.assertEqual(sum(value != IGNORED_INDEX for value in result["Q"]), 1)

    def test_qwen_messages_accept_assistant_prefix_when_empty_think_is_collapsed(self) -> None:
        tokenizer = FakeDeepSeekTokenizer()
        messages = [
            {"role": "user", "content": "score code"},
            {"role": "assistant", "content": "<think>\n\n</think>\n\n<review>B</review>"},
        ]
        assistant_parts = [{"type": "review", "text": "<review>B</review>", "q_value": -0.5}]

        result = qwen_messages_to_training_ids(
            tokenizer,
            messages,
            assistant_parts,
            train_lm=True,
        )

        self.assertEqual(len(result["input_ids"]), len(result["labels"]))
        self.assertEqual(sum(value != IGNORED_INDEX for value in result["Q"]), 1)

    def test_qwen_messages_accept_user_only_prefix_when_deepseek_review_starts_without_think(self) -> None:
        tokenizer = FakeDeepSeekReviewTokenizer()
        messages = [
            {"role": "user", "content": "score code"},
            {"role": "assistant", "content": "<review>B</review>"},
        ]
        assistant_parts = [{"type": "review", "text": "<review>B</review>", "q_value": 0.5}]

        result = qwen_messages_to_training_ids(
            tokenizer,
            messages,
            assistant_parts,
            train_lm=True,
        )

        self.assertEqual(len(result["input_ids"]), len(result["labels"]))
        self.assertEqual(sum(value != IGNORED_INDEX for value in result["Q"]), 1)


if __name__ == "__main__":
    unittest.main()
