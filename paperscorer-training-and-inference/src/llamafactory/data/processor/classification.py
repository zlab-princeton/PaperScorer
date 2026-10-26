# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from .processor_utils import DatasetProcessor


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import AudioInput, ImageInput, VideoInput
    from ..template import Template


logger = logging.get_logger(__name__)


@dataclass
class ClassificationDatasetProcessor(DatasetProcessor):
    r"""Dataset processor for binary classification using last token hidden state.

    Uses the last token in the input sequence (naturally <|im_end|> for Qwen) for classification.
    No CLS token is added - we rely on the natural end-of-turn token.
    """

    positive_token: str = "Accept"
    negative_token: str = "Reject"

    def _extract_label(self, response_text: str) -> int:
        """Extract binary label from response text."""
        if self.positive_token in response_text:
            return 1
        elif self.negative_token in response_text:
            return 0
        raise ValueError(f"Could not extract label from response: {response_text}")

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], int]:
        """Encode data example for classification.

        Input = system message + user message (prompt only), ending with <|im_end|>
        Response text is used ONLY to extract the label, not included in input_ids.

        Returns:
            tuple: (input_ids ending with <|im_end|>, binary label)
        """
        # Process messages for multimodal (need prompt + response for encode_multiturn to work)
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, _ = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )

        # encode_multiturn returns [(source_ids, target_ids), ...]
        # source_ids = user turn ending with "<|im_start|>assistant\n" (where model generates)
        # This is natural for classification - predict from where the model would generate
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        source_ids, target_ids = encoded_pairs[0]

        # TODO: add truncation

        input_ids += source_ids

        # Extract binary label from response text (response NOT included in input_ids)
        response_text = response[0]["content"] if response else ""
        label = self._extract_label(response_text)

        return input_ids, label

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        r"""Build model inputs for binary classification.

        Returns inputs ending with <|im_end|> and binary labels (0 or 1).
        """
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            try:
                input_ids, label = self._encode_data_example(
                    prompt=examples["_prompt"][i],
                    response=examples["_response"][i],
                    system=examples["_system"][i],
                    tools=examples["_tools"][i],
                    images=examples["_images"][i] or [],
                    videos=examples["_videos"][i] or [],
                    audios=examples["_audios"][i] or [],
                )
            except ValueError as e:
                logger.warning_rank0(f"Dropped example due to label extraction error: {e}")
                continue

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(label)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])
            # Pass through metadata for predictions
            if "_metadata" in examples and examples["_metadata"][i] is not None:
                model_inputs["_metadata"].append(examples["_metadata"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print(f"label: {example['labels']}")
