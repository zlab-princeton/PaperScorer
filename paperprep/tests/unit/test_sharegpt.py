"""Tests for paperprep's minimal ShareGPT builder."""

from paper_anonymizer.core.constants import (
    SHAREGPT_SYSTEM_PROMPT,
    SHAREGPT_USER_PROMPT_BINARY,
)
from paper_anonymizer.dataset.sharegpt import create_sharegpt_conversation


class TestCreateSharegptConversation:
    def test_with_label_includes_gpt_turn(self):
        entry = create_sharegpt_conversation("paper text", label="Accept")
        roles = [c["from"] for c in entry["conversations"]]
        assert roles == ["system", "human", "gpt"]
        assert entry["conversations"][-1]["value"] == "Outcome: \\boxed{Accept}"

    def test_label_none_drops_gpt_turn(self):
        entry = create_sharegpt_conversation("paper text", label=None)
        roles = [c["from"] for c in entry["conversations"]]
        assert roles == ["system", "human"]
        assert all(c["from"] != "gpt" for c in entry["conversations"])

    def test_label_none_preserves_metadata_and_images(self):
        entry = create_sharegpt_conversation(
            "p", label=None, metadata={"id": "x"}, images=["a.png", "b.png"]
        )
        assert entry["_metadata"] == {"id": "x"}
        assert entry["images"] == ["a.png", "b.png"]
        assert len(entry["conversations"]) == 2

    def test_human_turn_is_prompt_plus_paper(self):
        entry = create_sharegpt_conversation(
            "paper body here", label="Reject", task_prompt="MY PROMPT"
        )
        assert entry["conversations"][1]["value"] == "MY PROMPT\n\npaper body here"

    def test_custom_system_prompt(self):
        entry = create_sharegpt_conversation(
            "x", label=None, system_prompt="custom system"
        )
        assert entry["conversations"][0]["value"] == "custom system"


class TestConstants:
    def test_prompts_defined(self):
        assert "expert academic reviewer" in SHAREGPT_SYSTEM_PROMPT
        assert "predict its acceptance outcome" in SHAREGPT_USER_PROMPT_BINARY.lower()
        assert "\\boxed{Accept}" in SHAREGPT_USER_PROMPT_BINARY
