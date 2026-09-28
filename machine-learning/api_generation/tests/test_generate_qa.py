"""Unit tests for qa_generation_api.generate_qa.

The Anthropic client is replaced with a fake that returns canned responses,
so no API key or network access is needed.
"""

import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from qa_generation_api import generate_qa as gqa


def make_message(*texts):
    """Build an object shaped like an Anthropic Message with text blocks."""
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=t) for t in texts])


class FakeClient:
    """Mimics client.messages.create, returning queued responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


# --- load_processes ---------------------------------------------------------


def test_load_processes_reads_pickled_list(tmp_path):
    path = tmp_path / "processes.pkl"
    processes = ["step one", "step two"]
    with open(path, "wb") as f:
        pickle.dump(processes, f)

    assert gqa.load_processes(path) == processes


def test_load_processes_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        gqa.load_processes(tmp_path / "missing.pkl")


def test_default_processes_path_is_next_to_module():
    assert gqa.PROCESSES_PATH.parent == gqa.PACKAGE_DIR
    assert gqa.PROCESSES_PATH.name == "flowchart_instructions.pkl"


# --- parse_questions --------------------------------------------------------


def test_parse_questions_splits_on_pipe():
    message = make_message("What is A?|What is B?|What is C?")

    assert gqa.parse_questions(message) == ["What is A?", "What is B?", "What is C?"]


def test_parse_questions_single_question():
    assert gqa.parse_questions(make_message("Only one?")) == ["Only one?"]


def test_parse_questions_uses_last_content_block():
    message = make_message("preamble|ignored", "Q1?|Q2?")

    assert gqa.parse_questions(message) == ["Q1?", "Q2?"]


def test_parse_questions_preserves_whitespace_and_empty_entries():
    # Documents current behavior: no stripping or filtering is applied.
    message = make_message(" Q1? | Q2? |")

    assert gqa.parse_questions(message) == [" Q1? ", " Q2? ", ""]


def test_parse_questions_empty_content_raises():
    with pytest.raises(ValueError, match="Cannot parse"):
        gqa.parse_questions(SimpleNamespace(content=[]))


def test_parse_questions_non_text_block_raises():
    message = SimpleNamespace(content=[SimpleNamespace(type="tool_use")])

    with pytest.raises(ValueError, match="Cannot parse"):
        gqa.parse_questions(message)


# --- generate_question_set --------------------------------------------------


def test_generate_question_set_maps_questions_to_process():
    client = FakeClient([make_message("Q1?|Q2?"), make_message("Q3?")])

    df = gqa.generate_question_set(client, ["process A", "process B"])

    expected = pd.DataFrame({
        "question": ["Q1?", "Q2?", "Q3?"],
        "process": ["process A", "process A", "process B"],
    })
    pd.testing.assert_frame_equal(df, expected)


def test_generate_question_set_has_continuous_index():
    client = FakeClient([make_message("a|b"), make_message("c|d")])

    df = gqa.generate_question_set(client, ["p1", "p2"])

    assert list(df.index) == [0, 1, 2, 3]


def test_generate_question_set_calls_api_once_per_process():
    client = FakeClient([make_message("Q?")] * 3)

    gqa.generate_question_set(client, ["p1", "p2", "p3"])

    assert len(client.calls) == 3


def test_generate_question_set_request_parameters():
    client = FakeClient([make_message("Q?")])

    gqa.generate_question_set(client, ["Do X then Y."])

    call = client.calls[0]
    assert call["model"] == gqa.MODEL
    assert call["max_tokens"] == gqa.MAX_TOKENS
    assert call["system"] == gqa.SYSTEM_PROMPT
    assert len(call["messages"]) == 1
    assert call["messages"][0]["role"] == "user"
    assert call["messages"][0]["content"].endswith("Do X then Y.")


def test_generate_question_set_empty_input_makes_no_calls():
    client = FakeClient([])

    df = gqa.generate_question_set(client, [])

    assert df.empty
    assert client.calls == []


def test_generate_question_set_propagates_parse_errors():
    client = FakeClient([make_message("Q1?"), SimpleNamespace(content=[])])

    with pytest.raises(ValueError, match="Cannot parse"):
        gqa.generate_question_set(client, ["p1", "p2"])


def test_generate_question_set_roundtrips_through_parquet(tmp_path):
    client = FakeClient([make_message("Q1?|Q2?")])
    df = gqa.generate_question_set(client, ["process A"])

    path = tmp_path / "question_set.pq"
    df.to_parquet(path)

    pd.testing.assert_frame_equal(pd.read_parquet(path), df)
