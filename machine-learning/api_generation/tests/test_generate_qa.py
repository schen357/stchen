"""Unit tests for api_generation.generate_question_set.

The Anthropic client is replaced with a fake that returns canned responses,
so no API key or network access is needed.
"""

import json
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from api_generation import generate_question_set as gqa
from api_generation import utils


def make_message(*texts):
    """Build an object shaped like an Anthropic Message with text blocks."""
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=t) for t in texts])


def make_payload(title, questions):
    """Build the JSON text Claude is prompted to return."""
    return json.dumps({"process_title": title, "question_set": questions})


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


def test_default_paths_are_next_to_module():
    assert gqa.PROCESSES_PATH.parent == gqa.PACKAGE_DIR
    assert gqa.PROCESSES_PATH.name == "flowchart_instructions.pkl"
    assert gqa.QUESTION_SET_PATH.parent == gqa.PACKAGE_DIR
    assert gqa.QUESTION_SET_PATH.name == "question_set.pq"


# --- parse_questions --------------------------------------------------------


def test_parse_questions_reads_title_and_questions():
    message = make_message(make_payload("Order QA", ["What is A?", "What is B?"]))

    assert gqa.parse_questions(message) == ("Order QA", ["What is A?", "What is B?"])


@pytest.mark.parametrize("wrap", [
    "```json\n{}\n```",
    "```\n{}\n```",
    "  \n{}\n  ",
])
def test_parse_questions_strips_markdown_fences_and_whitespace(wrap):
    message = make_message(wrap.format(make_payload("T", ["Q?"])))

    assert gqa.parse_questions(message) == ("T", ["Q?"])


def test_parse_questions_uses_last_content_block():
    message = make_message("Here is your JSON:", make_payload("T", ["Q1?", "Q2?"]))

    assert gqa.parse_questions(message) == ("T", ["Q1?", "Q2?"])


def test_parse_questions_invalid_json_raises():
    with pytest.raises(ValueError, match="Cannot parse"):
        gqa.parse_questions(make_message("Q1?|Q2?"))


def test_parse_questions_missing_key_raises():
    message = make_message(json.dumps({"process_title": "T"}))

    with pytest.raises(ValueError, match="Cannot parse"):
        gqa.parse_questions(message)


def test_parse_questions_empty_content_raises():
    with pytest.raises(ValueError, match="Cannot parse"):
        gqa.parse_questions(SimpleNamespace(content=[]))


def test_parse_questions_non_text_block_raises():
    message = SimpleNamespace(content=[SimpleNamespace(type="tool_use")])

    with pytest.raises(ValueError, match="Cannot parse"):
        gqa.parse_questions(message)


# --- generate_question_set --------------------------------------------------


def test_generate_question_set_maps_questions_to_process():
    client = FakeClient([
        make_message(make_payload("Title A", ["Q1?", "Q2?"])),
        make_message(make_payload("Title B", ["Q3?"])),
    ])

    df = gqa.generate_question_set(client, ["process A", "process B"])

    expected = pd.DataFrame({
        "process_title": ["Title A", "Title B"],
        "process": ["process A", "process B"],
        "question_set": [["Q1?", "Q2?"], ["Q3?"]],
    })
    pd.testing.assert_frame_equal(df, expected)


def test_generate_question_set_has_continuous_index():
    client = FakeClient([make_message(make_payload(t, ["Q?"])) for t in "abc"])

    df = gqa.generate_question_set(client, ["p1", "p2", "p3"])

    assert list(df.index) == [0, 1, 2]


def test_generate_question_set_calls_api_once_per_process():
    client = FakeClient([make_message(make_payload("T", ["Q?"]))] * 3)

    gqa.generate_question_set(client, ["p1", "p2", "p3"])

    assert len(client.calls) == 3


def test_generate_question_set_request_parameters():
    client = FakeClient([make_message(make_payload("T", ["Q?"]))])

    gqa.generate_question_set(client, ["Do X then Y."])

    call = client.calls[0]
    assert call["model"] == utils.MODEL
    assert call["max_tokens"] == utils.MAX_TOKENS
    assert call["system"] == gqa.SYSTEM_PROMPT
    assert len(call["messages"]) == 1
    assert call["messages"][0]["role"] == "user"
    assert call["messages"][0]["content"].endswith("Do X then Y.")


def test_generate_question_set_sends_each_process():
    client = FakeClient([make_message(make_payload("T", ["Q?"]))] * 2)

    gqa.generate_question_set(client, ["first process", "second process"])

    assert client.calls[0]["messages"][0]["content"].endswith("first process")
    assert client.calls[1]["messages"][0]["content"].endswith("second process")


def test_generate_question_set_empty_input_makes_no_calls():
    client = FakeClient([])

    df = gqa.generate_question_set(client, [])

    assert df.empty
    assert list(df.columns) == ["process_title", "process", "question_set"]
    assert client.calls == []


def test_generate_question_set_propagates_parse_errors():
    client = FakeClient([make_message(make_payload("T", ["Q?"])), SimpleNamespace(content=[])])

    with pytest.raises(ValueError, match="Cannot parse"):
        gqa.generate_question_set(client, ["p1", "p2"])


def test_generate_question_set_roundtrips_through_parquet(tmp_path):
    client = FakeClient([make_message(make_payload("T", ["Q1?", "Q2?"]))])
    df = gqa.generate_question_set(client, ["process A"])

    path = tmp_path / "question_set.pq"
    df.to_parquet(path)
    loaded = pd.read_parquet(path)

    # Parquet reads list columns back as numpy arrays
    loaded["question_set"] = loaded["question_set"].map(list)
    pd.testing.assert_frame_equal(loaded, df)
