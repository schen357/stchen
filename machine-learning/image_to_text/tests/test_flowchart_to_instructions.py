"""Unit tests for image_to_text.flowchart_to_instructions.

The Qwen model and processor are replaced with lightweight fakes so these
tests run in milliseconds without downloading weights or needing MPS.
"""

import pytest
import torch
from PIL import Image

from image_to_text import flowchart_to_instructions as fti


PROMPT_LEN = 3
NEW_TOKENS = [101, 102]


class FakeProcessor:
    """Records calls and returns tensors shaped like the real processor's."""

    def __init__(self):
        self.chat_template_calls = []
        self.call_kwargs = None
        self.decode_kwargs = None
        self.decoded_ids = None

    def apply_chat_template(self, messages, **kwargs):
        self.chat_template_calls.append((messages, kwargs))
        return "FORMATTED_PROMPT"

    def __call__(self, text, images, **kwargs):
        self.call_kwargs = {"text": text, "images": images, **kwargs}
        n = len(images)
        return {
            "input_ids": torch.arange(n * PROMPT_LEN).reshape(n, PROMPT_LEN),
            "attention_mask": torch.ones(n, PROMPT_LEN, dtype=torch.long),
        }

    def batch_decode(self, ids, **kwargs):
        self.decode_kwargs = kwargs
        self.decoded_ids = [row.tolist() for row in ids]
        return [" ".join(str(t) for t in row) for row in self.decoded_ids]


class FakeModel:
    """Echoes the prompt ids followed by NEW_TOKENS, like a causal LM."""

    def __init__(self):
        self.generate_kwargs = None
        self.inference_mode_enabled = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        self.inference_mode_enabled = torch.is_inference_mode_enabled()
        input_ids = kwargs["input_ids"]
        new = torch.tensor(NEW_TOKENS).repeat(input_ids.shape[0], 1)
        return torch.cat([input_ids, new], dim=1)


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch):
    monkeypatch.setattr(fti, "DEVICE", "cpu")


@pytest.fixture
def processor():
    return FakeProcessor()


@pytest.fixture
def model():
    return FakeModel()


def make_image(path, mode="RGB"):
    Image.new(mode, (8, 8)).save(path)
    return path


# --- module constants -------------------------------------------------------


def test_paths_are_resolved_relative_to_package():
    assert fti.PACKAGE_DIR.is_absolute()
    assert fti.PACKAGE_DIR.name == "image_to_text"
    assert fti.DATASET_DIR == fti.PACKAGE_DIR / "flowcharts" / "FlowChart.v3i.multiclass"
    assert fti.TRAIN_DIR == fti.DATASET_DIR / "train"


# --- image_paths ------------------------------------------------------------


def test_image_paths_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Split directory not found"):
        fti.image_paths(tmp_path / "does_not_exist")


def test_image_paths_file_instead_of_dir_raises(tmp_path):
    f = tmp_path / "not_a_dir.jpg"
    f.touch()
    with pytest.raises(FileNotFoundError):
        fti.image_paths(f)


def test_image_paths_empty_dir(tmp_path):
    assert fti.image_paths(tmp_path) == []


def test_image_paths_filters_by_suffix_case_insensitive(tmp_path):
    for name in ["b.jpg", "a.PNG", "c.jpeg", "d.JPG", "notes.txt", ".DS_Store", "e.gif"]:
        (tmp_path / name).touch()

    result = fti.image_paths(tmp_path)

    assert [p.name for p in result] == ["a.PNG", "b.jpg", "c.jpeg", "d.JPG"]


def test_image_paths_returns_sorted_paths_inside_split(tmp_path):
    for name in ["003.jpg", "001.jpg", "002.jpg"]:
        (tmp_path / name).touch()

    result = fti.image_paths(tmp_path)

    assert result == sorted(result)
    assert all(p.parent == tmp_path for p in result)


def test_image_paths_default_uses_bundled_train_split():
    result = fti.image_paths()

    assert result, "expected the bundled train split to contain images"
    assert all(p.parent == fti.TRAIN_DIR for p in result)
    assert all(p.suffix.lower() in fti.IMAGE_SUFFIXES for p in result)


# --- image_to_instructions --------------------------------------------------


def test_returns_one_decoded_string_per_image(tmp_path, model, processor):
    paths = [make_image(tmp_path / f"{i}.png") for i in range(3)]

    output = fti.image_to_instructions(model, processor, paths, "describe")

    assert output == ["101 102"] * 3


def test_output_is_trimmed_to_newly_generated_tokens(tmp_path, model, processor):
    paths = [make_image(tmp_path / f"{i}.png") for i in range(2)]

    fti.image_to_instructions(model, processor, paths, "describe")

    assert processor.decoded_ids == [NEW_TOKENS, NEW_TOKENS]


def test_chat_template_includes_prompt_and_generation_flags(tmp_path, model, processor):
    path = make_image(tmp_path / "a.png")

    fti.image_to_instructions(model, processor, [path], "my prompt")

    assert len(processor.chat_template_calls) == 1
    messages, kwargs = processor.chat_template_calls[0]
    assert kwargs == {"tokenize": False, "add_generation_prompt": True}
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert content[0]["type"] == "image"
    assert isinstance(content[0]["image"], Image.Image)
    assert content[1] == {"type": "text", "text": "my prompt"}


def test_each_image_gets_its_own_chat_prompt(tmp_path, model, processor):
    sizes = [(8, 8), (16, 16), (24, 24)]
    paths = []
    for i, size in enumerate(sizes):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", size).save(path)
        paths.append(path)

    fti.image_to_instructions(model, processor, paths, "describe")

    assert len(processor.chat_template_calls) == len(sizes)
    prompt_images = [
        messages[0]["content"][0]["image"] for messages, _ in processor.chat_template_calls
    ]
    assert [im.size for im in prompt_images] == sizes


def test_empty_filepaths_returns_empty_list(model, processor):
    assert fti.image_to_instructions(model, processor, [], "describe") == []
    assert processor.chat_template_calls == []
    assert model.generate_kwargs is None


def test_processor_receives_one_prompt_per_image(tmp_path, model, processor):
    paths = [make_image(tmp_path / f"{i}.png") for i in range(4)]

    fti.image_to_instructions(model, processor, paths, "describe")

    kw = processor.call_kwargs
    assert kw["text"] == ["FORMATTED_PROMPT"] * 4
    assert len(kw["images"]) == 4
    assert kw["padding"] is True
    assert kw["return_tensors"] == "pt"


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_images_are_converted_to_rgb(tmp_path, model, processor, mode):
    path = make_image(tmp_path / "img.png", mode=mode)

    fti.image_to_instructions(model, processor, [path], "describe")

    assert [im.mode for im in processor.call_kwargs["images"]] == ["RGB"]


def test_generate_called_with_deterministic_settings(tmp_path, model, processor):
    path = make_image(tmp_path / "a.png")

    fti.image_to_instructions(model, processor, [path], "describe")

    kw = model.generate_kwargs
    assert kw["max_new_tokens"] == 512
    assert kw["do_sample"] is False
    assert {"input_ids", "attention_mask"} <= kw.keys()


def test_inputs_are_moved_to_device(tmp_path, model, processor):
    path = make_image(tmp_path / "a.png")

    fti.image_to_instructions(model, processor, [path], "describe")

    for name in ("input_ids", "attention_mask"):
        assert model.generate_kwargs[name].device.type == "cpu"


def test_generation_runs_in_inference_mode(tmp_path, model, processor):
    path = make_image(tmp_path / "a.png")

    fti.image_to_instructions(model, processor, [path], "describe")

    assert model.inference_mode_enabled is True


def test_batch_decode_skips_special_tokens(tmp_path, model, processor):
    path = make_image(tmp_path / "a.png")

    fti.image_to_instructions(model, processor, [path], "describe")

    assert processor.decode_kwargs == {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }


def test_missing_image_returns_none_and_skips_inference(tmp_path, model, processor, capsys):
    good = make_image(tmp_path / "good.png")
    missing = tmp_path / "missing.png"

    result = fti.image_to_instructions(model, processor, [good, missing], "describe")

    assert result is None
    assert f"Image file not found at {missing}" in capsys.readouterr().out
    assert processor.call_kwargs is None
    assert model.generate_kwargs is None
