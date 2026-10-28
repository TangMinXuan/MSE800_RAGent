import hashlib
import json
from dataclasses import dataclass

import pytest

from util.collection_util import batched, dedupe
from util.env_util import get_env, get_env_bool, get_env_int, require_env
from util.file_util import file_sha256, list_files, read_text
from util.json_util import from_json, to_json
from util.retry_util import retry
from util.string_util import is_blank
from util.text_util import estimate_tokens, normalize_whitespace, split_paragraphs, truncate
from util.time_util import timer, utc_now, utc_now_iso


def test_batched_and_invalid_size():
    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(batched([], 2)) == []
    with pytest.raises(ValueError, match="size must be positive"):
        list(batched([1], 0))


def test_dedupe_preserves_order_and_supports_key():
    assert dedupe(["a", "b", "a", "c"]) == ["a", "b", "c"]
    rows = [{"id": 1}, {"id": 2}, {"id": 1, "v": "second"}]
    assert dedupe(rows, key=lambda row: row["id"]) == rows[:2]


@pytest.mark.parametrize("value,expected", [(None, True), ("", True), (" \t ", True), ("x", False)])
def test_is_blank(value, expected):
    assert is_blank(value) is expected


def test_environment_helpers(monkeypatch):
    monkeypatch.setenv("TEST_VALUE", "  hello ")
    assert get_env("TEST_VALUE") == "hello"
    monkeypatch.setenv("TEST_VALUE", "  ")
    assert get_env("TEST_VALUE", "fallback") == "fallback"
    monkeypatch.delenv("TEST_VALUE", raising=False)
    assert get_env("TEST_VALUE", "fallback") == "fallback"
    with pytest.raises(RuntimeError, match="TEST_VALUE"):
        require_env("TEST_VALUE")
    monkeypatch.setenv("TEST_INT", "12")
    assert get_env_int("TEST_INT", 4) == 12
    monkeypatch.setenv("TEST_BOOL", "YES")
    assert get_env_bool("TEST_BOOL") is True
    monkeypatch.setenv("TEST_BOOL", "no")
    assert get_env_bool("TEST_BOOL", True) is False


def test_file_helpers(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("Kia ora Māori", encoding="utf-8")
    assert read_text(p) == "Kia ora Māori"
    assert file_sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path / "ignored.py").write_text("x", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "notes.MD").write_text("md", encoding="utf-8")
    assert [x.name for x in list_files(tmp_path)] == ["sample.txt", "notes.MD"]
    assert [x.name for x in list_files(tmp_path, (".py",))] == ["ignored.py"]


@dataclass
class Sample:
    name: str
    count: int


def test_json_helpers():
    encoded = to_json(Sample("Māori", 2))
    assert json.loads(encoded) == {"name": "Māori", "count": 2}
    assert from_json(encoded) == {"name": "Māori", "count": 2}
    fallback = object()
    assert from_json(None, fallback) is fallback
    assert from_json("not json", fallback) is fallback
    with pytest.raises(TypeError, match="not JSON serializable"):
        to_json(object())


def test_retry_success_retry_and_failure(monkeypatch):
    sleeps = []
    monkeypatch.setattr("util.retry_util.time.sleep", sleeps.append)
    attempts = []

    @retry(attempts=3, delay=0.1, backoff=2)
    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ValueError("temporary")
        return "ok"

    assert flaky() == "ok"
    assert len(attempts) == 3
    assert sleeps == [0.1, 0.2]
    with pytest.raises(ValueError, match="attempts must be >= 1"):
        retry(attempts=0)


def test_text_helpers():
    assert normalize_whitespace("  hello\n\t world ") == "hello world"
    assert truncate("abcdefgh", 6) == "abc..."
    assert truncate("abc", 0) == ""
    assert estimate_tokens("abcde") == 2
    assert split_paragraphs(" one \n\n two \n \n three ") == ["one", "two", "three"]


def test_time_helpers_return_utc_and_elapsed():
    now = utc_now()
    assert now.tzinfo is not None and now.utcoffset().total_seconds() == 0
    assert utc_now_iso().endswith("+00:00")
    with timer() as elapsed:
        assert elapsed() >= 0
