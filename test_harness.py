import asyncio
import itertools
import json
from types import SimpleNamespace

import pytest

import harness

# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Point every path at a temp dir and start each test with a clean history."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(harness, "WORKSPACE", workspace)
    monkeypatch.setattr(harness, "MEMORY_FILE", tmp_path / "memory.md")
    monkeypatch.setattr(harness, "messages", [{"role": "system", "content": "test"}])
    return workspace


def make_call(name, args, call_id="call_1"):
    function = SimpleNamespace(name=name, arguments=json.dumps(args))
    return SimpleNamespace(id=call_id, function=function)


def make_message(content=None, tool_calls=None):
    """A stand-in for the SDK's chat message: .content, .tool_calls, .model_dump()."""

    def model_dump(**_):
        data = {"role": "assistant", "content": content}
        if tool_calls:
            data["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in tool_calls
            ]
        return data

    return SimpleNamespace(content=content, tool_calls=tool_calls, model_dump=model_dump)


class FakeClient:
    """Mimics client.chat.completions.create(); replays queued replies in order.

    A reply that is an Exception is raised instead of returned.
    """

    def __init__(self, replies):
        self._replies = iter(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        reply = next(self._replies)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=reply)])


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------


def test_write_then_read_roundtrip(sandbox):
    assert harness.write_file("todo.md", "buy milk") == "wrote todo.md"
    assert harness.read_file("todo.md") == "buy milk"
    assert (sandbox / "todo.md").read_text(encoding="utf-8") == "buy milk"


def test_read_missing_file_returns_error():
    assert harness.read_file("nope.txt") == "error: no nope.txt"


def test_list_files_empty_and_sorted(sandbox):
    assert harness.list_files() == "(empty)"
    (sandbox / "b.txt").write_text("b")
    (sandbox / "a.txt").write_text("a")
    assert harness.list_files() == "a.txt\nb.txt"


def test_path_traversal_is_rejected(sandbox):
    outside = sandbox.parent / "secret.txt"
    outside.write_text("top secret")

    assert harness.read_file("../secret.txt").startswith("error:")
    assert "top secret" not in harness.read_file("../secret.txt")
    assert harness.read_file(str(outside)).startswith("error:")  # absolute path

    assert harness.write_file("../evil.txt", "x").startswith("error:")
    assert not (sandbox.parent / "evil.txt").exists()


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_memory_empty_by_default():
    assert harness.load_memory() == "(nothing saved yet)"


def test_memory_roundtrip():
    assert harness.save_memory("likes tea") == "saved: likes tea"
    harness.save_memory("lives in Prague")
    memory = harness.load_memory()
    assert "- likes tea" in memory
    assert "- lives in Prague" in memory


# ---------------------------------------------------------------------------
# call_tool: always returns a string, never raises
# ---------------------------------------------------------------------------


def test_call_tool_runs_local_tool():
    assert run(harness.call_tool("write_file", '{"filename": "a.txt", "content": "hi"}')) == "wrote a.txt"
    assert run(harness.call_tool("read_file", '{"filename": "a.txt"}')) == "hi"


def test_call_tool_accepts_empty_arguments():
    assert run(harness.call_tool("list_files", "")) == "(empty)"


def test_call_tool_unknown_tool():
    assert run(harness.call_tool("does_not_exist", "{}")).startswith("error: unknown tool")


def test_call_tool_bad_json_and_bad_arguments():
    assert run(harness.call_tool("read_file", "{not json")).startswith("error:")
    assert run(harness.call_tool("read_file", "{}")).startswith("error:")  # missing filename
    assert run(harness.call_tool("read_file", "[]")).startswith("error:")  # not an object


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def test_run_agent_plain_reply():
    client = FakeClient([make_message(content="hello!")])

    assert run(harness.run_agent("hi", client=client)) == "hello!"
    assert [m["role"] for m in harness.messages] == ["system", "user", "assistant"]


def test_run_agent_tool_call_then_reply(sandbox):
    call = make_call("write_file", {"filename": "note.md", "content": "remember"})
    client = FakeClient(
        [
            make_message(tool_calls=[call]),
            make_message(content="saved your note"),
        ]
    )

    assert run(harness.run_agent("write a note", client=client)) == "saved your note"

    assert (sandbox / "note.md").read_text(encoding="utf-8") == "remember"
    roles = [m["role"] for m in harness.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    tool_message = harness.messages[3]
    assert tool_message["tool_call_id"] == "call_1"
    assert tool_message["content"] == "wrote note.md"


def test_run_agent_tool_error_goes_back_to_the_model():
    call = make_call("read_file", {"filename": "../../etc/passwd"})
    client = FakeClient([make_message(tool_calls=[call]), make_message(content="can't do that")])

    assert run(harness.run_agent("read it", client=client)) == "can't do that"
    assert harness.messages[3]["content"].startswith("error:")


def test_run_agent_stops_after_max_steps_and_rolls_back():
    looping = make_message(tool_calls=[make_call("list_files", {})])
    client = FakeClient(itertools.repeat(looping))

    with pytest.raises(RuntimeError, match="no final answer"):
        run(harness.run_agent("loop forever", client=client, max_steps=3))

    assert len(harness.messages) == 1  # only the system message is left


def test_run_agent_rolls_back_history_when_the_api_fails():
    call = make_call("list_files", {})
    client = FakeClient([make_message(tool_calls=[call]), ConnectionError("ollama is down")])

    with pytest.raises(ConnectionError):
        run(harness.run_agent("hi", client=client))

    assert len(harness.messages) == 1
