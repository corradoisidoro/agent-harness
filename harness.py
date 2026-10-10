"""A minimal personal-assistant agent harness.

Connects one OpenAI-compatible chat endpoint to a small set of local tools
(workspace file access and persistent memory) plus any tools exposed by the MCP
servers listed in ``mcp_servers.json``, and runs an interactive chat loop.
"""

import asyncio
import json
import sys
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from pydantic import BaseModel

ROOT = Path(__file__).parent
WORKSPACE = ROOT / "workspace"
MEMORY_FILE = ROOT / "memory.md"
MCP_CONFIG = ROOT / "mcp_servers.json"

# Upper bound on model <-> tool round trips per user message. Prevents a model
# that keeps requesting tools from looping indefinitely.
MAX_STEPS = 10


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Connection details for one OpenAI-compatible chat completions endpoint."""

    base_url: str
    api_key: str = "none"  # local servers such as Ollama ignore the key
    model: str


# Every entry speaks the same protocol, so the agent loop is provider-agnostic.
# Adding a model means adding an entry here; no other code changes.
MODELS: dict[str, ModelConfig] = {
    "local": ModelConfig(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3.5:4b",
    ),
    "local-small": ModelConfig(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3.5:2b",
    ),
}

# Key into MODELS for the active model. Changed at runtime with /model.
model_name = "local"


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def load_memory() -> str:
    """Return the contents of the memory file, or a placeholder if it is absent."""
    if MEMORY_FILE.is_file():
        return MEMORY_FILE.read_text(encoding="utf-8")
    return "(nothing saved yet)"


def save_memory(fact: str) -> str:
    """Append one fact to the memory file as a markdown bullet.

    Memory is read once at startup (see ``main``), so a saved fact becomes
    visible to the model in the next session, not the current one. The file is
    append-only and entries are not deduplicated.
    """
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- {fact}\n")
    return f"saved: {fact}"


# ---------------------------------------------------------------------------
# Local tools
# ---------------------------------------------------------------------------


def _safe_path(filename: str) -> Path | None:
    """Resolve ``filename`` inside ``WORKSPACE``.

    Returns None if the resolved path falls outside the workspace. Resolution
    follows symlinks, so a link inside the workspace that points elsewhere is
    rejected as well. Absolute paths and ``..`` segments that escape the
    workspace are rejected for the same reason.
    """
    path = (WORKSPACE / filename).resolve()
    return path if path.is_relative_to(WORKSPACE.resolve()) else None


async def ainput(prompt: str) -> str | None:
    """Read a line from stdin on a daemon thread so Ctrl+C won't hang shutdown.

    This avoids blocking the default ThreadPoolExecutor that ``asyncio.to_thread``
    uses (which would hang on shutdown when joined). Returns ``None`` on EOF or
    if Ctrl+C is received while reading.
    """
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def _deliver(value: str | None) -> None:
        if not fut.done():  # may already be cancelled by Ctrl+C
            fut.set_result(value)

    def _read() -> None:
        try:
            line = input(prompt)
            loop.call_soon_threadsafe(_deliver, line)
        except (EOFError, KeyboardInterrupt):
            loop.call_soon_threadsafe(_deliver, None)
        except RuntimeError:
            # Event loop is closing; ignore.
            pass

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    return await fut


def list_files() -> str:
    """List the workspace's top-level entries, one name per line, sorted."""
    return "\n".join(sorted(p.name for p in WORKSPACE.iterdir())) or "(empty)"


def read_file(filename: str) -> str:
    """Read a UTF-8 text file from the workspace.

    Returns file contents, or an ``error: ...`` string if the path escapes the
    workspace or the file does not exist.
    """
    path = _safe_path(filename)
    if path is None:
        return "error: path is outside the workspace"
    return (
        path.read_text(encoding="utf-8") if path.is_file() else f"error: no {filename}"
    )


def write_file(filename: str, content: str) -> str:
    """Write (create or overwrite) a UTF-8 text file in the workspace.

    Parent directories are not created, so a nested path only works if its
    directory already exists. Returns a confirmation, or an ``error: ...``
    string if the path escapes the workspace. Other I/O failures (for example a
    missing parent directory) raise and are converted to error strings by
    ``call_tool``.
    """
    path = _safe_path(filename)
    if path is None:
        return "error: path is outside the workspace"
    path.write_text(content, encoding="utf-8")
    return f"wrote {filename}"


# Name -> callable for tools implemented in this module. ``call_tool`` invokes
# these with the model's JSON arguments as keyword arguments, so parameter
# names must match the schemas below.
LOCAL_TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "save_memory": save_memory,
}

# JSON schemas advertised to the model. MCP tools are appended at startup by
# ``connect_mcp``.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in the user's workspace folder.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one file from the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file in the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Save one short fact about the user to long-term memory. "
                "Use whenever you learn something lasting: their name, "
                "preferences, projects, recurring tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
            },
        },
    },
]


SYSTEM_PROMPT = """You are a helpful personal assistant running inside a \
custom harness. Be concise.

The user's workspace folder holds their personal files: notes, todo lists, \
ideas. You have tools to list, read, and write those files. Never claim you \
lack access to the user's files or tasks - use your tools to look.

Here is what you remember about the user from previous sessions:
{memory}

When you learn a new lasting fact about the user, save it with save_memory."""

# Conversation history in OpenAI chat format, shared across model switches.
# Populated in ``main`` rather than at import time so that importing this
# module has no side effects (no file reads, no network).
messages: list[dict] = []


# ---------------------------------------------------------------------------
# MCP integration
# ---------------------------------------------------------------------------

# Tool name -> the MCP session that serves it. Used by ``call_tool`` to route.
mcp_sessions: dict[str, ClientSession] = {}


async def connect_mcp(stack: AsyncExitStack) -> None:
    """Start the MCP servers in ``mcp_servers.json`` and register their tools.

    Each server is launched as a subprocess over stdio. Transports and sessions
    are entered into ``stack`` so they are shut down when the caller's stack
    exits. A server that fails to start or to list its tools is reported and
    skipped; an absent, unreadable, or malformed config file means no MCP
    servers. An MCP tool whose name matches a local tool, or one already
    registered by an earlier server, is skipped, because tool names must be
    unique in the schema list.

    Args:
        stack: Owns the lifetime of the server processes and sessions.
    """
    if not MCP_CONFIG.is_file():
        print("  [mcp] no mcp_servers.json - running without MCP servers")
        return

    try:
        config = json.loads(MCP_CONFIG.read_text(encoding="utf-8"))
        servers = config.get("mcpServers", {})
    except (OSError, json.JSONDecodeError, AttributeError) as e:
        # A malformed or unreadable config should degrade to "no MCP servers",
        # not stop the harness from starting.
        print(f"  [mcp] ignoring mcp_servers.json: {e}")
        return

    for name, spec in servers.items():
        params = StdioServerParameters(
            command=spec["command"], args=spec.get("args", [])
        )
        try:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            # list_tools() lives inside the same guard: a server that starts but
            # then fails to answer the tool listing is skipped too, rather than
            # crashing startup. Contexts already entered on the stack are still
            # torn down when the caller's stack exits.
            tools = (await session.list_tools()).tools
        except Exception as e:  # one broken server must not prevent startup
            print(f"  [mcp] '{name}' failed to start or list tools: {e}")
            continue
        for tool in tools:
            # Tool names must be unique across the whole schema list. Skip a name
            # already claimed by a local tool or by an earlier MCP server: two
            # servers exposing the same tool would otherwise append duplicate
            # functions to TOOL_SCHEMAS (which the API rejects) and overwrite the
            # mcp_sessions entry, misrouting calls. First registration wins.
            if tool.name in LOCAL_TOOLS or tool.name in mcp_sessions:
                print(f"  [mcp] skipping '{tool.name}': name already registered")
                continue
            mcp_sessions[tool.name] = session
            # The SDK exposes the tool's JSON schema as `inputSchema`.
            schema = getattr(tool, "inputSchema", None) or {
                "type": "object",
                "properties": {},
            }
            TOOL_SCHEMAS.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": schema,
                    },
                }
            )
        print(f"  [mcp] connected '{name}': {[t.name for t in tools]}")


async def call_tool(name: str, raw_args: str) -> str:
    """Execute one tool call and return its result as text.

    Dispatches to a local tool or the owning MCP session. This function never
    raises: any failure (malformed JSON, wrong arguments, I/O errors, MCP
    errors, unknown tool) is returned as an ``error: ...`` string so the model
    can see what went wrong and retry.

    Args:
        name: Tool name as requested by the model.
        raw_args: Arguments as the JSON string the model produced. Empty is
            treated as no arguments.

    Returns:
        The tool's output, or a string starting with ``error:``.
    """
    try:
        args = json.loads(raw_args or "{}")
        if name in LOCAL_TOOLS:
            return LOCAL_TOOLS[name](**args)
        if name in mcp_sessions:
            result = await mcp_sessions[name].call_tool(name, args)
            text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
            return f"error: {text}" if getattr(result, "isError", False) else text
        return f"error: unknown tool {name}"
    except Exception as e:  # the contract above: tool failures never propagate
        return f"error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


async def run_agent(
    user_message: str, client: Any = None, max_steps: int = MAX_STEPS
) -> str:
    """Run one user turn to completion and return the model's final answer.

    The loop sends the history and tool schemas to the active model, executes
    any tool calls it requests, appends the results, and repeats until the
    model replies without tool calls.

    On any failure, ``messages`` is restored to its state before this call so
    that a half-finished turn (for example, tool calls without results) cannot
    corrupt later requests.

    Args:
        user_message: The user's input for this turn.
        client: Object exposing ``chat.completions.create``. Defaults to an
            ``AsyncOpenAI`` client for the active model; tests inject a fake.
        max_steps: Maximum model round trips before giving up.

    Returns:
        The model's final text response.

    Raises:
        RuntimeError: The model was still requesting tools after ``max_steps``.
        Exception: Any error from the model API call is propagated unchanged.
    """
    cfg = MODELS[model_name]
    client = client or AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    checkpoint = len(messages)
    messages.append({"role": "user", "content": user_message})
    try:
        for _ in range(max_steps):
            response = await client.chat.completions.create(
                model=cfg.model, messages=messages, tools=TOOL_SCHEMAS
            )
            message = response.choices[0].message
            if not message.tool_calls:
                messages.append({"role": "assistant", "content": message.content or ""})
                return message.content or ""

            # Store as a plain dict so history stays portable across providers.
            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                print(f"  [tool] {call.function.name}({call.function.arguments})")
                result = await call_tool(call.function.name, call.function.arguments)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
        raise RuntimeError(f"no final answer after {max_steps} tool-call rounds")
    except Exception:
        del messages[checkpoint:]
        raise


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def handle_command(line: str) -> bool:
    """Handle a slash command typed at the prompt.

    Supported: /models, /model <name>, /tools, /memory, /quit.

    Args:
        line: Raw, stripped input line.

    Returns:
        True if the line was a command (including unknown ones, which print
        usage), False if it is ordinary chat input.

    Raises:
        SystemExit: On /quit.
    """
    global model_name
    if not line.startswith("/"):
        return False
    cmd, _, arg = line.partition(" ")
    if cmd == "/models":
        for name, cfg in MODELS.items():
            marker = "*" if name == model_name else " "
            print(f" {marker} {name:<12} {cfg.model}  ({cfg.base_url})")
    elif cmd == "/model":
        if arg in MODELS:
            model_name = arg
            print(f"  switched to {arg} ({MODELS[arg].model})")
        else:
            print(f"  unknown model '{arg}' - try /models")
    elif cmd == "/tools":
        for schema in TOOL_SCHEMAS:
            fn = schema["function"]
            origin = "local" if fn["name"] in LOCAL_TOOLS else "mcp"
            print(f"  [{origin}] {fn['name']}: {fn['description'][:60]}")
    elif cmd == "/memory":
        print(load_memory())
    elif cmd == "/quit":
        raise SystemExit
    else:
        print("  commands: /models /model <name> /tools /memory /quit")
    return True


async def main() -> None:
    """Create the workspace, start MCP servers, load memory, and chat."""
    if hasattr(sys.stdout, "reconfigure"):
        # Avoid UnicodeEncodeError on consoles that default to a legacy
        # encoding (notably Windows) when the model emits emoji.
        sys.stdout.reconfigure(encoding="utf-8")

    WORKSPACE.mkdir(exist_ok=True)
    messages[:] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(memory=load_memory())}
    ]

    async with AsyncExitStack() as stack:
        print("starting harness...")
        await connect_mcp(stack)
        print(
            f"\nharness ready - model: {model_name} "
            f"({MODELS[model_name].model}), "
            f"{len(TOOL_SCHEMAS)} tools, memory loaded"
        )
        print("type /models, /model <name>, /tools, /memory, or just talk\n")
        try:
            while True:
                line = await ainput("you: ")
                if line is None:
                    break
                line = line.strip()
                if not line or handle_command(line):
                    continue
                try:
                    print("\nassistant:", await run_agent(line), "\n")
                except Exception as e:  # keep the session alive after a failed turn
                    print(f"\n  [error] {type(e).__name__}: {e}\n")
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            print("\nbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        # Ctrl+C can surface while asyncio.run tears the event loop down, i.e.
        # outside the try/except inside main(), so catch it here too and exit
        # without dumping a KeyboardInterrupt traceback.
        print("\nbye!")
