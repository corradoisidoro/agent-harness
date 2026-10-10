# agent-harness

[![Python](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/release/python-3140/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-15%20passed-brightgreen.svg)](#tests-and-lint)
[![Ruff](https://img.shields.io/badge/ruff-passing-brightgreen.svg)](https://docs.astral.sh/ruff/)
[![CI](https://github.com/corradoisidoro/agent-harness/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/corradoisidoro/agent-harness/actions/workflows/ci.yml)

A minimal personal-assistant agent harness. It connects one OpenAI-compatible
chat endpoint to a small set of built-in tools (workspace file access and
persistent memory) plus any tools exposed by [MCP](https://modelcontextprotocol.io)
servers, and runs an interactive chat loop in the terminal.

The agent loop is provider-agnostic: any endpoint that speaks the OpenAI
chat-completions protocol works. Today it ships configured for a local
[Ollama](https://ollama.com) server.

## Features

- Chat with the model from a terminal REPL.
- Built-in tools for the user's `workspace/` folder: list, read, and write files.
- Persistent memory: facts the model chooses to remember are appended to
  `memory.md` and loaded at the start of every session.
- MCP support: tools from any number of MCP servers are registered and used
  alongside the built-in ones.
- Multiple named models with runtime switching.
- Failures are contained: a broken MCP server, a bad tool call, or a failed
  model request never crashes the session.

## Requirements

- Python 3.14+ and [uv](https://docs.astral.sh/uv/)
- A local Ollama server with the default models pulled:

  ```sh
  ollama pull qwen3.5:4b
  ollama pull qwen3.5:2b
  ```

- `uvx` on `PATH` if you want the example MCP servers (it ships with uv).

## Install

```sh
uv sync
```

This creates `.venv/` and installs the runtime plus dev dependencies.

## Run

Make sure Ollama is running, then:

```sh
uv run harness.py
```

You'll see the MCP servers connect, then a `you:` prompt.

## Examples

Type prompts in plain language; the agent decides which tools to use.

| Ask | Demonstrates |
| --- | --- |
| `What files are in my workspace?` | `list_files` |
| `Give me the contents of notes.txt.` | `read_file` |
| `Add a "groceries" section to notes.txt.` | `read_file` then `write_file` |
| `Create ideas.txt with three project ideas.` | `write_file` |
| `Remember that I prefer short bullet-point answers.` | `save_memory`, persisted to `memory.md` |
| `What do you remember about me?` | memory is loaded into the system prompt |
| `What is the current time?` | `time` MCP server |
| `Summarize https://modelcontextprotocol.io/docs/2026-07-28/getting-started/intro and save it to a new file in my workspace.` | `fetch` MCP server then `write_file` |
| `List my workspace, summarize every file, and write the result to summary.txt.` | several tools in one turn |
| `Read does-not-exist.txt.` | a missing file returns `error: ...` to the model; the session continues |

The `save_memory` row only shows up across sessions, since memory is loaded
once at startup. MCP-backed rows need the `time` and `fetch` servers from
`mcp_servers.json` (see [MCP servers](#mcp-servers)); the file and memory rows
work without MCP.

## Commands

| Command | Description |
| --- | --- |
| `/models` | List the configured models, marking the active one. |
| `/model <name>` | Switch the active model for subsequent turns. |
| `/tools` | List every registered tool (local and MCP). |
| `/memory` | Print what the agent remembers about you. |
| `/quit` | Exit. |

Anything that doesn't start with `/` is sent to the model.

## Configuration

### Models

Models are defined in the `MODELS` registry in `harness.py`. Each entry is an
OpenAI-compatible endpoint (`base_url`, `api_key`, `model`). Adding a model is
just adding an entry; the agent loop needs no other changes. `api_key` is
ignored by local servers such as Ollama.

### MCP servers

MCP servers are listed in `mcp_servers.json`:

```json
{
  "mcpServers": {
    "time": { "command": "uvx", "args": ["mcp-server-time"] },
    "fetch": { "command": "uvx", "args": ["mcp-server-fetch"] }
  }
}
```

Each server is launched as a subprocess over stdio. Delete the file (or your
entries) to run without MCP. A server that fails to start, or a malformed
config, is reported and skipped rather than stopping the harness.

### Workspace and memory

- `workspace/` holds the user's files. It is created on first run. All file
  paths are confined to this folder - absolute paths and `..` escapes are
  rejected.
- `memory.md` is the agent's long-term memory. It is created on first save and
  loaded once at startup, so a fact saved in one session is known in the next.

Both are ignored by git.

## Project layout

```
harness.py         # the whole harness: model registry, tools, agent loop, REPL
test_harness.py    # pytest suite (file tools, memory, call_tool, agent loop)
mcp_servers.json   # MCP server definitions
pyproject.toml     # project metadata and dependencies
README.md
LICENSE
workspace/         # local user files (gitignored)
memory.md          # local agent memory (gitignored)
```

## Tests and lint

```sh
uv run pytest
uv run ruff check .
```

## License

MIT - see [LICENSE](LICENSE).
