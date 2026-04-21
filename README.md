# memra-local

Local-first memory server for AI agents. Offline, private, fast.

`memra-local` gives your coding agent persistent memory that lives entirely on your machine — no account, no network, no data leaves the laptop. Works with Claude Code, Cursor, Zed, Droid, Hermes Agent, OpenClaw, and any MCP-compatible client.

When you're ready to sync across devices or share with a team, a single command pushes your local namespace to [Memra Cloud](https://usememra.com). Same tools, same API, your choice.

## Install

```bash
pip install memra-local
memra mcp          # start the MCP server
```

Requires Python 3.10+.

## Wire it into your editor

### Claude Code / Cursor

```json
{
  "mcpServers": {
    "memra": {
      "command": "memra",
      "args": ["mcp"]
    }
  }
}
```

### Zed

```json
{
  "context_servers": {
    "memra": {
      "command": { "path": "memra", "args": ["mcp"] }
    }
  }
}
```

### Droid (Factory.ai) / Hermes Agent / OpenClaw

See [usememra.com/install](https://usememra.com/install) for client-specific snippets.

## What you get

- **Flat-file memory** in `~/.memra/` — plain YAML, inspectable, greppable, diff-able
- **MCP server** exposing `memra_add`, `memra_recall`, `memra_get`, `memra_list`, `memra_supersede`, `memra_history`, and more
- **Local embeddings** via `sentence-transformers` — no OpenAI key required
- **Sync to cloud** optional: `memra sync enable <namespace> --api-key memra_live_...`

## Commands

```bash
memra mcp          # MCP server over stdio
memra status       # verify server + list namespaces
memra hooks install  # optional — auto-capture decisions/patterns as you work
memra --help       # full CLI reference
```

## Docs + source

- Install snippets and client configs: https://usememra.com/install
- Memra Cloud (hosted EU): https://usememra.com
- Source: https://github.com/usememra/memra-local

## License

[BUSL-1.1](./LICENSE). Change Date 2030-04-17 — on that date the license auto-converts to Apache-2.0. Until then, personal and non-production use are unrestricted; commercial production use requires a separate license.
