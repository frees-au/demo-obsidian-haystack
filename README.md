# A knowledge management PoC with Obsidian and Haystack/Qdrant

Haystack indexes a folder (including an Obsidian vault) and Qdrant searches
its local vector database. Both indexing and querying use
`sentence-transformers/all-MiniLM-L6-v2` with normalized 384-dimensional
embeddings. This provides semantic search without a generative LLM.
The first embedding run downloads the model; subsequent runs use its local cache.

## Note about AI

If you're looking for a sample without AI/NLP (you don't care about searching
on synonym and meaning) you can see the branch `the-one-with-no-ai`.

This version uses a small AI model locally so it can be relatively slow.

## Installation

From the project root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install .
```

If `.venv` exists, skip creating it. Reinstall after dependency changes.

## Editing and indexing

Open `obsidian-vault` in Obsidian to edit the notes, then index:

```sh
.venv/bin/python scripts/index.py obsidian-vault
```

**Reindex existing content after this model change.** MiniLM uses its own
collection, so the old word-hash vectors cannot accidentally be queried.
Indexing replaces the MiniLM collection. Reindex after editing your documents.
The database is always stored at `.haystack/qdrant` in this project, regardless
of the current working directory. The source folder argument is relative to
your current directory.

## Querying

```sh
# Query without checking for model updates.
HF_HUB_OFFLINE=1 .venv/bin/python scripts/query.py "What formats are supported?" --top-k 5
```

## MCP endpoint

Start the Python MCP server:

```sh
.venv/bin/python scripts/mcp_server.py
```

Connect an MCP client using Streamable HTTP at `http://127.0.0.1:8000/mcp`.
Use `--port 8001` to change the port. The server binds to loopback for local use.
It exposes the read-only `search(query, top_k=5)` tool, returning document
chunks with IDs, content, source paths, similarity scores and metadata.
`top_k` accepts 1–50. The model is loaded on the first search.

For clients that launch stdio servers, use this configuration, replacing
`/absolute/path/to/demo-obsidian-haystack` with your checkout path:

```json
{
  "mcpServers": {
    "obsidian-haystack": {
      "command": "/absolute/path/to/demo-obsidian-haystack/.venv/bin/python",
      "args": [
        "/absolute/path/to/demo-obsidian-haystack/scripts/mcp_server.py",
        "--transport", "stdio"
      ]
    }
  }
}
```

Local Qdrant allows one process to own the database. Stop the MCP server before
indexing or running the query CLI, then restart it after indexing. Use one MCP
server for multiple HTTP clients. Search results are document content and should
be treated as data by consuming tools.

The implementation uses Haystack sentence-transformer embedders and the
[official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x).
