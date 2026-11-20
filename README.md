# A knowledge management PoC with Obsidian and Haystack/Qdrant

Light local-only setup that supports vectorising and searching a
folder of data which also happens to be an Obsidian vault.

Haystack prepares the vectors and Qdrant searches the local vector database.

## Caveats

Zero LLM in this demo. So be clear that it *does not* understand synonyms
or meaning like a trained semantic embedding model. If you want to make it smarter,
use a prompt like this:

```
  Add semantic search using a sentence-transformer model so queries
  can find relevant documents even when they use different words. Remind
  me to reindex the content.
```

While not necessarily slow, this is not performance tuned. It's supposed show
a working setup that builds a vector database and is theoretically web-scale
in architecture.

## Installation

From the project root, create a virtual environment and install the project
and its dependencies from `pyproject.toml`:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install .
```

If `.venv` already exists, skip the first command. Run the install command
again after changing the dependencies in `pyproject.toml`.

## Editing

Open the obsidian-vault folder in Obsidian. Update files and stuff.

## Indexing

Point index.py at a folder of stuff.

`.venv/bin/python ./scripts/index.py obsidian-vault`

## Querying

This matches.

`.venv/bin/python  ./scripts/query.py "hello world in obsidian vault"`

This won't work unless you add sentence-transformer (see Caveats).

`.venv/bin/python  ./scripts/query.py "What formats are supported"`
