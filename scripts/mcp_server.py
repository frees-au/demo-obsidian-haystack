"""Expose local semantic search over MCP Streamable HTTP or stdio."""

import argparse
from contextlib import redirect_stdout
import sys
from threading import Lock

from mcp.server.fastmcp import FastMCP
from query import SearchEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["streamable-http", "stdio"],
                        default="streamable-http")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    mcp = FastMCP("Obsidian Haystack", host="127.0.0.1", port=args.port,
                  stateless_http=True, json_response=True)
    lock = Lock()
    engine = None

    @mcp.tool()
    def search(query: str, top_k: int = 5) -> list[dict]:
        """Search indexed vault chunks by meaning. Return content, source, score and metadata.

        top_k must be between 1 and 50. Index the vault before searching.
        """
        nonlocal engine
        # Local Qdrant and the shared pipeline are used serially. Keep stdout
        # reserved for protocol messages when using the stdio transport.
        with lock, redirect_stdout(sys.stderr):
            if not query.strip() or not 1 <= top_k <= 50:
                raise ValueError("Provide a nonblank query and top_k between 1 and 50")
            if engine is None:
                engine = SearchEngine()
            return engine.search(query, top_k)

    try:
        mcp.run(transport=args.transport)
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    main()
