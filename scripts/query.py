"""Semantic search shared by the CLI and MCP server."""

import argparse

from runtime import configure_logging

configure_logging()

from haystack import Pipeline
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from embeddings import EMBEDDING_DIM, INDEX_NAME, MODEL_NAME, STORE_PATH
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


class SearchEngine:
    def __init__(self):
        self.store = QdrantDocumentStore(
            path=str(STORE_PATH), index=INDEX_NAME, embedding_dim=EMBEDDING_DIM,
        )
        if not self.store.count_documents():
            self.close()
            raise RuntimeError("No MiniLM documents indexed. Run scripts/index.py first.")
        self.pipeline = Pipeline()
        self.pipeline.add_component("embedder", SentenceTransformersTextEmbedder(
            model=MODEL_NAME, normalize_embeddings=True, progress_bar=False,
        ))
        self.pipeline.add_component("retriever", QdrantEmbeddingRetriever(
            document_store=self.store, top_k=5,
        ))
        self.pipeline.connect("embedder.embedding", "retriever.query_embedding")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip():
            raise ValueError("Query must not be blank")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        self.pipeline.get_component("embedder").warm_up()
        results = self.pipeline.run({
            "embedder": {"text": query}, "retriever": {"top_k": top_k},
        })
        return [
            {"id": doc.id, "score": doc.score,
             "source": str(doc.meta.get("file_path", "unknown")),
             "content": doc.content, "metadata": doc.meta}
            for doc in results["retriever"]["documents"]
        ]

    def close(self):
        self.store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="+")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    engine = SearchEngine()
    try:
        for doc in engine.search(" ".join(args.query), args.top_k):
            print(f"\nScore: {doc['score']:.3f}")
            print(f"Source: {doc['source']}")
            print(doc["content"])
    finally:
        engine.close()


if __name__ == "__main__":
    main()
