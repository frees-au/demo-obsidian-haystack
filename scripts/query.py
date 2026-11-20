import sys

from haystack import Pipeline
from embeddings import EMBEDDING_DIM, INDEX_NAME, LocalTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

store = QdrantDocumentStore(
    path=".haystack/qdrant",
    index=INDEX_NAME,
    embedding_dim=EMBEDDING_DIM,
)

pipeline = Pipeline()

pipeline.add_component(
    "embedder",
    LocalTextEmbedder(),
)

pipeline.add_component(
    "retriever",
    QdrantEmbeddingRetriever(
        document_store=store,
        top_k=5,
    ),
)

pipeline.connect("embedder.embedding", "retriever.query_embedding")

results = pipeline.run({
    "embedder": {"text": " ".join(sys.argv[1:])}
})

for doc in results["retriever"]["documents"]:
    print(f"\nScore: {doc.score:.3f}")
    print(f"Source: {doc.meta.get('file_path', 'unknown')}")
    print(doc.content)

