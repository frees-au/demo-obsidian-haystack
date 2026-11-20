from pathlib import Path
import sys

from haystack import Pipeline
from haystack.components.converters import MultiFileConverter
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from embeddings import EMBEDDING_DIM, INDEX_NAME, LocalDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

source = Path(sys.argv[1]).resolve()

store = QdrantDocumentStore(
    path=".haystack/qdrant",
    index=INDEX_NAME,
    embedding_dim=EMBEDDING_DIM,
    recreate_index=True,
)

pipeline = Pipeline()

pipeline.add_component("converter", MultiFileConverter())
pipeline.add_component("cleaner", DocumentCleaner())
pipeline.add_component(
    "splitter",
    DocumentSplitter(
        split_by="word",
        split_length=250,
        split_overlap=50,
    ),
)
pipeline.add_component(
    "embedder",
    LocalDocumentEmbedder(),
)
pipeline.add_component(
    "writer",
    DocumentWriter(
        document_store=store,
        policy=DuplicatePolicy.OVERWRITE,
    ),
)

pipeline.connect("converter", "cleaner")
pipeline.connect("cleaner", "splitter")
pipeline.connect("splitter", "embedder")
pipeline.connect("embedder", "writer")

extensions = {
    ".csv", ".docx", ".html", ".json", ".md",
    ".pdf", ".pptx", ".txt", ".xlsx",
}

files = [
    p for p in source.rglob("*")
    if p.is_file()
    and p.suffix.lower() in extensions
    and not any(part.startswith(".") for part in p.relative_to(source).parts)
]

pipeline.run({"converter": {"sources": files}})

print(f"Indexed {len(files)} files")

