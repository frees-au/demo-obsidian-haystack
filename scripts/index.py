from pathlib import Path
import argparse

from runtime import configure_logging

configure_logging()

from haystack import Pipeline
from haystack.components.converters import MultiFileConverter
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from embeddings import EMBEDDING_DIM, INDEX_NAME, MODEL_NAME, STORE_PATH
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

parser = argparse.ArgumentParser(description="Index a folder using all-MiniLM-L6-v2")
parser.add_argument("source", type=Path)
args = parser.parse_args()
source = args.source.resolve()
if not source.is_dir():
    parser.error(f"Source is not a directory: {source}")

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

if not files:
    parser.error("No supported files found in source folder")

store = QdrantDocumentStore(
    path=str(STORE_PATH),
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
    SentenceTransformersDocumentEmbedder(model=MODEL_NAME, normalize_embeddings=True),
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

try:
    pipeline.get_component("embedder").warm_up()
    pipeline.run({"converter": {"sources": files}})
finally:
    store.close()

print(f"Indexed {len(files)} files")

