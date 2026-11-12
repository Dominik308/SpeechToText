"""
Vector store (RAG) for the assistant.

All documents live in ./documents. On startup every file in that folder that is
not yet in the vector database is indexed, so the database can always be rebuilt
by deleting the database folder.
"""
import os

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "bge-m3"
DB_LOCATION = "./chrome_langchain_db"
DOCUMENTS_PATH = "./documents"
SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf", ".csv")

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name="rag_collection",
    persist_directory=DB_LOCATION,
    embedding_function=embeddings,
)

_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def _normalize(path):
    return os.path.normpath(path)


def _get_loader(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".txt", ".md"):
        return TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    if ext == ".csv":
        return CSVLoader(file_path, autodetect_encoding=True)
    return None


def _source_ids(source):
    """Return the chunk ids that belong to a given source file."""
    return vector_store.get(where={"source": source}, include=[])["ids"]


def add_file_to_db(file_path):
    """
    Adds (or re-indexes) a single file in the vector database.
    Existing chunks of the same file are replaced, so adding a file twice
    does not create duplicate results.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False

    loader = _get_loader(file_path)
    if loader is None:
        print(f"Unsupported file type: {os.path.splitext(file_path)[1]}")
        return False

    source = _normalize(file_path)
    try:
        print(f"Processing {file_path}...")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = source
        splits = _text_splitter.split_documents(docs)

        old_ids = _source_ids(source)
        if old_ids:
            vector_store.delete(ids=old_ids)
        if splits:
            vector_store.add_documents(documents=splits)
        print(f"Successfully added {file_path} ({len(splits)} chunks).")
        return True
    except Exception as e:
        print(f"Error adding file {file_path}: {e}")
        return False


def sync_documents_folder():
    """Index every supported file in DOCUMENTS_PATH that is not in the database yet."""
    os.makedirs(DOCUMENTS_PATH, exist_ok=True)
    added = 0
    for root, _dirs, files in os.walk(DOCUMENTS_PATH):
        for name in sorted(files):
            if not name.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            path = _normalize(os.path.join(root, name))
            try:
                if _source_ids(path):
                    continue
            except Exception as e:
                print(f"Could not query vector database: {e}")
                return added
            if add_file_to_db(path):
                added += 1
    if added:
        print(f"Indexed {added} new document(s) from {DOCUMENTS_PATH}.")
    return added


sync_documents_folder()

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
