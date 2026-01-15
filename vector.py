from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Define the embedding model
embeddings = OllamaEmbeddings(model="bge-m3")

# Define the database location
db_location = "./chrome_langchain_db"

# Initialize the vector store
vector_store = Chroma(
    collection_name="rag_collection",
    persist_directory=db_location,
    embedding_function=embeddings
)

documents_path = "./documents"
if not os.path.exists(documents_path):
    os.makedirs(documents_path)

# Only process if DB seems new (folder doesn't exist)
if not os.path.exists(db_location):
    print("Initializing vector database...")
    loaders = [
        DirectoryLoader(documents_path, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(documents_path, glob="**/*.md", loader_cls=TextLoader),
        DirectoryLoader(documents_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    ]
    docs = []
    for loader in loaders:
        try:
            print(f"Loading files with {loader.loader_cls.__name__}...")
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            print(f"Loaded {len(loaded_docs)} documents.")
        except Exception as e:
            print(f"Error loading files with {loader}: {e}")

    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vector_store.add_documents(documents=splits)
        print(f"Added {len(splits)} chunks to the database.")
    else:
        print("No documents found to add. Place .txt, .md, or .pdf files in the 'documents' folder and delete the 'chrome_langchain_db' folder to try again.")
else:
    print("Vector database already exists. Using existing data.")
    print("To refresh data: Delete the 'chrome_langchain_db' folder and run this script again.")

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

def add_file_to_db(file_path):
    """
    Adds a single file to the vector database.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
        
    file_ext = os.path.splitext(file_path)[1].lower()
    loader = None
    
    if file_ext == ".txt" or file_ext == ".md":
        loader = TextLoader(file_path)
    elif file_ext == ".pdf":
        loader = PyPDFLoader(file_path)
    # Add CSV support
    elif file_ext == ".csv":
        from langchain_community.document_loaders import CSVLoader
        loader = CSVLoader(file_path)
    else:
        print(f"Unsupported file type: {file_ext}")
        return False
        
    try:
        print(f"Processing {file_path}...")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vector_store.add_documents(documents=splits)
        print(f"Successfully added {file_path} to database.")
        return True
    except Exception as e:
        print(f"Error adding file: {e}")
        return False
