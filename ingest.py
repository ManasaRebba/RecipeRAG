import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_PATH = "Recipe_books"
DB_PATH = "vector_db"

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=250,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for doc in documents:

        # Ignore tiny pages
        if len(doc.page_content.strip()) < 100:
            continue

        chunks.extend(
            text_splitter.split_documents([doc])
        )

    print(f"Created {len(chunks)} chunks.")

    return chunks

def create_vector_store(chunks):
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=DB_PATH
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    create_vector_store(chunks)