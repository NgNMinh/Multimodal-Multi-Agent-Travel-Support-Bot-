import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)

file_path = "assets/tourist_destination.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()
full_docs = ""
for doc in docs:
    full_docs = full_docs + doc.page_content

all_splits = [Document(page_content=txt) for txt in re.split(r"(?m)^(?=\d+\.\s)", full_docs)]
all_splits = all_splits[1:]

_ = vector_store.add_documents(documents=all_splits)