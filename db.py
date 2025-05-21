from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import re 
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)

# client = MongoClient("mongodb://localhost:27017")
# db = client["flight_booking"]
# collection = db["tourist_destination"]

# vector_store = MongoDBAtlasVectorSearch(
#     embedding=embeddings,
#     collection=collection,
#     index_name="my_vector_index",
#     relevance_score_fn="cosine",
# )

file_path = "tourist_destination.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()
full_docs = ""
for doc in docs:
    full_docs = full_docs + doc.page_content


all_splits = [Document(page_content=txt) for txt in re.split(r"(?m)^(?=\d+\.\s)", full_docs)]
all_splits = all_splits[1:]

_ = vector_store.add_documents(documents=all_splits)


# docs = vector_store.similarity_search("địa điểm du lịch ở Việt Nam")
# final= "\n\n".join([doc.page_content for doc in docs])
# print(final)