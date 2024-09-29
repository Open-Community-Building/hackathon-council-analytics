import pickle
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from llama_index.core import VectorStoreIndex
from llama_index.core import Document as LlamaDocument
from llama_index.core import Settings
import os
from tqdm import tqdm

from preprocessing import download_from_nextcloud


def load_txt_files(start_idx, end_idx):
    documents = []
    for idx in tqdm(range(start_idx, end_idx + 1), desc="Loading documents", unit="docs"):
        filename = f"{idx}.txt"
        content = download_from_nextcloud("CouncilDocuments", filename)
        if content:
            # Convert to LlamaIndex Document format
            documents.append(LlamaDocument(text=content, metadata={"name": filename}))
        else:
            print(f"Skipping {filename} due to download error.")
    return documents


def initialize_faiss_store(embedding_model):
    # Get embedding dimensions
    test_embedding = embedding_model.embed_query("test")
    embedding_dim = len(test_embedding)
    
    # Initialize FAISS index
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    print(f"FAISS index initialized with embedding dimension: {embedding_dim}")
    
    # Initialize document store and FAISS store
    docstore = InMemoryDocstore({})
    index_to_docstore_id = {}
    faiss_store = FAISS(embedding_function=embedding_model.embed_query, 
                        index=faiss_index, 
                        docstore=docstore, 
                        index_to_docstore_id=index_to_docstore_id)
    print("FAISS store initialized.")
    
    return faiss_store


def initialize_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model initialized.")
    return embedding_model


def initialize_llama_index(embedding_model):
    Settings.embed_model = embedding_model
    index = VectorStoreIndex([])
    print("LlamaIndex initialized.")
    return index


def add_documents_to_stores(documents, llama_index, faiss_store):
    for doc in tqdm(documents, desc="Embedding documents", unit="docs"):
        llama_index.insert(doc)  # Add to LlamaIndex
        faiss_store.add_texts([doc.text], metadatas=[doc.metadata])  # Add documents to FAISS for retrieval
    print(f"Added {len(documents)} documents to LlamaIndex and FAISS.")


def save_index(llama_index, faiss_store):
	# Save LlamaIndex
    filename = "llama_index.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(llama_index, f)  # Save any additional metadata if needed
    print(f"LlamaIndex metadata saved to {filename}.")

	# Save FAISS index
    filename = "faiss_index"
    faiss_store.save_local(filename)
    print(f"FAISS index saved successfully as {filename}.")


if __name__ == "__main__":
    
    embedding_model = initialize_embedding_model()
    faiss_store = initialize_faiss_store(embedding_model)
    llama_index = initialize_llama_index(embedding_model)
    
    documents = load_txt_files(start_idx=200010, end_idx=200020)
    
    if documents:
        add_documents_to_stores(documents, llama_index, faiss_store)
        save_index(llama_index, faiss_store)
    else:
        print("No documents loaded for embedding.")
