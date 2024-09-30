import pickle
import torch
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.index_store import SimpleIndexStore
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_llama_model(model_name, token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
    model.eval()
    return tokenizer, model


def load_faiss_index(faiss_index_path="faiss_index", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # Load FAISS store from the saved directory with the embedding model used for indexing
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    faiss_store = FAISS.load_local(faiss_index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
    print(f"FAISS store loaded from {faiss_index_path}.")
    return faiss_store


def load_llama_index(faiss_store):
    docstore = InMemoryDocstore({})
    index_store = SimpleIndexStore()
    vector_stores = {"default_namespace": faiss_store}  # The FAISS store with a namespace
    
    # Initialize StorageContext with docstore, index_store, and vector_stores
    storage_context = StorageContext(
        docstore=docstore,
        index_store=index_store,
        vector_stores=vector_stores,
        graph_store=None  # Optional
    )
    print("StorageContext initialized.")

    Settings.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llama_index = VectorStoreIndex([], storage_context=storage_context)
    print("LlamaIndex initialized.")
    
    return llama_index

    
def huggingface_login():
    token = "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"
    # token = os.getenv("HUGGINGFACE_TOKEN")  # Use environment variable for security
    if not token:
        raise ValueError("Please set your Hugging Face token in the HUGGINGFACE_TOKEN environment variable.")
    login(token=token)
    print("Logged in successfully!")


def query_rag_system(query, llama_index, model, tokenizer):
    # Retrieve relevant documents using LlamaIndex
    docs = llama_index.query(query, top_k=3)
    
    # Concatenate retrieved documents and pass them to the Llama model for answer generation
    context = "\n".join([doc.page_content for doc in docs])
    input_text = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():  # No gradients needed for inference
        outputs = model.generate(**inputs, max_new_tokens=150)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    token = "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"
    model_name = "meta-llama/Llama-3.1-8B"
    # tokenizer, model = load_llama_model(model_name, token)
    
    faiss_path = "../preprocessing/faiss_index"
    faiss_store = load_faiss_index(faiss_index_path=faiss_path)
    llama_index = load_llama_index(faiss_store)

    # Example query
    query = "What was approved in the council meeting?"
    answer = query_rag_system(query, llama_index, model, tokenizer)
    print(f"Answer: {answer}")