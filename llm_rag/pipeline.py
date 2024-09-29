import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import VectorStoreIndex

from huggingface_hub import login


def load_llama_model(model_name, token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
    model.eval()
    return tokenizer, model


def load_precomputed_faiss(faiss_index_path, docstore_path):
    # Load FAISS index from file
    faiss_index = faiss.read_index(faiss_index_path)
    
    # Load the document store (document metadata)
    with open(docstore_path, 'rb') as f:
        docstore_data = pickle.load(f)
    
    docstore = InMemoryDocstore(docstore_data)
    index_to_docstore_id = {i: i for i in range(len(docstore_data))}
    
    faiss_store = FAISS(
        embedding_function=None,
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    
    return faiss_store


def initialize_llama_index(faiss_store):
    storage_context = StorageContext.from_vector_store(faiss_store)
    index = VectorStoreIndex([], storage_context=storage_context)
    return index


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
    model_name = "meta-llama/Llama-3.1-8B"  # Use Llama 2 if Llama 3 is unavailable
    tokenizer, model = load_llama_model(model_name, token)
    
    faiss_store = load_precomputed_faiss(faiss_index_path="../preprocessing/faiss_index", docstore_path="../preprocessing/faiss_index/faiss_store.pkl")
    
    # Initialize LlamaIndex with the loaded FAISS store
    llama_index = initialize_llama_index(faiss_store)

    # Example query
    query = "What was approved in the council meeting?"
    answer = query_rag_system(query, llama_index, model, tokenizer)
    print(f"Answer: {answer}")