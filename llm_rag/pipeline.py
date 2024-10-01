import pickle
import torch
import faiss
from llama_index.core import Settings, load_index_from_storage
from llama_index.core import ServiceContext, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate


def load_llama_model(model_name, token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
    model.eval()
    return tokenizer, model

def initialize_embedding_model():
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbedding(model_name=model_name)
    print(f"Embedding model {model_name} initialized.")
    return embedding_model

    
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
    # token = "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"
    huggingface_login()
    model_name = "meta-llama/Llama-3.1-8B"
    embed_model = initialize_embedding_model()

    SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
    - Generate human readable output, avoid creating output with gibberish text.
    - Generate only the requested output, don't include any other language before or after the requested output.
    - Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
    - Generate professional language typically used in business documents in North America.
    - Never generate offensive or foul language.
    """

    query_wrapper_prompt = PromptTemplate(
        "[INST]<>\n" + SYSTEM_PROMPT + "<>\n\n{query_str}[/INST] "
    )

    llm_model = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=2048,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name=model_name,
        model_name=model_name,
        device_map="auto",
        # change these settings below depending on your GPU
        # model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
    )

    Settings.llm = llm_model
    Settings.embed_model = embed_model
    # faiss_path = "faiss_vectorstore_index.json"
    # faiss_store = load_faiss_index(faiss_index_path=faiss_path)

    # vector_store = FaissVectorStore.from_persist_dir("../preprocessing/vectorstore_index")
    storage_context = StorageContext.from_defaults(persist_dir="../preprocessing/vectorstore_index")
    # service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm_model)
    index = load_index_from_storage(storage_context)

    # Example query
    # query = "What was approved in the council meeting?"
    # answer = query_rag_system(query, llama_index, model, tokenizer)
    # print(f"Answer: {answer}")

    # set Logging to DEBUG for more detailed outputs
    query_engine = index.as_query_engine()
    query = "What was approved in the council meeting?"
    response = query_engine.query(query)
    print(response)
