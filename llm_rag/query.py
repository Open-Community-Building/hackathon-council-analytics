import os
import torch
import faiss
from llama_index.core import Settings, load_index_from_storage
from llama_index.core import ServiceContext, StorageContext
from llama_index.core import PromptTemplate
from llama_index.core.prompts import QueryWrapperPrompt
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM


def huggingface_login():
    token = "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"
    # token = os.getenv("HUGGINGFACE_TOKEN")  # Use environment variable for security
    if not token:
        raise ValueError("Please set your Hugging Face token in the HUGGINGFACE_TOKEN environment variable.")
    login(token=token)
    print("Logged in successfully!")


def init_llm_model(llm_name, token):
    tokenizer = AutoTokenizer.from_pretrained(llm_name, token=token)
    stopping_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    model = HuggingFaceLLM(
        model_name=llm_name,
        model_kwargs={
            "token": token,
            "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
            # "quantization_config": quantization_config
        },
        device_map="cuda",
        generate_kwargs={
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
        },
        tokenizer_name=llm_name,
        tokenizer_kwargs={"token": token},

        stopping_ids=stopping_ids,
    )

    return tokenizer, model


def init_embedding_model(embed_name):
    embedding_model = HuggingFaceEmbedding(model_name=embed_name)
    print(f"Embedding model {embed_name} initialized.")
    return embedding_model

    
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
    huggingface_login()
    embed_name="sentence-transformers/all-MiniLM-L6-v2"
    llm_name = "meta-llama/Meta-Llama-3.1-8B"
    index_dir = "../CouncilEmbeddings/vectorstore_index"
    # llm_name = "Intel/dynamic_tinybert"
    embed_model = init_embedding_model(embed_name)

    # SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
    # - Generate human readable output, avoid creating output with gibberish text.
    # - Generate only the requested output, don't include any other language before or after the requested output.
    # - Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
    # - Generate professional language typically used in business documents in North America.
    # - Never generate offensive or foul language.
    # """

    # query_wrapper_prompt = PromptTemplate(
    #     "[INST]<>\n" + SYSTEM_PROMPT + "<>\n\n{query_str}[/INST] "
    # )

    tokenizer, llm_model = init_llm_model(llm_name=llm_name, token=token)

    Settings.llm = llm_model
    Settings.embed_model = embed_model

    faiss_index = faiss.read_index(os.path.join(index_dir, "faiss_index.idx"))
    faiss_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=faiss_store, persist_dir=index_dir)
    index = load_index_from_storage(storage_context)

    print(f"Number of vectors stored: {faiss_index.ntotal}")
    print(f"Number of nodes in index: {len(index.ref_doc_info)}")

    # Example query
    # query = "What was approved in the council meeting?"
    # answer = query_rag_system(query, llama_index, model, tokenizer)
    # print(f"Answer: {answer}")

    german_prompt = QueryWrapperPrompt(
        template="Lies die folgenden Informationen sorgfältig und beantworte die folgende Frage basierend auf meinen Dokumenten auf Deutsch: {query}"
    )
    query_engine = index.as_query_engine(query_wrapper_prompt=german_prompt)


    query = "Welche Themen wurden in der letzten Sitzung des Gemeinderates besprochen?"
    response = query_engine.query(query)
    print("\n=================")
    print(query)
    print("---------------")
    print(response)