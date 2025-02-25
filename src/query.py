#Todo: This needs to be moved to the framworks files
# code shared by frameworks to frameworks/utils.py or similar
import os
import torch
import faiss
from llama_index.core import Settings, load_index_from_storage
from llama_index.core import StorageContext
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from huggingface_hub import login
from transformers import AutoTokenizer, BitsAndBytesConfig
import tomllib

#Define some defaults
ll_name    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#TODO: refactor this to embedding_model_name
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model_dir  = "model"
index_dir  = "CouncilEmbeddings"
system_prompt = """Du bist ein intelligentes System, das deutsche Dokumente durchsucht und auf Basis der enthaltenen Informationen präzise Antworten auf gestellte Fragen gibt. Wenn du eine Antwort formulierst, gib die Antwort in klaren und präzisen Sätzen an und nenne dabei mindestens eine oder mehrere relevante Quellen im Format: (Quelle: Dokumentname, Abschnitt/Seite, Filename des TXT)."""

class RAG_LLM:

    def __init__(self, configfile):

        try:
            with open(configfile, "rb") as f:
                config = tomllib.load(f)
        except FileNotFoundError:
            raise Exception(f"missing configfile at {configfile}")

        try:
            self.token = config['api']['hf_key']
        except KeyError:
            raise Exception("API Key is requirerd in config")
        #is this correct for all models, or do we need to elaborate for local models
            
        self.llm_name = config.get('model',{}).get('llm_name') or llm_name
        self.model_name = config.get('model',{}).get('model_name') or model_name
        self.model_dir = config.get('model',{}).get('model_dir') or model_dir
        self.index_dir = config.get('source',{}).get('folderEmbeddings') or index_dir
        self.system_prompt = config.get('query',{}).get('system_prompt') or system_prompt
        self.huggingface_login(self.token)

        print(f"Model name: {self.llm_name}")

        self.embed_model = self._init_embedding_model()
        tokenizer, self.llm_model = self._init_llm_model()

        Settings.llm = self.llm_model
        # Settings.tokenizer = tokenizer
        Settings.embed_model = self.embed_model

        self.index = self._load_index_storage()
        self.query_engine = self._configure_query_engine()


    def huggingface_login(self, token):
        if not token:
            raise ValueError("Please set your Hugging Face token in the HUGGINGFACE_TOKEN environment variable.")
        login(token=token)
        print("Logged in successfully!")


    def _init_llm_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.llm_name, token=self.token)
        stopping_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            tokenizer.convert_tokens_to_ids("Query"),
            tokenizer.convert_tokens_to_ids("---------------"),
        ]
        # This will wrap the default prompts that are internal to llama-index
        query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        model = HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=1024,
            model_name=self.llm_name,
            model_kwargs={
                "token": self.token,
                # "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
                "quantization_config": quantization_config,
                "cache_dir": self.model_dir,
            },
            device_map="cuda",
            generate_kwargs={
                "do_sample": True, 
                "temperature": 0.3,
                "top_p": 0.9,
            },
            system_prompt=self.system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=self.llm_name,
            tokenizer_kwargs={
                "token": self.token,
                "cache_dir": self.model_dir,
            },
            stopping_ids=stopping_ids,
        )

        return tokenizer, model


    def _init_embedding_model(self):
        embedding_model = HuggingFaceEmbedding(model_name=self.embed_name)
        print(f"Embedding model {self.embed_name} initialized.")
        return embedding_model


    def _load_index_storage(self):

        faiss_store = FaissVectorStore.from_persist_dir(self.index_dir)
        storage_context = StorageContext.from_defaults(vector_store=faiss_store, persist_dir=self.index_dir)
        # storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
        print(f"Number of vectors stored: {faiss_store._faiss_index.ntotal}")
        print(f"Number of nodes in index: {len(index.ref_doc_info)}")

        return index


    def _configure_query_engine(self) -> RetrieverQueryEngine:
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=3,
        )

        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
        )

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        summary_prompt =  (
            "Nachfolgend sind passensten Kontextinformationen.\n"
            "---------------\n"
            "{context_str}\n"
            "---------------\n"
            f"{self.system_prompt}\n"
            "Query: {query_str}\n"
            "Antwort: "
        )
        prompt_template = PromptTemplate(summary_prompt)
        query_engine.update_prompts(
            {"response_synthesizer:summary_template": prompt_template}
        )

        return query_engine


    def _display_prompt_dict(self, prompts_dict):
        for k, p in prompts_dict.items():
            text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
            print(text_md)
            print(p.get_template())


    def query_rag_llm(self, user_query):
        # Function to interact with the query engine and return a response
        with torch.no_grad():
            response = self.query_engine.query(user_query)
        torch.cuda.empty_cache()
        return str(response)


    def search_relevant_documents(self, user_query) -> list[str]:
        """Retrieve relevant documents supporting the user query from the RAG query engine."""
        
        retrieved_nodes = self.query_engine.retriever.retrieve(user_query)
        retrieved_files = [node.metadata for node in retrieved_nodes]
        retrieved_texts = [node.text for node in retrieved_nodes]

        return retrieved_files, retrieved_texts


if __name__ == "__main__":

    configfile = os.path.expanduser(os.path.join('~','.config','hca','config.toml'))
    rag_llm = RAG_LLM(configfile)

    query = "Wie viele Unterlagen des Finanzausschusses sind vorhanden und welche sind das?"
    response = rag_llm.query_rag_llm(query)

    print("\n=================")
    print(query)
    print("---------------")
    print(response)
