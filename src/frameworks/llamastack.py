import faiss
import pickle
import torch
from llama_index.core import (VectorStoreIndex,
                              StorageContext,
                              PromptTemplate,
                              Document,
                              Settings,
                              load_index_from_storage,
                              load_indices_from_storage,
                              get_response_synthesizer)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from preprocessor import Preprocessor
from huggingface_hub import login
from transformers import AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from typing import Optional
from utils import vprint, is_docker
import os

"""
This Module provides functions to work on the vector store

Classes:
- 'Helper': common functions for the other classes
- 'Embedor': provide embedding functions
- 'Query':   provide query function

Functions:

Example Usage:


"""
# Defaults
index_dir = "/media/CouncilEmbeddings"
llm_model_name    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#TODO: refactor this to embedding_model_name
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding_dim = 1024 # embedding dimension given by Ollama embedding
model_dir  = "model"
system_prompt = """Du bist ein intelligentes System, das deutsche Dokumente durchsucht und auf Basis der enthaltenen Informationen präzise Antworten auf gestellte Fragen gibt. Wenn du eine Antwort formulierst, gib die Antwort in klaren und präzisen Sätzen an und nenne dabei mindestens eine oder mehrere relevante Quellen im Format: (Quelle: Dokumentname, Abschnitt/Seite, Filename des TXT)."""


# TODO: this is named index_dir in query.py
# storage_dir = "vectorstore_index"

    
class Helper:
    """
    helper class provides common functions
    """
    def __init__(self, config: dict) -> None:
        """
        initialize the helper class
        params: config: the configuration dict
        """
        self.config = config
        self.llm_model_name = config.get('model', {}).get('llamastack',{}).get('llm_model_name') or llm_model_name
        if config.get('embedding', {}).get('faiss'):
            #ToDo: catch key errors here
            self.embedding_model_name = config['embedding']['faiss']['embedding_model_name']
            self.embedding_dim = config['embedding']['faiss']['embedding_dim']
        else:
            self.embedding_model_name = config.get('embedding', {}).get('embedding_model_name') or embedding_model_name
            self.embedding_dim = config.get('embedding', {}).get('embedding_dim') or embedding_dim
        self.index_dir = config.get('embedding', {}).get('faiss',{}).get('index_dir') or index_dir
        self.faiss_index_path = os.path.join(self.index_dir, "faiss_index.idx")

    def initialize_embedding_model(self):
        """
        initialise the embedding model
        """
        # embedding_model = HuggingFaceEmbedding(model_name=self.embedding_model_name)
        base_url = "http://hca-ollama-cpu:11434" if is_docker() else "http://localhost:11434"
        embedding_model = OllamaEmbedding(
            model_name="mxbai-embed-large",
            base_url=base_url,
            ollama_additional_kwargs={"prostatic": 0},
        )
        if embedding_model:
            vprint(f"Embedding model '{embedding_model.model_name}' initialized.", self.config)
        return embedding_model

    def init_faiss_index(self) -> faiss.IndexFlatL2:
        faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        return faiss_index

    def get_faiss_index(self) -> faiss.IndexFlatL2:
        """
        get the existing faiss index from index_dir and return if exits
        else create one
        """
        faiss_index = faiss.read_index(self.faiss_index_path)
        vprint(f"Loaded FAISS index with {faiss_index.ntotal} vectors.", self.config)
        return faiss_index

    def init_vector_store(self) -> FaissVectorStore:
        faiss_index = self.init_faiss_index()
        faiss_store = FaissVectorStore(faiss_index=faiss_index)
        return faiss_store


    def get_vector_store(self) -> FaissVectorStore:
        """
        get an existing vector_store
        """
        faiss_store = FaissVectorStore.from_persist_dir(self.index_dir)
        return faiss_store

    def init_storage_context(self):
        faiss_store = self.init_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=faiss_store)
        return storage_context


    def get_storage_context(self):
        faiss_store = self.get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=faiss_store, persist_dir=self.index_dir)
        return storage_context



class Embedor:
    """
    class Emebedor embedes textfiles in vector store
    """
    def __init__(self, config: dict, secrets: dict) -> None:
        """
        Constructs all the necessary attributes for the Embedor object.
        params: config: the configuration dict
        """
        self.config = config
        self.helper = Helper(config)
        self.pp = Preprocessor(config=config, secrets=secrets)
        self.fs = self.pp.fs
        self.index_dir = self.helper.index_dir
        self.metadata_path = os.path.join(self.index_dir, "document_metadata.pkl")




    def embed(self, start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> int:
        """
        This function is called from the admin interface
        """
        documents = self.fs.get_documents(start_idx, end_idx)
        index = self.embed_and_index_documents(documents)
        return len(index.ref_doc_info)

    def update_faiss_index(self, start_idx: int, end_idx: int) -> int:
        """
        Update index and metadata
        """
        # Load only new documents
        # self.load_txt_files_from_directory(directory, processed_filenames=document_metadata.keys())
        document_metadata = self.get_document_metadata()
        new_documents = self.fs.get_documents(start_idx, end_idx, exclude_filenames=document_metadata.keys())
        if new_documents:
            self.embed_and_index_documents(new_documents, document_metadata)
            vprint(f"Added {len(new_documents)} new documents to the index.", self.config)
        else:
            vprint("No new documents found.", self.config)
        return len(document_metadata)

    def embed_and_index_documents(self,
                                  documents: list,
                                  document_metadata: Optional[dict] = None) -> VectorStoreIndex:
        """
        Embed and index the documents given
        update metadata
        save.
        """
        storage_context = self.helper.init_storage_context()
        # Configure text splitter for chunking
        Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

        llama_documents = self.build_llama_documents(documents)
        # index = load_index_from_storage(storage_context)
        # Embed and add new documents to the FAISS index
        index = VectorStoreIndex.from_documents(
            llama_documents,
            storage_context=storage_context,
            embed_model=self.helper.initialize_embedding_model(),
            transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)],
            show_progress=True,
        )

        if document_metadata:
            # is update
            for doc in llama_documents:
                document_metadata[doc.metadata["filename"]] = doc.metadata
            # Save the updated FAISS index and metadata
            self.save_index_and_metadata(vector_store._faiss_index, document_metadata)
        else:
            # is embed
            index.storage_context.persist(persist_dir=self.index_dir)  # save the data
        vprint(f"Total vectors in FAISS index: {storage_context.vector_store._faiss_index.ntotal}", self.config)
        if document_metadata:
            vprint(f"Total documents in metadata: {len(document_metadata)}", self.config)
        return index

    def build_llama_documents(self,documents: list) -> list:
        llama_documents = []
        for document in documents:
            doc = Document(text=document['text'],
                           metadata={"filename": document['filename']})
            llama_documents.append(doc)
        return llama_documents


    def get_document_metadata(self):
        # Load document metadata if exists
        if os.path.exists(self.metadata_path ):
            with open(self.metadata_path , "rb") as f:
                document_metadata = pickle.load(f)
            vprint(f"Loaded metadata for {len(document_metadata)} documents.", self.config)
        else:
            document_metadata = {}
            vprint("No existing document metadata found. Starting fresh.", self.config)

        return document_metadata

    def save_index_and_metadata(self, faiss_index, document_metadata):
        """Save the FAISS index and document metadata."""
        faiss.write_index(faiss_index, self.faiss_index_path)
        with open(self.metadata_path , "wb") as f:
            pickle.dump(document_metadata, f)
        vprint(f"Saved FAISS index and metadata for {len(document_metadata)} documents.",self.config)

class Query:
    """
    query the AI Model

    """

    def __init__(self, config: dict, secrets: dict):
        try:
            self.token = secrets['api']['hf_key']
        except KeyError:
            raise Exception("API Key is required in secrets")
        self.config = config
        self.helper = Helper(config)
        self.index_dir = self.helper.index_dir
        self.embedding_model_name = self.helper.embedding_model_name
        # is this correct for all models, or do we need to elaborate for local models
        self.llm_model_name = config.get('model', {}).get('llamastack',{}).get('llm_model_name') or llm_model_name
        self.model_dir = config.get('model', {}).get('llamastack',{}).get('model_dir') or model_dir
        self.system_prompt = config.get('query', {}).get('system_prompt') or system_prompt
        self.query_engine = None

    def _init_llm_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, token=self.token)
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
            model_name=self.llm_model_name,
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
            tokenizer_name=self.llm_model_name,
            tokenizer_kwargs={
                "token": self.token,
                "cache_dir": self.model_dir,
            },
            stopping_ids=stopping_ids,
        )
        return model


    def get_vector_store_indices(self):
        storage_context = self.helper.get_storage_context()
        structs = storage_context.index_store.index_structs()
        return structs

    def huggingface_login(self):
        res = login(token=self.token)
        #Todo: test res, not just assume login was successful
        vprint("Logged in successfully to Huggingface!",self.config)
        return res

    def get_vector_store_index(self):
        storage_context = self.helper.get_storage_context()
        self.huggingface_login()
        embed_model = self.helper.initialize_embedding_model()
        # llm_model = self._init_llm_model()
        base_url = "http://hca-ollama-cpu:11434" if is_docker() else "http://localhost:11434"
        llm_model = Ollama(
            model="llama3.2",
            base_url=base_url,
            request_timeout=600.0)
        Settings.llm = llm_model
        # Settings.tokenizer = tokenizer
        Settings.embed_model = embed_model
        vector_store_index = load_index_from_storage(storage_context=storage_context)
        #index_id=storage_context.index_store.index_structs()[-1].index_id
        if vector_store_index:
            vprint(f"Number of nodes in index: {len(vector_store_index.ref_doc_info)}", self.config)
        return vector_store_index

    def report_status(self):
        vector_store = self.helper.get_vector_store()
        index = self.get_vector_store_index()
        print(f"Vectors in FAISS index: {vector_store._faiss_index.ntotal}")
        print(f"Documents in Vector Store Index: {len(index.ref_doc_info)}")

    def _configure_query_engine(self) -> RetrieverQueryEngine:
        index = self.get_vector_store_index()
        retriever = VectorIndexRetriever(
            index=index,
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
            "Nachfolgend die passensten Kontextinformationen.\n"
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

    def query_rag_llm(self, user_query):
        # Function to interact with the query engine and return a response
        if not self.query_engine:
            self.query_engine = self._configure_query_engine()
        with torch.no_grad():
            response = self.query_engine.query(user_query)
        torch.cuda.empty_cache()
        return str(response)

    def retrieve_docs(self, user_query) -> list[str]:
        """Retrieve relevant documents supporting the user query from the RAG query engine."""
        if not self.query_engine:
            self.query_engine = self._configure_query_engine()
        retrieved_nodes = self.query_engine.retriever.retrieve(user_query)
        result = []
        for node in retrieved_nodes:
            result.append({'score': node.score, 'metadata': node.metadata, 'content': node.text })
        result_sorted = sorted(result, key=lambda x: x['score'], reverse=True)
        return result_sorted
