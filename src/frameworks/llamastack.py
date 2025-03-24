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
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from preprocessor import Preprocessor
from huggingface_hub import login
from transformers import AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from typing import Optional
from utils import vprint
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
        self.embedding_model_name = config.get('embedding', {}).get('embedding_model_name') or embedding_model_name
        self.index_dir = config.get('embedding', {}).get('index_dir') or index_dir

    def initialize_embedding_model(self):
        """
        initialise the embedding model
        """
        embedding_model = HuggingFaceEmbedding(model_name=self.embedding_model_name)
        if embedding_model:
            vprint(f"Embedding model '{embedding_model.model_name}' initialized.", self.config)
        return embedding_model

    def get_faiss_index(self) -> faiss.IndexFlatL2:
        """
        get the existing faiss index from index_dir and return if exits
        else create one
        """
        faiss_index_path = os.path.join(self.index_dir, "faiss_index.idx")
        if os.path.exists(faiss_index_path):
            faiss_index = faiss.read_index(faiss_index_path)
            vprint(f"Loaded FAISS index with {faiss_index.ntotal} vectors.", self.config)
        else:
            embedding_model = self.initialize_embedding_model()
            test_embedding = embedding_model.get_text_embedding("test")
            embedding_dim = len(test_embedding)  # 384
            faiss_index = faiss.IndexFlatL2(embedding_dim)
            faiss.write_index(faiss_index, faiss_index_path)
            vprint("Created a new FAISS index.", self.config)
        return faiss_index

    def get_vector_store(self) -> FaissVectorStore:
        """
        get an existing vector_store
        else create one
        """
        vector_store_path = os.path.join(self.index_dir, "default__vector_store.json")
        if not os.path.exists(vector_store_path):
            faiss_index = self.get_faiss_index()
            faiss_store = FaissVectorStore(faiss_index=faiss_index)
            faiss_store.persist(vector_store_path)
        else:
            faiss_store = FaissVectorStore.from_persist_dir(self.index_dir)
        return faiss_store


    def get_storage_context(self):
        storage_context_path = os.path.join(self.index_dir, "docstore.json")
        if not os.path.exists(storage_context_path):
            faiss_store = self.get_vector_store()
            storage_context = StorageContext.from_defaults(vector_store=faiss_store)
            storage_context.persist(self.index_dir)
        else:
            faiss_store = FaissVectorStore.from_persist_dir(self.index_dir)
            storage_context = StorageContext.from_defaults(vector_store=faiss_store, persist_dir=self.index_dir)
        return storage_context

    def get_vector_store_indices(self):
        storage_context = self.get_storage_context()
        structs = storage_context.index_store.index_structs()

        return structs


    def get_vector_store_index(self):
        storage_context = self.get_storage_context()
        if storage_context.docstore.docs:
            vector_store_index = load_index_from_storage(storage_context, index_id=storage_context.index_store.index_structs()[-1].index_id)
            if vector_store_index:
                vprint(f"Number of nodes in index: {len(vector_store_index.ref_doc_info)}", self.config)
        else:
            vector_store_index = None
        return vector_store_index

    def report_status(self):
        vector_store =  self.get_vector_store()
        index = self.get_vector_store_index()
        print(f"Vectors in FAISS index: {vector_store._faiss_index.ntotal}")
        print(f"Documents in Vector Store Index: {len(index.ref_doc_info)}")

class Embedor:
    """
    class Emebedor embedes textfiles in vector store
    """
    def __init__(self, config: dict) -> None:
        """
        Constructs all the necessary attributes for the Embedor object.
        params: config: the configuration dict
        """
        self.config = config
        self.helper = Helper(config)
        self.pp = Preprocessor(config)
        self.fs = self.pp.fs
        self.index_dir = self.helper.index_dir




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
        storage_context = self.helper.get_storage_context()
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
        metadata_path = os.path.join(self.index_dir, "document_metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                document_metadata = pickle.load(f)
            vprint(f"Loaded metadata for {len(document_metadata)} documents.", self.config)
        else:
            document_metadata = {}
            vprint("No existing document metadata found. Starting fresh.", self.config)

        return document_metadata

    def save_index_and_metadata(self, faiss_index, document_metadata):
        """Save the FAISS index and document metadata."""
        faiss.write_index(faiss_index, os.path.join(self.index_dir, "faiss_index.idx"))
        metadata_path = os.path.join(self.index_dir, "document_metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(document_metadata, f)
        print(f"Saved FAISS index and metadata for {len(document_metadata)} documents.")

class Query:
    """
    query the AI Model

    """

    def __init__(self, config):
        try:
            self.token = config['api']['hf_key']
        except KeyError:
            raise Exception("API Key is required in config")
        self.helper = Helper(config)
        self.index_dir = self.helper.index_dir
        self.embedding_model_name = self.helper.embedding_model_name
        # is this correct for all models, or do we need to elaborate for local models
        self.llm_model_name = config.get('model', {}).get('llamastack',{}).get('llm_model_name') or llm_model_name
        self.model_dir = config.get('model', {}).get('llamastack',{}).get('model_dir') or model_dir
        self.system_prompt = config.get('query', {}).get('system_prompt') or system_prompt
        self.query_engine = self._configure_query_engine()

    def huggingface_login(self, token):
        if not token:
            raise ValueError("Please set your Hugging Face token in the HUGGINGFACE_TOKEN environment variable.")
        login(token=token)
        print("Logged in successfully!")

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
        #TODO: tokenizer is not used in consecutive code
        return tokenizer, model
                
    def _configure_query_engine(self) -> RetrieverQueryEngine:
        self.huggingface_login(self.token)
        embed_model = self.helper.initialize_embedding_model()
        tokenizer, llm_model = self._init_llm_model()
        Settings.llm = llm_model
        # Settings.tokenizer = tokenizer
        Settings.embed_model = embed_model
        index = self.helper.get_vector_store_index()
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
