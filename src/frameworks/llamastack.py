import faiss
import pickle
import torch
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Document
from llama_index.core import Settings, load_index_from_storage
from preprocessor import Preprocessor
from huggingface_hub import login
from tqdm import tqdm
from typing import Optional
from utils import vprint
import os

"""
This Module provides functions to work on the vector store

Classes:
- 'Embedor': provide embedding functions
- 'Query':   provide query function

Functions:

Example Usage:


"""
# Defaults
index_dir = "CouncilEmbeddings"
ll_name    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#TODO: refactor this to embedding_model_name
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model_dir  = "model"
system_prompt = """Du bist ein intelligentes System, das deutsche Dokumente durchsucht und auf Basis der enthaltenen Informationen präzise Antworten auf gestellte Fragen gibt. Wenn du eine Antwort formulierst, gib die Antwort in klaren und präzisen Sätzen an und nenne dabei mindestens eine oder mehrere relevante Quellen im Format: (Quelle: Dokumentname, Abschnitt/Seite, Filename des TXT)."""


# TODO: this is named index_dir in query.py
# storage_dir = "vectorstore_index"

class Embedor:
    """
    embedding class
    """

    def __init__(self, config: dict) -> None:
        """
        Constructs all the necessary attributes for the Embedor object.
        params: config: the configuration dict
        """
        self.config = config
        self.pp = Preprocessor(config)
        self.fs = self.pp.fs
        self.verbose = config.get('verbose')
        self.model_name = config.get('model', {}).get('model_name')
        self.index_dir = config.get('embedding', {}).get('index_dir') or index_dir
        self.embedding_model = self.initialize_embedding_model()

    def report_status(self):
        faiss_index = self.load_existing_index()
        vector_store =  self.init_vector_store(faiss_index=faiss_index)
        print(f"Vectors in FAISS index: {vector_store._faiss_index.ntotal}")
        # print(f"Documents in Vector Store Index: {len(index.ref_doc_info)}")

    def initialize_embedding_model(self):
        embedding_model = HuggingFaceEmbedding(model_name=self.model_name)
        vprint(f"Embedding model '{self.model_name}' initialized.", self.config)
        return embedding_model


    def init_vector_store(self, faiss_index=None):
        """Initialize or load the FaissVectorStore."""
        if faiss_index is None:
            test_embedding = self.embedding_model.get_text_embedding("test")
            embedding_dim = len(test_embedding)  # 384
            faiss_index = faiss.IndexFlatL2(embedding_dim)
            vprint("Created a new FAISS index.", self.config)
        faiss_store = FaissVectorStore(faiss_index=faiss_index)  # Initialize vector store
        return faiss_store

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
        seve.
        """
        # Load existing FAISS index and metadata
        faiss_index = self.load_existing_index()
        # Initialize vector store (reuse the existing FAISS index if available)
        vector_store = self.init_vector_store(faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # Configure text splitter for chunking
        Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

        llama_documents = self.build_llama_documents(documents)
        # Embed and add new documents to the FAISS index
        index = VectorStoreIndex.from_documents(
            llama_documents,
            storage_context=storage_context,
            embed_model=self.embedding_model,
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
        vprint(f"Total vectors in FAISS index: {vector_store._faiss_index.ntotal}", self.config)
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

    def load_existing_index(self):
        """Load the existing FAISS index and document metadata."""
        faiss_index_path = os.path.join(self.index_dir, "faiss_index.idx")
        if os.path.exists(faiss_index_path):
            faiss_index = faiss.read_index(faiss_index_path)
            vprint(f"Loaded FAISS index with {faiss_index.ntotal} vectors.", self.config)
        else:
            faiss_index = None
            vprint("No existing FAISS index found. Creating a new one.", self.config)
        return faiss_index

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
    guery the AI Model

    """

    def __init__(self, config):
        try:
            self.token = config['api']['hf_key']
        except KeyError:
            raise Exception("API Key is requirerd in config")
        # is this correct for all models, or do we need to elaborate for local models
        self.llm_name = config.get('model', {}).get('llm_name') or llm_name
        self.model_name = config.get('model', {}).get('model_name') or model_name
        self.model_dir = config.get('model', {}).get('model_dir') or model_dir
        self.index_dir = config.get('source', {}).get('folderEmbeddings') or index_dir
        self.system_prompt = config.get('query', {}).get('system_prompt') or system_prompt
        self.huggingface_login(self.token)
        self.query_engine = self._configure_query_engine()

    def huggingface_login(self, token):
        if not token:
            raise ValueError("Please set your Hugging Face token in the HUGGINGFACE_TOKEN environment variable.")
        login(token=token)
        print("Logged in successfully!")

    def load_index_storage(self):
        """
        load index
        """
        #ToDo: could we use load_existing_index from Embedor Class and refactor in a Helper Class?
        faiss_store = FaissVectorStore.from_persist_dir(self.index_dir)
        storage_context = StorageContext.from_defaults(vector_store=faiss_store, persist_dir=self.index_dir)
        # storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
        print(f"Number of vectors stored: {faiss_store._faiss_index.ntotal}")
        print(f"Number of nodes in index: {len(index.ref_doc_info)}")
        return index

    def _configure_query_engine(self) -> RetrieverQueryEngine:
        retriever = VectorIndexRetriever(
            index=self.load_index_storage(),
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