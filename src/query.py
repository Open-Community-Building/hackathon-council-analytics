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
from llama_index.llms.ollama import Ollama


class RAG_LLM:

    def __init__(self):
        self.embed_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.llm_name = "llama3"  # Or any Ollama model name
        index_dir = "CouncilEmbeddings/"
        # index_dir = "../preprocessing/vectorstore_index"

        self.embed_model = self.init_embedding_model(self.embed_name)
        self.llm_model = self.init_llm_model()

        Settings.llm = self.llm_model
        # Settings.tokenizer = tokenizer
        Settings.embed_model = self.embed_model

        self.index = self.load_index_storage(index_dir)
        self.query_engine = self.configure_query_engine(self.index)
        # display_prompt_dict(prompts_dict)


    def init_llm_model(self):
        system_prompt = """Du bist ein intelligentes System, das deutsche Dokumente durchsucht und auf Basis der enthaltenen Informationen präzise Antworten auf gestellte Fragen gibt. Wenn du eine Antwort formulierst, gib die Antwort in klaren und präzisen Sätzen an und nenne dabei mindestens eine oder mehrere relevante Quellen im Format: (Quelle: Dokumentname, Abschnitt/Seite, Filename des TXT)."""
        
        llm = Ollama(
            model=self.llm_name,
            base_url="http://localhost:11434",
            system_prompt=system_prompt,
            temperature=0.3,
            request_timeout=300,
        )

        print(f"Ollama model {self.llm_name} initialized.")
        return llm


    def init_embedding_model(self, embed_name):
        from llama_index.embeddings.ollama import OllamaEmbedding
        embedding_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        print(f"Ollama embedding model initialized.")
        return embedding_model


    def load_index_storage(self, index_dir):

        faiss_store = FaissVectorStore.from_persist_dir(index_dir)
        storage_context = StorageContext.from_defaults(vector_store=faiss_store, persist_dir=index_dir)
        # storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
        print(f"Number of vectors stored: {faiss_store._faiss_index.ntotal}")
        print(f"Number of nodes in index: {len(index.ref_doc_info)}")

        return index


    def configure_query_engine(self, index) -> RetrieverQueryEngine:
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=2,
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
            "Du bist ein intelligentes System, das diese deutschen Kontextinformationen durchsucht und auf Basis der enthaltenen Informationen präzise Antworten auf gestellte Fragen gibt. Wenn du eine Antwort in klaren und präzisen Sätzen formulierst, nenne dabei mindestens eine oder mehrere relevante Quellen auf die entsprechenden Textstellen des Kontexts im Format: (Quelle: Dokumentname, Abschnitt/Seite).\n"
            "Query: {query_str}\n"
            "Antwort: "
        )
        prompt_template = PromptTemplate(summary_prompt)
        query_engine.update_prompts(
            {"response_synthesizer:summary_template": prompt_template}
        )

        return query_engine


    def display_prompt_dict(self, prompts_dict):
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


if __name__ == "__main__":

    rag_llm = RAG_LLM()

    query = "Wie viele Unterlagen des Finanzausschusses sind vorhanden und welche sind das?"
    response = rag_llm.query_rag_llm(query)

    print("\n=================")
    print(query)
    print("---------------")
    print(response)
