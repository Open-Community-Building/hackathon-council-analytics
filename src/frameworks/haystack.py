from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack import Pipeline
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack import Document
from preprocessor import Preprocessor
from typing import Optional
from utils import vprint

"""
refactored from https://github.com/medulka/LLMs/blob/main/RAG_haystack_hanka_mistral.ipynb
"""

#Defaults
llm_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# llm_model_name = "utter-project/EuroLLM-1.7B-Instruct"
# llm_model_name = "utter-project/EuroLLM-9B-Instruct"
# llm_model_name = "openGPT-X/Teuken-7B-instruct-research-v0.4"
# llm_model_name = "Aleph-Alpha/Pharia-1-LLM-7B-control-aligned-hf"
# llm_model_name = "BSC-LT/salamandra-7b-instruct"
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
prompt_template = """
You are the council of the town Heidelberg in Germany. Do following steps:

First, response to the Question. Find the Answer in the Documents stored in the document_store and returned it. Please provide your Answer in German.

Next, return the document where you find the Answer. Return the positive hits only.

Documents:
{% for doc in documents %}
    Document {{ loop.index }}:
    Document name: {{ doc.meta['name'] }}
    {{ doc.content }}
{% endfor %}

Question: {{ query }}

Answer:
"""


class Embedor:

    #TODO: Floating Window

    def __init__(self, config: dict, secrets: dict) -> None:
        self.config = config
        self.pp = Preprocessor(config,secrets)
        self.fs = self.pp.fs
        self.qdrant_url = secrets['api']['qdrant_url']
        self.qdrant_api_key = secrets['api']['qdrant_api_key']
        self.hf_token = secrets['api']['hf_key']
        self.document_store_path = config.get('embedding',{}).get('qdrant',{}).get('index_dir')
        self.document_store = self.build_local_document_store()
        #self.document_store = self.build_server_document_store()
        self.embedding_model_name = config.get('embedding', {}).get('qdrant', {}).get('embedding_model_name')

    def build_server_document_store(self) -> QdrantDocumentStore:
        return QdrantDocumentStore(
            url=self.qdrant_url,
            api_key=Secret.from_token(self.qdrant_api_key),
            index="Document",
            recreate_index=True,
            return_embedding=True,
            wait_result_from_api=True,
            use_sparse_embeddings=True,
            embedding_dim=384,
        )

    def build_local_document_store(self):
        return QdrantDocumentStore(
            path=self.document_store_path,
            index="Document",
            recreate_index=True,
            return_embedding=True,
            use_sparse_embeddings=True,
            embedding_dim=384,
        )


    def embed(self,  start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> None:
        """
        embedding function
        to be called by admin.py
        params:
        - start_idx
        - end_idx
        #ToDo: preprocessed documents, update    
        """
        documents = self.fs.get_documents(start_idx=start_idx,end_idx=end_idx)
        count = self.embed_and_index_documents(documents)
        return count

    def embed_and_index_documents(self, documents: list):
        """
        embedder function
        params:
        - docucuments
        """
        haystack_documents = []
        for doc in documents:
            haystack_documents.append(Document(content=doc['text'],
                                               meta={'name': doc['filename']}))
        document_embedder = SentenceTransformersDocumentEmbedder(
            model=self.embedding_model_name,
            token=Secret.from_token(self.hf_token),
        )
        document_store = self.document_store
        document_embedder.warm_up()
        document_with_embeddings = document_embedder.run(haystack_documents)
        document_store.write_documents(document_with_embeddings.get("documents"), policy=DuplicatePolicy.OVERWRITE)
        vprint(document_store.count_documents(), self.config)
        return document_store.count_documents()


class Query:
    """
    query the Model
    """

    def __init__(self, config: dict, secrets: dict) -> None:
        self.config = config
        self.prompt_template = config.get('model',{}).get('haystack',{}).get('prompt_template') or prompt_template
        self.hf_token = secrets['api']['hf_key']
        self.qdrant_url = secrets['api']['qdrant_url']
        self.qdrant_api_key = secrets['api']['qdrant_api_key']
        self.llm_model_name = config.get('model',{}).get('haystack',{}).get('llm_model_name') or llm_model_name
        self.document_store_path = config.get('embedding', {}).get('qdrant', {}).get('index_dir')
        self.document_store = self.get_local_document_store()
        self.embedding_model_name = config.get('embedding', {}).get('qdrant', {}).get('embedding_model_name')
        self.rag_pipeline = self.build_pipeline()

    def get_local_document_store(self):
        return QdrantDocumentStore(
            path=self.document_store_path,
            index="Document",
            recreate_index=False,
            return_embedding=True,
            use_sparse_embeddings=True,
            embedding_dim=384,
        )

    def get_server_document_store(self):
        return QdrantDocumentStore(
            url=self.qdrant_url,
            api_key=Secret.from_token(self.qdrant_api_key),
            index="Document",
            recreate_index=False,
            return_embedding=True,
            wait_result_from_api=True,
            use_sparse_embeddings=True,
            embedding_dim=384,
        )

    def init_retriever(self):
        return QdrantEmbeddingRetriever(document_store=self.document_store, top_k=3)

    def init_text_embedder(self):
        return SentenceTransformersTextEmbedder(
            model=self.embedding_model_name,
            token=Secret.from_token(self.hf_token),
        )

    def init_generator(self):
        return HuggingFaceAPIGenerator(
            api_type="serverless_inference_api",
            api_params={"model": self.llm_model_name},
            token=Secret.from_token(self.hf_token),
            generation_kwargs={"max_new_tokens": 2000}
        )

    def init_prompt_builder(self):
        return PromptBuilder(template=self.prompt_template)

    def build_pipeline(self):
        rag_pipeline = Pipeline()

        rag_pipeline.add_component('text_embedder', self.init_text_embedder())
        rag_pipeline.add_component('retriever', self.init_retriever())
        rag_pipeline.add_component('prompt_builder', self.init_prompt_builder())
        rag_pipeline.add_component('generator', self.init_generator())

        rag_pipeline.connect('text_embedder.embedding', 'retriever.query_embedding')
        rag_pipeline.connect('retriever.documents', 'prompt_builder.documents')
        rag_pipeline.connect('prompt_builder', 'generator')
        return rag_pipeline

    def retrieve_docs(self, user_query: str):
        retriever_pipeline = Pipeline()
        retriever_pipeline.add_component("text_embedder", self.init_text_embedder())
        retriever_pipeline.add_component("retriever", self.init_retriever())
        retriever_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        retriever_result = retriever_pipeline.run({"text_embedder": {"text": user_query}})
        result = []
        for document in retriever_result['retriever']['documents']:
            result.append({'score': document.score, 'metadata': document.meta, 'content': document.content})
        result_sorted = sorted(result, key=lambda x: x['score'], reverse=True)
        return result_sorted

    def query_rag_llm(self, user_query: str) -> str:
        """
        Query function to be called by webApp
        params:
        - user_query
        """
        ans = self.rag_pipeline.run(
            {"text_embedder": {"text": user_query}}
        )
        return ans['generator']['replies'][0].strip()
