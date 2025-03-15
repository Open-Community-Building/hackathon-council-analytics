from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Pipeline
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from preprocessor import Preprocessor
from typing import Optional

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

    def __init__(self, config: dict) -> None:
        self.config = config
        self.pp = Preprocessor(config)
        self.fs = self.pp.fs
        self.verbose = config.get('verbose')
        self.qdrant_url = config['api']['qdrant_url']
        self.qdrant_api_key = config['api']['qdrant_api_key']
        self.hf_token = config['api']['hf_key']
        self.document_store = self._init_document_store()

    def _init_document_store(self) -> QdrantDocumentStore:
        return QdrantDocumentStore(
            url=self.qdrant_url,
            api_key=Secret.from_token(self.hf_token),
            index="Document",
            recreate_index=True,
            return_embedding=True,
            wait_result_from_api=True,
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
        document_embedder = SentenceTransformersDocumentEmbedder(
            model=embedding_model_name,
            token=Secret.from_token(self.hf_token),
        )
        document_embedder.warm_up()
        document_with_embeddings = document_embedder.run(documents)
        self.document_store.write_documents(document_with_embeddings.get("documents"), policy=DuplicatePolicy.OVERWRITE)
        vprint(self.document_store.count_documents(), config)
        return self.document_store.count_documents()


class Query:
    """
    query the Model
    """
    def run_pipeline(self):
        pipeline_text_embedder = SentenceTransformersTextEmbedder(
            model=embedding_model_name,
            token=Secret.from_token(self.hf_token),
        )
        pipeline_retriever = QdrantEmbeddingRetriever(document_store=self.document_store)

        pipeline_prompt_builder = PromptBuilder(template=prompt_template)

        pipeline_generator = HuggingFaceAPIGenerator(api_type="serverless_inference_api",
                                                     api_params={"model": llm_model_name},
                                                     token=Secret.from_token(HF_TOKEN),
                                                     generation_kwargs={"max_new_tokens": 2000}
                                                     )

        rag_pipeline = Pipeline()

        rag_pipeline.add_component('text_embedder', pipeline_text_embedder)
        rag_pipeline.add_component('retriever', pipeline_retriever)
        rag_pipeline.add_component('prompt_builder', pipeline_prompt_builder)
        rag_pipeline.add_component('generator', pipeline_generator)

        rag_pipeline.connect('text_embedder.embedding', 'retriever.query_embedding')
        rag_pipeline.connect('retriever.documents', 'prompt_builder.documents')
        rag_pipeline.connect('prompt_builder', 'generator')

    def query_rag_llm(self, user_query: str) -> str:
        """
        Query function to be called by webApp
        params:
        - user_query
        """
        ans = rag_pipeline.run(
            {"text_embedder": {"text": user_query}}
        )
        return ans['generator']['replies'][0].strip()
