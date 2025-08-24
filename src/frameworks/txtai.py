from txtai import Embeddings, RAG
import torch
from preprocessor import Preprocessor
from typing import Optional
from utils import vprint

"""
https://github.com/neuml/txtai/blob/master/examples/01_Introducing_txtai.ipynb
"""

#Defaults
llm_model_name = "google/flan-t5-large"
embedding_model_name = "sentence-transformers/nli-mpnet-base-v2"
index_dir = "/media/ncdata/__groupfolders/4/TestEmbeddings/index"

class Helper:

    def __init__(self, config: dict):
        self.config = config
        self.index_dir = config.get('embedding', {}).get('txtai', {}).get('index_dir') or index_dir
        self.embedding_model_name = config.get('embedding', {}).get('txtai', {}).get(
            'embedding_model_name') or embedding_model_name

    def get_embeddings(self):
        embeddings = Embeddings(hybrid=True, path=self.embedding_model_name, content=True, objects=True)
        if embeddings.exists(self.index_dir):
            embeddings.load(self.index_dir)
        return embeddings


class Embedor:


    def __init__(self, config: dict, secrets: dict) -> None:
        self.config = config
        self.pp = Preprocessor(config,secrets)
        self.fs = self.pp.fs
        helper = Helper(config)
        #self.embeddings = Embeddings(path=self.embedding_model_name)
        self.embeddings = helper.get_embeddings()
        self.index_dir = helper.index_dir

    def embed_data(self,data):
        """
        Create an index for the list of text
        """
        self.embeddings.index(data)
        return self.embeddings

    def save_index(self):
        self.embeddings.save(self.index_dir)


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
        self.embeddings.index([{"text": doc['text'], "filename": doc['filename'], "length": len(doc['text'])} for doc in documents])
        self.save_index()
        return self.embeddings.count()


class Query:
    """
    query the Model
    txtai is an all-in-one embeddings database. It is the only vector database that also supports sparse indexes, graph networks and relational databases with inline SQL support. In addition to this, txtai has support for LLM orchestration.

The RAG pipeline is txtai's spin on retrieval augmented generation (RAG). This pipeline extracts knowledge from content by joining a prompt, context data store and generative model together.
    """

    def __init__(self, config: dict, secrets: dict) -> None:
        self.config = config
        self.helper = Helper(config)
        self.embeddings = self.helper.get_embeddings()


    def prompt(self, question):
        return [{"query": question,
            "question": f"""
Answer the following question using the context below.
Question: {question}
Context:
"""
}]    

    def query_rag_llm(self, user_query: str) -> str:
        """
        Query function to be called by webApp
        params:
        - user_query
        """
        rag = RAG(self.embeddings,
                  "google/flan-t5-large",
                  torch_dtype=torch.bfloat16,
                  output="reference")
        ans = rag(self.prompt(user_query))[0]
        return ans

    def retrieve_docs(self, user_query) -> list[str]:
        retrieved_nodes = self.embeddings.search(f"select text, filename, score from txtai where similar('{user_query}') and score >= 0.15")
        result = []
        for node in retrieved_nodes:
            result.append({'score': node['score'], 'metadata': {'filename': node['filename']}, 'content': node['text']})
        return result

    def report_status(self):
        res = self.embeddings.search("select count(*), min(length), max(length), sum(length) from txtai")
        return res

