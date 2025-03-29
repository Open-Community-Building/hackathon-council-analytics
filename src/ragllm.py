from importlib import import_module
from typing import Optional

#Defaults
framework = 'haystack'
#framework = 'llamastack'

class RagLlm:
    """
    Wrapper Class for LLM Frameworks
    provides index and query functionality
    Usage:
    from ragllm import RagLlm
    fw = RagLlm(config)
    """

    def __init__(self, config: dict, secrets: dict) -> None:
        self.config = config
        self.secrets = secrets
        _framework = config.get('model',{}).get('framework') or framework
        self.fwm = import_module(f"frameworks.{_framework}")

    def index(self, start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> list:
        """
        Wrapper function
        """
        #TODO: rename index to something else
        emb = self.fwm.Embedor(config=self.config, secrets=self.secrets)
        doc_count = emb.embed(start_idx=start_idx, end_idx=end_idx)
        return doc_count

    def update_index(self, start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> list:
        """
        Wrapper function
        """
        print("Not implemented yet")
        doc_count = 0
        return doc_count

    def retrieve_docs(self, user_query: str) -> list:
        query = self.fwm.Query(config=self.config, secrets=self.secrets)
        docs = query.retrieve_docs(user_query=user_query)
        return docs

    def run_query(self, user_query: str) -> str:
        """
        Wrapper function
        """
        query = self.fwm.Query(config=self.config, secrets=self.secrets)
        return query.query_rag_llm(user_query)


