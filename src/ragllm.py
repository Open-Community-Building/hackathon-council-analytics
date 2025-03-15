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

    def __init__(self, config: dict) -> None:
        self.config = config
        _framework = config.get('model',{}).get('framework') or framework
        _fwm = import_module(f"frameworks.{_framework}")
        self.emb = _fwm.Embedor(config=config)
        self.query = _fwm.Query()

    def index(self, start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> list:
        """
        Wrapper function
        """
        #TODO: rename index to something else
        doc_count = self.emb.embed(start_idx=start_idx, end_idx=end_idx)
        return doc_count

    def query(self, user_query: str) -> str:
        """
        Wrapper function
        """
        return self.query.query_rag_llm(user_query)


