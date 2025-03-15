# TODO: import code from olama branch into this skeleton
from preprocessor import Preprocessor
from tqdm import tqdm
from typing import Optional
from utils import vprint

class Embedor:

    def __init__(self,, config: dict) -> None:
        self.config = config
        self.pp = Preprocessor(config)
        self.fs = self.pp.fs

    def embed(self, start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> None:
        """
        This function is called from the admin interface
        """
        documents = self.fs.get_documents(start_idx, end_idx)

class Query:

    def query_rag_llm(self, user_query: str) -> str:
        """
        Query function to be called by webApp
        params:
        - user_query
        """
        response = self.query_engine.query(user_query)
        return str(response)

