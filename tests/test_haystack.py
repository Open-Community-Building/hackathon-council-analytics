import sys

sys.path.append("../src")

from ragllm import RagLlm
from frameworks.haystack import Embedor
from frameworks.haystack import Query

def test_embedor_class(my_config):
    emb = Embedor(my_config)
    assert type(emb).__name__ == 'Embedor'


def test_init_document_store(my_config):
    emb = Embedor(my_config)
    dstore = emb._init_document_store()
    assert type(dstore).__name__ == 'QdrantDocumentStore'
    assert type(dstore.count_documents()) == int


def test_embed_and_index_documents(my_config):
    emb = Embedor(my_config)
    doc_count = emb.embed()
    print(doc_count)
    assert type(doc_count) == int

def test_query_class(my_config):
    query = Query(my_config)
    assert type(query).__name__ == 'Query'

def test_run_pipeline(my_config):
    query = Query(my_config)
    assert type(query.rag_pipeline).__name__ == 'Pipeline'


def test_query_rag_llm(my_config):
    query = Query(my_config)
    ans = query.query_rag_llm("Wie viele Unterlagen des Finanzausschusses sind vorhanden und welche sind das?")
    print(ans)
    assert type(ans) == str
