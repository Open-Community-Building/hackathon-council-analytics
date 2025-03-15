import sys

sys.path.append("../src")

from ragllm import RagLlm

def test_embedor_class(my_config):
    my_config['model']['framework'] = 'haystack'
    rag_llm = RagLlm(my_config)
    assert type(rag_llm).__name__ == 'RagLlm'
    emb = rag_llm.emb
    assert type(emb).__name__ == 'Embedor'


def test_init_document_store(my_config):
    my_config['model']['framework'] = 'haystack'
    rag_llm = RagLlm(my_config)
    emb = rag_llm.emb
    dstore = emb._init_document_store()
    assert type(dstore).__name__ == 'QdrantDocumentStore'
    assert type(dstore.count_documents()) == int


def test_embed_and_index_documents(my_config):
    my_config['model']['framework'] = 'haystack'
    rag_llm = RagLlm(my_config)
    emb = rag_llm.emb
    doc_count = emb.embed(367896,367881)
    assert type(doc_count) == int


def test_run_pipeline():
    assert False


def test_query_rag_llm():
    assert False
