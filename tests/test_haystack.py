import sys

sys.path.append("../src")

from ragllm import RagLlm
from frameworks.haystack import Embedor
from frameworks.haystack import Query


#################
# Embedor Class #
#################

def test_embedor_class(my_config):
    emb = Embedor(my_config)
    assert type(emb).__name__ == 'Embedor'
    assert emb.document_store_path == '/media/ncdata/__groupfolders/4/qdrant/CouncilEmbeddings'


def test_build_local_document_store(my_config):
    emb = Embedor(my_config)
    document_store = emb.build_local_document_store()
    assert type(document_store).__name__ == 'QdrantDocumentStore'


def test_build_server_document_store(my_config):
    emb = Embedor(my_config)
    document_store = emb.build_server_document_store()
    assert type(document_store).__name__ == 'QdrantDocumentStore'



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


###############
# Query Class #
###############

def test_query_class(my_config):
    query = Query(my_config)
    assert type(query).__name__ == 'Query'


def test_get_local_document_store(my_config):
    query = Query(my_config)
    document_store = query.get_local_document_store()
    print(f"\n\nDocuments in Store {document_store.count_documents()}\n")
    assert type(document_store).__name__ == 'QdrantDocumentStore'


def test_run_pipeline(my_config):
    query = Query(my_config)
    assert type(query.rag_pipeline).__name__ == 'Pipeline'


def test_query_rag_llm(my_config):
    query = Query(my_config)
    question = 'Wer ist Prof. Dr. Dieter Hermann?'
    # question = '"Wie viele Unterlagen des Finanzausschusses sind vorhanden und welche sind das?"'
    ans = query.query_rag_llm(question)
    print(ans)
    assert type(ans) == str


def test_run_retriever_pipeline(my_config):
    query = Query(my_config)
    user_query = "Wie viele Unterlagen des Finanzausschusses sind vorhanden und welche sind das?"
    retriever_documents = query.run_retriever_pipeline(user_query)
    print(retriever_documents)
    assert len(retriever_documents) > 0



def test_get_server_document_store(my_config):
    query = Query(my_config)
    dstore = query.get_server_document_store()
    print(dstore.count_documents())
    assert type(dstore).__name__ == 'QdrantDocumentStore'
