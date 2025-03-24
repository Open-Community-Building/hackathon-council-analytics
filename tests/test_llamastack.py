import sys

sys.path.append("../src")

from frameworks.llamastack import Embedor
from frameworks.llamastack import Query
from frameworks.llamastack import Helper


##############
# Helper Class
##############
def test_helper_class(my_config):
    helper = Helper(my_config)
    assert type(helper).__name__ == 'Helper'


def test_initialize_embedding_model(my_config):
    helper = Helper(my_config)
    embedding_model = helper.initialize_embedding_model()
    assert type(embedding_model).__name__ == 'HuggingFaceEmbedding'


def test_get_faiss_index(my_config):
    my_config['verbose'] = 1
    helper = Helper(my_config)
    faiss_index = helper.get_faiss_index()
    assert type(faiss_index).__name__ == 'IndexFlatL2'


def test_get_vector_store(my_config):
    helper = Helper(my_config)
    vector_store = helper.get_vector_store()
    assert type(vector_store).__name__ == 'FaissVectorStore'


def test_get_storage_context(my_config):
    helper = Helper(my_config)
    storage_context = helper.get_storage_context()
    assert type(storage_context).__name__ == 'StorageContext'

def test_get_vector_store_indices(my_config):
    helper = Helper(my_config)
    indices = helper.get_vector_store_indices()
    assert False
def test_get_vector_store_index(my_config):
    my_config['verbose'] = 1
    helper = Helper(my_config)
    index = helper.get_vector_store_index()
    assert type(index).__name__ == 'Unknown'


def test_report_status(my_config):
    helper = Helper(my_config)
    helper.report_status()
    assert type(helper).__name__ == "Helper"


###############
# Class Embedor
###############

def test_build_llama_documents(my_config):
    emb = Embedor(my_config)
    documents = emb.fs.get_documents(start_idx=368035, end_idx=368052)
    llama_documents = emb.build_llama_documents(documents)
    assert type(llama_documents) == list


def test_embed(my_config):
    emb = Embedor(my_config)
    doc_count = emb.embed(start_idx=368035, end_idx=368187)
    print(f"doc count: {doc_count}")
    assert type(doc_count) == int


def test_get_document_metadata(my_config):
    emb = Embedor(my_config)
    document_metadata = emb.get_document_metadata()
    assert type(document_metadata) == dict
    print(document_metadata.keys())


def test_update_faiss_index(my_config):
    emb = Embedor(my_config)
    doc_count = emb.update_faiss_index(start_idx=368035, end_idx=368199)
    assert type(doc_count) == int


##############
# class Query
#############

def test__init_llm_model(my_config):
    query = Query(my_config)
    tokenizer, llm_model = query._init_llm_model()
    assert type(llm_model).__name__ == 'HuggingFaceLLM'


def test__configure_query_engine(my_config):
    query = Query(my_config)
    query_engine = query._configure_query_engine()
    assert type(query_engine).__name__ == 'RetrieverQueryEngine'


def test_query_rag_llm(my_config):
    query = Query(my_config)
    question = "Wie viele Unterlagen des Finanzausschusses sind vorhanden und welche sind das?"
    response = query.query_rag_llm(question)
    print(response)
    assert type(response) == str

