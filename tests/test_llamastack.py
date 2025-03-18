import sys

sys.path.append("../src")

from frameworks.llamastack import Embedor
from frameworks.llamastack import Query


def test_report_status(my_config):
    emb = Embedor(my_config)
    emb.report_status()
    assert type(emb).__name__ == "Embedor"


def test_initialize_embedding_model(my_config):
    emb = Embedor(my_config)
    embedding_model = emb.initialize_embedding_model()
    assert type(embedding_model).__name__ == 'HuggingFaceEmbedding'


def test_init_faissindex(my_config):
    emb = Embedor(my_config)
    vector_store = emb.init_vector_store()
    assert type(vector_store).__name__ == 'FaissVectorStore'


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


def test_load_existing_index(my_config):
    emb = Embedor(my_config)
    faiss_index = emb.load_existing_index()
    assert type(faiss_index).__name__ == 'IndexFlatL2'


def test_get_document_metadata(my_config):
    emb = Embedor(my_config)
    document_metadata = emb.get_document_metadata()
    assert type(document_metadata) == dict
    print(document_metadata.keys())


def test_update_faiss_index(my_config):
    emb = Embedor(my_config)
    doc_count = emb.update_faiss_index(start_idx=368035, end_idx=368199)
    assert type(doc_count) == int


def test_query_rag_llm(my_config):
    query = Query(my_config)
    question = "Wie viele Unterlagen des Finanzausschusses sind vorhanden und welche sind das?"
    response = query.query_rag_llm(question)
    print(response)
    assert type(response) == str
