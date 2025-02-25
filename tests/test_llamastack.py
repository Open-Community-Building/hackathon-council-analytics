import sys

sys.path.append("../src")

from frameworks.llamastack import Embedor


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
    vector_store = emb.initFAISSIndex()
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
    assert type(faiss_index). __name__ == 'FaissVectorStore'
