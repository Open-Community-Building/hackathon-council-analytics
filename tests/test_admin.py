import sys

sys.path.append("../src")
import admin as admin


def test_show_config(my_config, my_secrets):
    admin.show_config(my_config, my_secrets)
    assert True


def test_download(my_config, my_secrets):
    admin.download(config=my_config, secrets=my_secrets, start_id=367896, end_id=368885)
    res = os.listdir(my_config['filestorage']['path'])
    assert len(res) > 0


def test_preprocess(my_config, my_secrets):
    admin.preprocess(config=my_config, secrets=my_secrets, start_id=367896, end_id=368885)
    assert False


def test_update_storage(my_config, my_secrets):
    admin.update_storage(config=my_config, secrets=my_secrets, amount=50)
    assert False


def test_main():
    assert True

def test_txtai_embed(my_config, my_secrets):
    my_config['model'] = {'framework': 'txtai'}
    my_config['embedding'] = {'txtai': {'index_dir': "/media/ncdata/__groupfolders/4/TestEmbeddings/txtai",
                                        'embedding_model_name': "sentence-transformers/nli-mpnet-base-v2"}}
    my_config['model'] = {'framework': 'txtai'}
    doc_count = admin.embed(config=my_config, secrets=my_secrets)
    assert doc_count == 245

def test_llama_embed(my_config, my_secrets):
    my_config['model'] = {'framework': 'llamastack'}
    my_config['embedding']['faiss']['index_dir'] = "/media/ncdata/__groupfolders/4/TestEmbeddings/faiss"
    doc_count = admin.embed(config=my_config, secrets=my_secrets)
    assert doc_count == 245

def test_haystack_embed(my_config, my_secrets):
    my_config['model'] = {'framework': 'haystack'}
    my_config['embedding']['qdrant']['index_dir'] = "/media/ncdata/__groupfolders/4/TestEmbeddings/qdrant"
    doc_count = admin.embed(config=my_config, secrets=my_secrets)
    assert doc_count == 245

def test_retriever(my_config, my_secrets):
    user_query = "Wer ist Prof. Dr. Dieter Hermann?"
    my_config['model'] = {'framework': 'haystack'}
    my_config['embedding']['qdrant']['index_dir'] = "/media/ncdata/__groupfolders/4/TestEmbeddings/qdrant"
    result = admin.retriever(config=my_config, secrets=my_secrets, user_query=user_query)
    assert type(result) == dict
    my_config['model'] = {'framework': 'llamastack'}
    my_config['embedding']['faiss']['index_dir'] = "/media/ncdata/__groupfolders/4/TestEmbeddings/faiss"
    result = admin.retriever(config=my_config, secrets=my_secrets, user_query=user_query)
    assert type(result) == dict
    my_config['embedding'] = {'txtai': {'index_dir': "/media/ncdata/__groupfolders/4/TestEmbeddings/txtai",
                                        'embedding_model_name': "sentence-transformers/nli-mpnet-base-v2"}}
    my_config['model'] = {'framework': 'txtai'}

    result = admin.retriever(config=my_config, secrets=my_secrets, user_query=user_query)
    assert type(result) == dict



