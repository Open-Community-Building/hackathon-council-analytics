import sys

sys.path.append("../src")
import admin as admin


def test_show_config(my_config, my_secrets):
    admin.show_config(config=my_config, secrets=my_secrets)
    assert True


def test_download(my_config, my_secrets):
    admin.download(config=my_config, secrets=my_secrets, start_id=367896, end_id=368885)
    res = os.listdir(my_config['filestorage']['path'])
    assert len(res) > 0


def test_preprocess(my_config, my_secrets):
    admin.preprocess(config=my_config, secrets=my_secrets, start_id=367896, end_id=368885)
    assert False


def test_update_storage(my_config, my_secrets):
    admin.update_storage(config=my_config, secrets=my_secrets, 50)
    assert False


def test_main():
    assert True
