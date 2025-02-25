import sys

sys.path.append("../src")
import admin as admin


def test_show_config(my_config):
    admin.show_config(config=my_config)
    assert True


def test_download(my_config):
    admin.download(config=my_config, start_id=367896, end_id=368885)
    res = os.listdir(my_config['filestorage']['path'])
    assert len(res) > 0

def test_preprocess(my_config):
    admin.preprocess(config=my_config, start_id=367896, end_id=368885)
    assert False

def test_update_storage(my_config):
    admin.update_storage(my_config,50)
    assert False


