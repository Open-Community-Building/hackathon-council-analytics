import sys

sys.path.append("../src")

from preprocessor import Preprocessor


def test_preprocessor(my_config, my_secrets):
    pp = Preprocessor(config=my_config, secrets=my_secrets)
    assert type(pp).__name__ == 'Preprocessor'


def test_preprocessor_config(my_config, my_secrets):
    pp = Preprocessor(config=my_config, secrets=my_secrets)
    pp.show_config()
    assert type(pp).__name__ == 'Preprocessor'


def test_filestorage(my_config, my_secrets):
    pp = Preprocessor(config=my_config, secrets=my_secrets)
    fs = pp.fs
    assert type(fs).__name__ == "FileStorage"


def test_request_pdf(my_config, my_secrets):
    pp = Preprocessor(config=my_config, secrets=my_secrets)
    result = pp.request_pdf(367896)  # 36777896 is the ID of the PDF in the database
    assert result


def test_get_pdf(my_config, my_secrets):
    pp = Preprocessor(config=my_config, secrets=my_secrets)
    result = pp.get_pdf(367896)
    assert result


def test_process_pdf(my_config, my_secrets):
    pp = Preprocessor(config=my_config, secrets=my_secrets)
    result = pp.process_pdf(367896)
    assert result


