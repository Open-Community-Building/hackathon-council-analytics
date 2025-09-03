import pytest
import os
import pprint
from importlib import import_module
import sys

sys.path.append("../src")

from storage.aws_s3 import FileStorage


def test_module(my_config, my_secrets):
    filestorage = 'aws_s3'
    fsm = import_module(f"src.storage.{filestorage}")
    assert type(fsm).__name__ == 'module'


def test_get_documents(my_config, my_secrets):
    # TODO: change to max_limit
    my_config['verbose'] = True
    fs = FileStorage(my_config, my_secrets)
    all_documents = fs.get_documents()
    assert type(all_documents) == list
    document = fs.get_documents(start_idx=200009)
    assert len(document) == 1
    documents = fs.get_documents(start_idx=200009, end_idx=200010)
    assert len(documents) == 2
    filelist  = ['200009.md','200010.md',]
    documents = fs.get_documents(filelist=filelist)
    assert len(documents) == 2
    exclude_filemanes = ['200010.md']
    documents = fs.get_documents(filelist=filelist, exclude_filenames=exclude_filemanes)
    assert len(documents) == 1

def test_class(my_config, my_secrets):
    filestorage = 'filesystem'
    fsm = import_module(f"src.storage.{filestorage}")
    fs = fsm.FileStorage(config=my_config, secrets=my_secrets)
    assert type(fs).__name__ == 'FileStorage'


def test_put_on_storage(my_config, my_secrets):
    fs = FileStorage(config=my_config, secrets=my_secrets)
    fs.put_on_storage('testfile.txt', 'Test content', content_type='text')
    result = fs.read_from_storage('testfile.txt')
    assert result == 'Test content'


def test_load_txt_files(my_config, my_secrets):
    fs = FileStorage(config=my_config, secrets=my_secrets)
    files = fs.get_txt_files()
    pprint.pp(files)
    assert type(files) == list




