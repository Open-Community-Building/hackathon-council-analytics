import pytest
import os
import tomllib

@pytest.fixture
def my_config(my_configfile):
    configfile = my_configfile
    with open(configfile, "rb") as f:
        config = tomllib.load(f)
    config['verbose'] = 1
    return config

@pytest.fixture
def my_configfile():
    configfile = os.path.expanduser(os.path.join('~', '.config', 'hca', 'config.toml'))
    return configfile

@pytest.fixture
def my_secrets():
    secretsfile = os.path.expanduser(os.path.join('~', '.config', 'hca', 'secrets.toml'))
    with open(secretsfile, "rb") as f:
        secrets = tomllib.load(f)
    return secrets

@pytest.fixture
def my_sample_config(my_sample_configfile):
    with open(my_sample_configfile, "rb") as f:
        config = tomllib.load(f)
    return config


@pytest.fixture
def my_sample_configfile():
    configfile = "../src/config_sample.toml"
    return os.path.abspath(configfile)