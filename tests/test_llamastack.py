import sys

sys.path.append("../src")

from frameworks.llamastack import Embedor
def test_report_status(my_config):
    emb = Embedor(my_config)
    emb.report_status()
    assert type(emb).__name__ == "Embedor"
