import sys

sys.path.append("../src")

from frameworks.txtai import Embedor
from frameworks.txtai import Query
from frameworks.txtai import Helper

import os

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"


#######################
### HELPER
########################

def test_helper(my_config):
    helper = Helper(config=my_config)
    assert type(helper).__name__ == 'Helper'


def test_get_embeddings(my_config):
    helper = Helper(config=my_config)
    embeddings = helper.get_embeddings()
    assert embeddings.count() == 6


#########################
### EMBEDOR
#########################

def test_embedor(my_config, my_secrets):
    emb = Embedor(config=my_config, secrets=my_secrets)
    assert type(emb).__name__ == 'Embedor'


def test_embeddings(my_config, my_secrets):
    emb = Embedor(config=my_config, secrets=my_secrets)
    data = [
        "US tops 5 million confirmed virus cases",
        "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
        "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
        "The National Park Service warns against sacrificing slower friends in a bear attack",
        "Maine man wins $1M from $25 lottery ticket",
        "Make huge profits without work, earn up to $100,000 a day"
    ]
    # Create an embeddings
    embeddings = emb.embed_data(data)

    print("%-20s %s" % ("Query", "Best Match"))
    print("-" * 50)

    # Run an embeddings search for each query
    for query in (
            "feel good story", "climate change", "public health story", "war", "wildlife", "asia", "lucky",
            "dishonest junk"):
        # Extract uid of first result
        # search result format: (uid, score)
        uid = embeddings.search(query, 1)[0][0]

        # Print text
        print("%-20s %s" % (query, data[uid]))


def test_embed(my_config, my_secrets):
    emb = Embedor(config=my_config, secrets=my_secrets)
    embeddings = emb.embed()
    assert embeddings.count() == 245


#######################
### QUERY           ###
#######################

def test_query(my_config, my_secrets):
    query = Query(config=my_config, secrets=my_secrets)
    assert type(query).__name__ == 'Query'


def test_query_rag_llm(my_config, my_secrets):
    query = Query(config=my_config, secrets=my_secrets)
    user_query = "Wer ist Prof. Dr. Dieter Hermann?"
    ans = query.query_rag_llm(user_query)
    print(ans)
    assert type(ans) == dict


def test_retrieve_docs(my_config, my_secrets):
    query = Query(config=my_config, secrets=my_secrets)
    user_query = "Wer ist Prof. Dr. Dieter Hermann?"
    retriever_documents = query.retrieve_docs(user_query)
    print(retriever_documents)
    assert len(retriever_documents) > 0


def test_report_status(my_config, my_secrets):
    query = Query(config=my_config, secrets=my_secrets)
    status = query.report_status()
    print(status)
    assert status[0]['count(*)'] == 245
