import torch
import json
import os
import pandas as pd
import numpy as np
from time import sleep
from matplotlib import pyplot as plt
import time

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case.llm_test_case import LLMTestCase
from deepeval.models import OllamaModel, OllamaEmbeddingModel

from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

from ragllm import RagLlm
from query import read_config, query, retrieve


DEFAULT_CONFIGFILE = os.path.expanduser(os.path.join('~','.config','hca','config.toml'))
DEFAULT_SECRETSFILE = os.path.expanduser(os.path.join('~','.config','hca','secrets.toml'))


class RagLlmTester:

    def __init__(self, config, secrets):
        self.config = config
        self.secrets = secrets

        self.eval_llm = OllamaModel(
            model="mistral:7b",
            base_url="http://localhost:11434",
        )
        self.metric_faithful = FaithfulnessMetric(model=self.eval_llm)
        self.scores = []


    def run_tests(self):
        _, _, questions, _ = self.load_ground_truth(os.path.join(self.config["documents"]["filestorage"]["path"], 'questions.csv'))

        for question in questions:
            self.run_test(question)

        return np.mean(self.scores), np.std(self.scores)


    def run_test(self, question):
        rag_llm = RagLlm(self.config, self.secrets)
        retrieved = rag_llm.retrieve_docs(question)
        generated_answer = rag_llm.run_query(question)

        print(f"Question: {question}")
        print(f"Retrieved text: {retrieved}")
        print(f"Answer: {generated_answer}")

        retrieved = self.to_strings_for_deepeval(retrieved)

        # Antwort absichern (DeepEval erwartet str)
        generated_answer = "" if generated_answer is None else str(generated_answer)

        test_case = LLMTestCase(
            input=question,
            actual_output=generated_answer,
            retrieval_context=retrieved
        )

        self.metric_faithful.measure(test_case)
        print(self.metric_faithful.score)
        print(self.metric_faithful.reason)

        self.scores.append(self.metric_faithful.score)


    def load_ground_truth(self, csv_path):
        """Load ground truth CSV file containing questions and related answers and the document to find it in"""
        print(f"Pfad: {csv_path}")
        df_truth = pd.read_csv(csv_path, usecols=range(4))

        df_truth = df_truth.dropna(subset=["question", "answer"])
        df_truth = df_truth[(df_truth["question"].str.strip() != "") & (df_truth["answer"].str.strip() != "")]
        
        documents = df_truth["document"].tolist()
        names = df_truth["name"].tolist()
        questions = df_truth["question"].tolist()
        answers = df_truth["answer"].tolist()

        return documents, names, questions, answers


    def to_strings_for_deepeval(self, retrieved):
        texts = []
        for r in retrieved or []:
            if isinstance(r, dict):
                if "content" in r and isinstance(r["content"], str):
                    texts.append(r["content"])
                # Fallbacks
                elif "text" in r and isinstance(r["text"], str):
                    texts.append(r["text"])
                elif "node" in r and hasattr(r["node"], "get_content"):
                    try:
                        texts.append(str(r["node"].get_content()))
                    except Exception:
                        pass

            elif hasattr(r, "get_content"):
                texts.append(str(r.get_content()))
            elif hasattr(r, "text"):
                texts.append(str(r.text))
            elif isinstance(r, str):
                texts.append(r)

        return [t.strip() for t in texts if isinstance(t, str) and t.strip()]


def sweep():
    
    secrets = read_config(DEFAULT_SECRETSFILE)
    config = read_config(DEFAULT_CONFIGFILE)

    ollama_models = ["mistral:7b", "qwen3:8b", "llama3.2:3b", "llama3.3:latest"]
    scores = {model: {"mean": 0, "std": 0, "time": 0} for model in ollama_models}
    print(scores)

    for model in ollama_models:
        config["model"]["llamastack"]["ollama_name"] = model
        tester = RagLlmTester(config, secrets)
        
        start_time = time.time()
        score_mean, score_std = tester.run_tests()
        end_time = time.time()
        
        test_time = end_time - start_time
        scores[model] = {"mean": score_mean, "std": score_std, "time": test_time}

    print("All scores:")
    print(scores)
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(scores.keys(), [s['mean'] for s in scores.values()], yerr=[s['std'] for s in scores.values()], fmt='x', capsize=3)
    plt.xlabel("Model")
    plt.ylabel("Score")
    
    # Add time information to title
    avg_time = np.mean([s['time'] for s in scores.values()])
    plt.title(f"Model Performance (Avg. Test Time: {avg_time:.2f}s)")
    plt.savefig("model_performance.png")


if __name__ == "__main__":

    sweep()
