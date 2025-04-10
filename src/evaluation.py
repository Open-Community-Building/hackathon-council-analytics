import torch
import json
import os
import pandas as pd
from time import sleep

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

from ragllm import RagLlm
from query import read_config, query, retrieve


DEFAULT_CONFIGFILE = os.path.expanduser(os.path.join('~','.config','hca','config_test.toml'))
DEFAULT_SECRETSFILE = os.path.expanduser(os.path.join('~','.config','hca','secrets.toml'))


class EvalHuggingFaceLLM(DeepEvalBaseLLM):
    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model_4bit = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            device_map="auto",
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct"
        )

        self.model = model_4bit
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        model = self.load_model()

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_new_tokens=500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        parser = JsonSchemaParser(schema.schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            pipeline.tokenizer, parser
        )

        output_dict = pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)
        output = output_dict[0]["generated_text"][len(prompt) :]
        json_result = json.loads(output)

        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Llama-3.1 8B"


def load_ground_truth(csv_path):
    """Load ground truth CSV file containing questions and related answers and the document to find it in"""
    df_truth = pd.read_csv(csv_path, usecols=range(4))

    documents = df_truth["document"].tolist()
    names = df_truth["name"].tolist()
    questions = df_truth["question"].tolist()
    answers = df_truth["answer"].tolist()

    return documents, names, questions, answers


def main():

    secrets = read_config(DEFAULT_SECRETSFILE)
    config = read_config(DEFAULT_CONFIGFILE)

    csv_path = "/media/ncdata/__groupfolders/4/TestDocuments/questions.csv"
    documents, names, questions, answers = load_ground_truth(csv_path)

    scores = []

    eval_llm = EvalHuggingFaceLLM()
    metric_faithful = FaithfulnessMetric(model=eval_llm)

    for question in questions:
        rag_llm = RagLlm(config=config, secrets=secrets)
        retrieved_texts = rag_llm.retrieve_docs(question)
        generated_answer = rag_llm.run_query(question) 

        print(f"Question: {question}")
        print(f"Retrieved text: {retrieved_texts}")
        print(f"Answer: {generated_answer}")

        # test_case = LLMTestCase(
        #     input=question,
        #     actual_output=generated_answer,
        #     retrieval_context=retrieved_texts
        # )

        # metric_faithful.measure(test_case)
        # print(metric_faithful.score)
        # print(metric_faithful.reason)

        # scores.append((metric_faithful.score, question))


if __name__ == "__main__":

    main()