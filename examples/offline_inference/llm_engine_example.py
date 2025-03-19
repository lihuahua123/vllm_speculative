# SPDX-License-Identifier: Apache-2.0

import argparse
from typing import List, Tuple
from transformers import AutoTokenizer, AutoConfig
import sys
import os
sys.path.append('/home/nudt/lirui/vllm_speculative/')

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser
import json

dataset2prompt = {
    "gov_report": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nYou are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\nNow, write a one-page summary of the report.</s>\n"
        "<s>assistant\nSummary:"
    ),
    "qmsum": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nYou are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}</s>\n"
        "<s>assistant\nAnswer:"
    ),
    "multi_news": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nYou are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\nNow, write a one-page summary of all the news.</s>\n"
        "<s>assistant\nSummary:"
    ),
    "lcc": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nPlease complete the code given below. \n{context}Now, complete the code given.</s>\n"
        "<s>assistant\n"
    ),
    "repobench-p": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nPlease complete the code given below. \n{context}Now, complete the code given.</s>\n"
        "<s>assistant\n"
    ),
}
target_model_name = "/data/model/gradientaiLlama-3-8B-Instruct-262k"
config = AutoConfig.from_pretrained(target_model_name)
tokenizer = AutoTokenizer.from_pretrained(target_model_name)
with open('/home/nudt/lirui/LongSpec-main/longspec/long-bench_results/gov_report/temp_data_task.json', "r") as f:
        data_task = json.load(f)
prompt_format = dataset2prompt["gov_report"]
context_length = {
        "longchat7b": 32768,
        "longchat13b": 16384,
        "vicuna7b": 16384,
        "vicuna13b": 16384,
        "llama8b": 262000,
}
context_length = context_length["llama8b"] - 2000
meta_prompts = []
for raw_data in data_task:
    prompt = prompt_format.format(**raw_data)
    
    tokenizer.pad_token_id = 128001
    tokenizer.eos_token_id = 128009
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    a = tokenizer(prompt, return_tensors="pt", padding=True, padding_side="right")
    a_cuda = a['input_ids'].cuda()
    len_cuda = a['attention_mask'].sum(dim=-1).cuda()
    if 1200 < len_cuda <= context_length:
        meta_prompts.append((
            prompt,
            SamplingParams(ignore_eos=True, max_tokens=1200, temperature=0.0)
        ))
def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    return meta_prompts[:2]
    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1)),
        ("To be or not to be,",
         SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
        ("What is the meaning of life?",
         SamplingParams(n=2,
                        best_of=5,
                        temperature=0.8,
                        top_p=0.95,
                        frequency_penalty=0.1)),
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    test_prompts = create_test_prompts()
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
