# SPDX-License-Identifier: Apache-2.0

import gc
import time
from typing import List
import sys
import os
from transformers import AutoTokenizer, AutoConfig
sys.path.append('/home/nudt/lirui/vllm_speculative/')
from vllm.inputs import TokensPrompt
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser
import json
from vllm import LLM, SamplingParams
from typing import Optional
from transformers import PreTrainedTokenizerBase
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from examples.utils_data import Gsm8k_dataset, CSQA_dataset, AQuA_dataset
import re
import json
import torch
import random
from typing import List
from pydantic import BaseModel
from openai import OpenAI
from vllm.sampling_params import GuidedDecodingParams
import pickle
from typing import Union, Sequence
class Step(BaseModel):
    explanation: str
    output: str

class MathResponse(BaseModel):
    steps: list[Step]
    final_answer: str

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

prompt_format = dataset2prompt["gov_report"]
context_length = {
        "longchat7b": 32768,
        "longchat13b": 16384,
        "vicuna7b": 16384,
        "vicuna13b": 16384,
        "llama8b": 262000,
}
context_length = 8000 #context_length["llama8b"] - 2000
max_tokens = 4090
num_speculative_tokens = 5
ngram_prompt_lookup_max = 4
def extract_numbers(text):
    pattern = r"\\boxed\{([^{}]*)\}"
    match = re.findall(pattern, text)
    number = None
    if match:
        # Reverse iterate through matches
        match = match[::-1]      
        for boxed_content in match:
            # Clean the content (remove all non-digit characters except decimal points)
            clean_number = re.sub(r'[^\d.]', '', boxed_content)
            if clean_number:
                return [clean_number]
    if not number or not number.replace('.', '').replace(',', '').isdigit():
        pattern = r'(\d+(?:[,\\!]+\d+)+|\d+)'
        results = []
        for match in re.finditer(pattern, text):
            # 提取匹配的完整文本
            full_match = match.group(0)
            # 移除所有非数字字符
            clean_number = re.sub(r'[^0-9]', '', full_match)
            results.append(clean_number)
        return results if results else [0]
    return [number]
def answer_cleansing_gsm8k(pred):
    pred = pred.lower()
    # direct_answer_trigger_for_fewshot = "final answer".lower()
    # preds = pred.split(direct_answer_trigger_for_fewshot)

    # pred = preds[-1]
    # pred = pred.replace(",", "")
    # pred = pred.replace(" ", "")
    pred = extract_numbers(pred)
    # If there is no candidate in list, null is set.
    # if len(pred) == 0:
    #     return random.randint(0, 100)

    pred = pred[-1]
    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "" and type(pred) == str:
        if pred[-1] == ".":
            pred = pred[:-1]
    return float(pred)
def generate_meta_prompts(tokenizer):
    start = time.time()
    meta_prompts = []
    data_task = []
    for line in open('/data/gov_report.jsonl').readlines():
        raw_data = json.loads(line)
        data_task.append(raw_data)
    for raw_data in data_task:
        prompt = prompt_format.format(**raw_data)
        
        # tokenizer.pad_token_id = 128001
        # tokenizer.eos_token_id = 128009
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        a = tokenizer(prompt)
        a_cuda = a.input_ids
        len_cuda = len(a_cuda)
        if context_length <= len_cuda:
            # Truncate to context_length tokens
            a_cuda = a_cuda[:context_length]
            # Re-decode truncated tokens back to text
            prompt = tokenizer.decode(a_cuda)
            len_cuda = context_length
            meta_prompts.append((
                prompt, # TODO
                a_cuda,
                SamplingParams(ignore_eos=True, max_tokens=1024, temperature=0.0)
            ))
    end = time.time()
    print(f"read datatime: {end-start}")
    return meta_prompts


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> list[tuple[str, int, int, None]]:
    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    # Keep all conversation turns
    # dataset = [[(turn["value"]) for turn in data["conversations"]] for data in dataset]
    # # Flatten the conversations into pairs of consecutive turns
    # dataset = [(turns[i], turns[i+1]) for turns in dataset for i in range(len(turns)-1)]
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    # random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    meta_prompts = []
    # Initialize progress bar for dataset processing
    from tqdm import tqdm
    pbar = tqdm(total=num_requests, desc="Processing dataset")
    for i in range(len(dataset)):
        if len(meta_prompts) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4: #or (fixed_output_len is None and prompt_len + output_len < 256):
            # Prune too short sequences.
            continue
        if prompt_len + output_len > 4096:
            # Prune too long sequences.
            continue
        
        # Truncate to context_length tokens
        # prompt_token_ids = prompt_token_ids[:context_length]
        # Re-decode truncated tokens back to text
        # prompt = tokenizer.decode(prompt_token_ids)
        meta_prompts.append((
            prompt, # TODO
            prompt_token_ids,
            SamplingParams(ignore_eos=False, max_tokens=output_len, temperature=0.0)
        ))
        pbar.update(1)
    pbar.close()
    return meta_prompts

def time_generation(llm: LLM, prompts: List[str],
                    sampling_params: SamplingParams):
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    # Warmup first
    # llm.generate(prompts, sampling_params)
    # llm.generate(prompts, sampling_params)
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()
    for idx, output in enumerate(outputs):
        # print(output)
        generated_text = output.outputs[0].text[-10:]
        print(f"text: {generated_text!r}")
        print(f"{idx},len(output.outputs[0].token_ids): ", len(output.outputs[0].token_ids))
        
    print("end-start: ", end-start)
    return (end-start) / sum([len(o.outputs[0].token_ids) for o in outputs])

def calculate_results(outputs):
    """计算结果统计信息"""
    early_time = float('inf')
    last_token_time = float(0)
    anss = []
    results = []
    for index,output in enumerate(outputs):
        if output is None:
            # 处理未生成的输出
            anss.append(None)
            continue
            
        generated_text = output.outputs[0].text[-100:]
        # print(f"text: {generated_text!r}")
        # print(f"len(output.outputs[0].token_ids): ", len(output.outputs[0].token_ids))
        
        if len(output.outputs[0].token_ids) > 0:
            early_time = min(early_time, output.metrics.first_token_time)
            last_token_time = max(last_token_time, output.metrics.finished_time)
        
        ans = answer_cleansing_gsm8k(output.outputs[0].text)
        anss.append(ans)
        result = {
            "result": output.outputs[0].text,
            "answer": ans,
            "time": output.metrics.finished_time - output.metrics.first_token_time
        }
        results.append(result)
    # 计算有效token
    valid_outputs = [o for o in outputs if o is not None]
    total_tokens = sum([len(o.outputs[0].token_ids) for o in valid_outputs])
    
    # 计算总时间
    total_time = last_token_time - early_time if valid_outputs else 0
    avg_time_per_token = total_time / total_tokens if total_tokens > 0 else 0
    
    print(f"Total generation time: {total_time:.2f}s for {total_tokens} tokens")
    print(f"Average time per token: {avg_time_per_token:.6f}s")
    print(f"Processed {len(valid_outputs)} out of {len(outputs)} prompts")
    
    return avg_time_per_token, anss,results

def update_typical_acceptance_threshold(llm, new_threshold,new_alpha):
    """Update the posterior_threshold of the TypicalAcceptanceSampler."""
    return llm.llm_engine.model_executor.update_typical_acceptance_threshold(new_threshold,new_alpha)

def change_spec_decode_sampler(llm, draft_token_acceptance_method):
    return llm.llm_engine.model_executor.change_spec_decode_sampler(draft_token_acceptance_method)

def get_metrics(llm):
    return llm.llm_engine.model_executor.get_metrics()

def clear_metrics(llm):
    return llm.llm_engine.model_executor.clear_metrics()


def get_times_list(llm,datasets,num_speculative_tokens,ngram_prompt_lookup_max):
    times_list = []
    
    for index,meta_prompts in enumerate(datasets):               
        for batch_size in [16,32]:
            prompts = []
            sampling_paramss = []
            for idx, (prompt, prompt_token_ids, sampling_params) in enumerate(meta_prompts):
                prompts.append(prompt)
                sampling_paramss.append(sampling_params)
                if len(prompts) >= batch_size:
                    break
            
            elapse_time1 = time_generation(llm, prompts, sampling_paramss)
            times_dict = {
                'num_speculative_tokens': num_speculative_tokens,
                'ngram_prompt_lookup_max': ngram_prompt_lookup_max,
                'batch_size': batch_size,
                'elapse_time': elapse_time1
            }
            print(times_dict)
            times_list.append(times_dict)
    return times_list
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_names', type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--max_seq_len', type=int, default=1024)
    parser.add_argument('--max_batch_size', type=int, default=4)
    parser.add_argument('--data_path', type=str, default="/home/nudt/lirui/vllm_speculative/examples/data")
    parser.add_argument('--dataset', choices=['GSM8K', 'CSQA',"AQuA"],default="GSM8K")
    parser.add_argument('--out_path', type=str, default="output/singlemodel")
    parser.add_argument('--max_gen_len', type=int, default=2000)
    parser.add_argument('--batch_size', type=list, default=[1,1])
    parser.add_argument('--few_shot', type=int,help="GSM8K:8 CSQA:7")
    parser.add_argument('--test_size', type=int, default=3, 
                       help="Number of samples to test. If None, use full dataset")
    args = parser.parse_args()

    model_name = "/data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    speedups = []
    right = 0
    
    # llm = None
    num_requests = 40
    datasets = []
    times_list1 = []
    datasets.append(sample_sharegpt_requests("/data/sharegpt.json", num_requests, tokenizer))
    # datasets.append(generate_meta_prompts(tokenizer))
    for i in range(len(datasets)):
        print("len datasets[",i,"]",len(datasets[i]))
    speedups = []
    llm = LLM(model=model_name,max_model_len=10156, enforce_eager=True)

    times_list1 = get_times_list(llm,datasets,0,0)
    del llm
    gc.collect()
    print(times_list1)
    
    llm = LLM(
        model=model_name,
        speculative_model="[ngram]",
        max_model_len=10156,
        num_speculative_tokens=num_speculative_tokens,
        ngram_prompt_lookup_max=ngram_prompt_lookup_max,
        enforce_eager=True
    )
    times_list2 = []
    for num_speculative_tokens in [1,2,3,4,5,6,7]:
        for ngram_prompt_lookup_max in [1,2,3,4,5,6,7]:
            llm.llm_engine.scheduler[0].set_num_lookahead_slots(num_speculative_tokens)
            llm.llm_engine.vllm_config.speculative_config.num_speculative_tokens = num_speculative_tokens
            llm.llm_engine.model_executor.set_ngram_prompt_lookup_window_size(1,ngram_prompt_lookup_max)
            times_list2.extend(get_times_list(llm,datasets,num_speculative_tokens,ngram_prompt_lookup_max))
    del llm
    gc.collect()
    print(times_list1)
    print(times_list2)
    
    print(len(times_list1),len(times_list2))
    times_list1 = times_list1 * int(len(times_list2)/len(times_list1))
    # Calculate speedup ratios
    speedup_results = []
    for t1, t2 in zip(times_list1, times_list2):
        # Extract the average time per token from elapse_time tuple (first element)
        baseline_time = t1['elapse_time']
        ngram_time = t2['elapse_time']
        
        if baseline_time > 0:  # Avoid division by zero
            speedup = baseline_time / ngram_time
        else:
            speedup = 0.0
            
        speedup_result = {
            'num_speculative_tokens': t2['num_speculative_tokens'],
            'ngram_prompt_lookup_max': t2['ngram_prompt_lookup_max'],
            'batch_size': t2['batch_size'],
            'baseline_time': baseline_time,
            'ngram_time': ngram_time,
            'speedup_ratio': speedup
        }
        speedup_results.append(speedup_result)
    
    # Save results to files
    import json
    
    # Save times_list1 (baseline model times)
    with open('baseline_times.json', 'w') as f:
        json.dump(times_list1, f, indent=2)
    
    # Save times_list2 (ngram speculative decoding times)
    with open('ngram_times.json', 'w') as f:
        json.dump(times_list2, f, indent=2)
    
    # Save speedup ratios
    with open('speedup_ratios.json', 'w') as f:
        json.dump(speedup_results, f, indent=2)
    
    # Print summary statistics
    avg_speedup = sum(result['speedup_ratio'] for result in speedup_results) / len(speedup_results)
    max_speedup = max(result['speedup_ratio'] for result in speedup_results)
    min_speedup = min(result['speedup_ratio'] for result in speedup_results)
    
    print(f"\nSummary of Acceleration Ratios:")
    print(f"Average Speedup: {avg_speedup:.2f}x")
    print(f"Maximum Speedup: {max_speedup:.2f}x")
    print(f"Minimum Speedup: {min_speedup:.2f}x")
    
    # Find best configuration
    best_config = max(speedup_results, key=lambda x: x['speedup_ratio'])
    print(f"\nBest Configuration:")
    print(f"  Speculative Tokens: {best_config['num_speculative_tokens']}")
    print(f"  NGram Lookup Max: {best_config['ngram_prompt_lookup_max']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  Speedup: {best_config['speedup_ratio']:.2f}x")
    
    # Analyze impact of different parameters
    # Group by num_speculative_tokens
    by_spec_tokens = {}
    for result in speedup_results:
        tokens = result['num_speculative_tokens']
        if tokens not in by_spec_tokens:
            by_spec_tokens[tokens] = []
        by_spec_tokens[tokens].append(result['speedup_ratio'])
    
    print("\nAverage Speedup by Number of Speculative Tokens:")
    for tokens, ratios in sorted(by_spec_tokens.items()):
        avg = sum(ratios) / len(ratios)
        print(f"  {tokens} tokens: {avg:.2f}x")
    
    # Group by ngram_prompt_lookup_max
    by_ngram_lookup = {}
    for result in speedup_results:
        lookup = result['ngram_prompt_lookup_max']
        if lookup not in by_ngram_lookup:
            by_ngram_lookup[lookup] = []
        by_ngram_lookup[lookup].append(result['speedup_ratio'])
    
    print("\nAverage Speedup by NGram Prompt Lookup Max:")
    for lookup, ratios in sorted(by_ngram_lookup.items()):
        avg = sum(ratios) / len(ratios)
        print(f"  Lookup {lookup}: {avg:.2f}x")
    
    # Group by batch_size
    by_batch_size = {}
    for result in speedup_results:
        batch = result['batch_size']
        if batch not in by_batch_size:
            by_batch_size[batch] = []
        by_batch_size[batch].append(result['speedup_ratio'])
    
    print("\nAverage Speedup by Batch Size:")
    for batch, ratios in sorted(by_batch_size.items()):
        avg = sum(ratios) / len(ratios)
        print(f"  Batch size {batch}: {avg:.2f}x")
                