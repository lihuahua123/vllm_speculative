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
import random
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
max_tokens = 1024
num_speculative_tokens = 10
def extract_numbers(text):
    # 1. 先找到所有数字组（可能被逗号或\!分隔）
    # 2. 然后将这些分隔符移除，只保留数字
    pattern = r'(\d+(?:[,\\!]+\d+)+|\d+)'
    
    results = []
    for match in re.finditer(pattern, text):
        # 提取匹配的完整文本
        full_match = match.group(0)
        # 移除所有非数字字符
        clean_number = re.sub(r'[^0-9]', '', full_match)
        results.append(clean_number)
    
    return results
def answer_cleansing_gsm8k(pred):
    pred = pred.lower()
    direct_answer_trigger_for_fewshot = "final answer".lower()
    preds = pred.split(direct_answer_trigger_for_fewshot)

    pred = preds[-1]
    pred = pred.replace(",", "")
    pred = pred.replace(" ", "")
    pred = extract_numbers(pred)
    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        return random.randint(0, 100)

    pred = pred[-1]
    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
    return int(pred)
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
        if prompt_len < context_length: #or (fixed_output_len is None and prompt_len + output_len < 256):
            # Prune too short sequences.
            continue
        if prompt_len + output_len > 28688:
            # Prune too long sequences.
            continue
        
        # Truncate to context_length tokens
        prompt_token_ids = prompt_token_ids[:context_length]
        # Re-decode truncated tokens back to text
        prompt = tokenizer.decode(prompt_token_ids)
        meta_prompts.append((
            prompt, # TODO
            prompt_token_ids,
            SamplingParams(ignore_eos=True, max_tokens=1024, temperature=0.0)
        ))
        pbar.update(1)
    pbar.close()
    return meta_prompts

def time_generation(llm: LLM, prompts,
                    sampling_params: SamplingParams):
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    # Warmup first
    # llm.generate(prompts, sampling_params)
    # llm.generate(prompts, sampling_params)
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    
    end = time.time()
    #print((end-start),sum([len(o.outputs[0].token_ids) for o in outputs]))
    # print((end - start) / sum([len(o.outputs[0].token_ids) for o in outputs]))
    # Print the outputs.
    early_time = float('inf')
    last_token_time = float(0)
    for idx, output in enumerate(outputs):
        # print(output)
        generated_text = output.outputs[0].text[-100:]
        print(f"text: {generated_text!r}")
        print(f"{idx},len(output.outputs[0].token_ids): ", len(output.outputs[0].token_ids))
        if len(output.outputs[0].token_ids) > 0:
            early_time = min(early_time, output.metrics.first_token_time)
            last_token_time = max(last_token_time, output.metrics.finished_time)
        ans = answer_cleansing_gsm8k(output.outputs[0].text)
    return (end-start) / sum([len(o.outputs[0].token_ids) for o in outputs]),ans
    

def update_typical_acceptance_threshold(llm, new_threshold):
    """Update the posterior_threshold of the TypicalAcceptanceSampler."""
    return llm.llm_engine.model_executor.update_typical_acceptance_threshold(new_threshold)
    

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

    # template = (
    #     "Below is an instruction that describes a task. Write a response "
    #     "that appropriately completes the request.\n\n### Instruction:\n{}"
    #     "\n\n### Response:\n")

    # # Sample prompts.
    # prompts = [
    #     "Write about the president of the United States.",
    # ]
    # prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids) for prompt, prompt_token_ids, _ in meta_prompts][:2]
    # meta_prompts = generate_meta_prompts(tokenizer)
    model_name = "/data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    datasets = []
    # datasets.append(sample_sharegpt_requests("/data/sharegpt.json", 56, tokenizer))
    if args.dataset =="GSM8K":
        dataset=Gsm8k_dataset(tokenizer,args,stage2=False)
    elif args.dataset =="CSQA":
        dataset=CSQA_dataset(tokenizer,args,stage2=False)
    elif args.dataset =="AQuA":
        dataset=AQuA_dataset(tokenizer,args,stage2=False)
    if args.test_size:
        dataset.data = dataset.data[:args.test_size]
    dataloader = DataLoader(dataset, batch_size=1)
    speedups = []
    right = 0
    # llm = LLM(model=model_name,max_model_len=10156, enforce_eager=True)
    llm = LLM(
            model=model_name,
            speculative_model="alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B",
            max_model_len=10156,
            num_speculative_tokens=num_speculative_tokens,
            # spec_decoding_acceptance_method="typical_acceptance_sampler",
            # typical_acceptance_sampler_posterior_alpha=0.3,
            # typical_acceptance_sampler_posterior_threshold=0.09,
            # ngram_prompt_lookup_max=4,
            enforce_eager=True
        )
    begin = time.time()
    indexs = [3,4]
    finish_count = 0
    for index,batch_data in tqdm(enumerate(dataloader)):
        print(index)
        if index not in indexs:
            continue
        finish_count += 1
        prompts,right_answer = batch_data
        # prompts = TokensPrompt(prompt_token_ids=prompts)
        # Create a sampling params object.
        sampling_params = SamplingParams(ignore_eos=False, max_tokens=max_tokens, temperature=0.8)

        # Create an LLM without spec decoding
        
        # max_model_len=40560
        # if update_typical_acceptance_threshold(llm,0.8) is None:
        #     print("update failed")
            
        print("Without speculation")
        elapse_time1,ans1 = time_generation(llm, prompts, sampling_params)
        
        if str(ans1)==str(right_answer.cpu().item()):
            right+=1
        else:
            print(f"wrong: {index}")
            print(f"ans1: {ans1}, ans2: {right_answer}")
        print(f"elapse_time: {elapse_time1}")

        # del llm
        # gc.collect()

        # # Create an LLM with spec decoding
        # # 这里的max_model_len 是一个句子最大长度
        # llm = LLM(
        #     model=model_name,
        #     speculative_model="[ngram]",#"alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B",
        #     max_model_len=10156,
        #     num_speculative_tokens=num_speculative_tokens,
        #     ngram_prompt_lookup_max=4,
        #     enforce_eager=True
        # )

        # print("With speculation")
        # elapse_time2,ans2 = time_generation(llm, prompts, sampling_params)
        # if str(ans2)==str(dataset.data[index]["answer"]):
        #     print(f"right222:!!!!!")
        # speedup = elapse_time1 / elapse_time2
        # speedups.append(speedup)
        # print(f" speedup: {speedup:.2f}x")
        # del llm
        # gc.collect()
        # #print(f"elapse_time: {elapse_time2}")
        # print("num_speculative_tokens: ", num_speculative_tokens)
        # print("context_length,max_tokens: ", context_length,max_tokens)
        # print(speedups)
    end = time.time()
    print(f"right: {right}, finish_count: {finish_count}")
    print(f"time: {end-begin}")