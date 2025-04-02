# SPDX-License-Identifier: Apache-2.0
import multiprocessing
# 在导入torch或其他库之前设置多进程启动方法
multiprocessing.set_start_method('spawn', force=True)
import gc
import time
from typing import List
import sys
import os
from transformers import AutoTokenizer, AutoConfig
sys.path.append('/root/vllm_speculative/')
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
num_speculative_tokens = 10

def extract_numbers(text):
    """
<<<<<<< HEAD
    Extract numbers from text using regex patterns.
    只适配DeepSeek-R1-Distill-Qwen-7B
=======
    提取文本中的数字
    只能适配deepseek-aiDeepSeek-R1-Distill-Qwen-7B
>>>>>>> 76c878e7f208371495207c73975f1108cd408537
    """
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
                    sampling_params: Union[SamplingParams, Sequence[SamplingParams]], 
                    checkpoint_file='generation_checkpoint.pkl',
                    resume=False):
    """
    批量生成文本，并处理OOM问题。支持断点续传功能。
    
    Args:
        llm: LLM实例
        prompts: 提示列表
        sampling_params: 采样参数或采样参数列表
        checkpoint_file: 检查点文件路径
        resume: 是否从检查点恢复
    
    Returns:
        avg_time_per_token: 每个token的平均生成时间
        anss: 生成的答案列表
    """
    # 如果恢复模式，尝试加载现有进度
    processed_indices = set()
    all_outputs = []
    
    if resume and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                all_outputs = checkpoint_data['outputs']
                processed_indices = set(checkpoint_data['processed_indices'])
                print(f"Resumed from checkpoint with {len(processed_indices)} processed prompts")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from beginning")
            all_outputs = []
            processed_indices = set()
    
    # 准备待处理的提示
    remaining_prompts = []
    remaining_indices = []
    remaining_params = []
    
    # 确定是否使用单一采样参数还是参数列表
    is_params_sequence = isinstance(sampling_params, Sequence) and not isinstance(sampling_params, SamplingParams)
    
    for i, prompt in enumerate(prompts):
        if i not in processed_indices:
            remaining_prompts.append(prompt)
            remaining_indices.append(i)
            if is_params_sequence:
                # 使用对应的采样参数
                if i < len(sampling_params):
                    remaining_params.append(sampling_params[i])
                else:
                    # 如果参数不够，使用最后一个参数
                    remaining_params.append(sampling_params[-1])
    
    if not remaining_prompts:
        print("All prompts already processed!")
        # 只返回已有的输出
        return calculate_results(all_outputs)
    
    print(f"Processing {len(remaining_prompts)} remaining prompts out of {len(prompts)} total")
    
    # 自适应批处理
    start = time.time()
    initial_batch_size = 32  # 初始批处理大小
    batch_size = initial_batch_size
    min_batch_size = 4  # 最小批处理大小
    
    i = 0
    while i < len(remaining_prompts):
        try:
            end_idx = min(i + batch_size, len(remaining_prompts))
            batch_prompts = remaining_prompts[i:end_idx]
            batch_indices = remaining_indices[i:end_idx]
            
            # 根据是否使用参数列表决定本批次的采样参数
            if is_params_sequence:
                batch_params = remaining_params[i:end_idx]
            else:
                # 使用单一参数
                batch_params = sampling_params
            
            print(f"Processing batch {i//batch_size + 1}/{(len(remaining_prompts)-1)//batch_size + 1} with {len(batch_prompts)} prompts (batch_size={batch_size})")
            
            # 记录内存使用前
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            
            # 生成文本
            batch_outputs = llm.generate(batch_prompts, batch_params, use_tqdm=True)
            
            # 记录内存使用后
            mem_after = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            mem_used = mem_after - mem_before
            print(f"Batch memory usage: {mem_used:.2f} GB")
            
            # 更新处理索引和输出
            for idx, output in zip(batch_indices, batch_outputs):
                processed_indices.add(idx)
                # 确保all_outputs的长度足够
                while len(all_outputs) <= idx:
                    all_outputs.append(None)
                all_outputs[idx] = output
            
            # 保存检查点
            checkpoint_data = {
                'outputs': all_outputs,
                'processed_indices': list(processed_indices)
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # 成功处理后前进到下一批
            i = end_idx
            
            # 清理内存
            torch.cuda.empty_cache()
            gc.collect()
            
            # 如果内存使用率适中，可以尝试增加批处理大小
            if mem_used < 0.7 * torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):
                batch_size = min(batch_size + 4, initial_batch_size)
                
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # 捕获OOM错误
            if "CUDA out of memory" in str(e) or "RuntimeError" in str(e):
                print(f"OOM encountered with batch_size={batch_size}, reducing batch size...")
                
                # 减少批处理大小
                batch_size = max(batch_size // 2, min_batch_size)
                
                if batch_size < min_batch_size:
                    print(f"Error: Cannot process prompts even with minimum batch size {min_batch_size}.")
                    print(f"Error details: {str(e)}")
                    
                    # 即使失败，也保存当前进度
                    checkpoint_data = {
                        'outputs': all_outputs,
                        'processed_indices': list(processed_indices)
                    }
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(checkpoint_data, f)
                    
                    raise
                
                # 清理内存并重试
                torch.cuda.empty_cache()
                gc.collect()
                print(f"Retrying with batch_size={batch_size}")
            else:
                # 非OOM错误，保存进度并抛出
                checkpoint_data = {
                    'outputs': all_outputs,
                    'processed_indices': list(processed_indices)
                }
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                
                raise
    
    end = time.time()
    
    return calculate_results(all_outputs, used_time=end-start)

def calculate_results(outputs, used_time=0):
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
        print(f"text: {generated_text!r}")
        # print(f"len(output.outputs[0].token_ids): ", len(output.outputs[0].token_ids))
        
        
        ans = answer_cleansing_gsm8k(output.outputs[0].text)
        anss.append(ans)
        result = {
            "result": output.outputs[0].text,
            "answer": ans,
        }
        results.append(result)
    # 计算有效token
    valid_outputs = [o for o in outputs if o is not None]
    total_tokens = sum([len(o.outputs[0].token_ids) for o in valid_outputs])
    
    # 计算总时间
    # total_time = last_token_time - early_time if valid_outputs else 0
    avg_time_per_token = used_time / total_tokens if total_tokens > 0 else 0
    
    print(f"Total generation time: {used_time:.2f}s for {total_tokens} tokens")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_names', type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--max_seq_len', type=int, default=1024)
    parser.add_argument('--max_batch_size', type=int, default=4)
    parser.add_argument('--data_path', type=str, default="/root/vllm_speculative/examples/data")
    parser.add_argument('--dataset', choices=['GSM8K', 'CSQA',"AQuA"],default="GSM8K")
    parser.add_argument('--out_path', type=str, default="output/singlemodel")
    parser.add_argument('--max_gen_len', type=int, default=2000)
    parser.add_argument('--batch_size', type=list, default=[1,1])
    parser.add_argument('--few_shot', type=int,help="GSM8K:8 CSQA:7")
    parser.add_argument('--test_size', type=int, default=3, 
                       help="Number of samples to test. If None, use full dataset")
    args = parser.parse_args()

    model_name = "/hy-tmp/lmsysvicuna-33b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    datasets = []
    # datasets.append(sample_sharegpt_requests("/data/sharegpt.json", 56, tokenizer))
    if args.dataset =="GSM8K":
        dataset=Gsm8k_dataset(tokenizer,args,stage2=False)
    elif args.dataset =="CSQA":
        dataset=CSQA_dataset(tokenizer,args,stage2=False)
    elif args.dataset =="AQuA":
        dataset=AQuA_dataset(tokenizer,args,stage2=False)
    # index =[4]
    if args.test_size and args.test_size > 0:
        dataset.data = dataset.data[:args.test_size]
    # dataset.data = [dataset.data[i] for i in range(args.test_size) if i   in index]
    dataloader = DataLoader(dataset, batch_size=1)
    speedups = []
    right = 0
    typical_acceptance_sampler_posterior_alpha=0.8
    typical_acceptance_sampler_posterior_threshold=0.5
    # llm = LLM(model=model_name,max_model_len=10156, enforce_eager=True)
    llm = LLM(
            model=model_name,
            tensor_parallel_size=4,
            speculative_model="[ngram]",#"alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B",
            #max_model_len=2048,
            num_speculative_tokens=num_speculative_tokens,
            # spec_decoding_acceptance_method="typical_acceptance_sampler",
            # typical_acceptance_sampler_posterior_alpha=typical_acceptance_sampler_posterior_alpha,
            # typical_acceptance_sampler_posterior_threshold=typical_acceptance_sampler_posterior_threshold,
            ngram_prompt_lookup_max=4,
            enforce_eager=True
        )
    # llm = None
    print("typical_acceptance_sampler_posterior_alpha",typical_acceptance_sampler_posterior_alpha)
    print("typical_acceptance_sampler_posterior_threshold",typical_acceptance_sampler_posterior_threshold)
    begin = time.time()
    
    indexs = [21,7]
    indexs2 = [4,12]
    # indexs2 = [3,21,36,37,38,43]
    # index3 = [17,36]
    # wrong_indexs = [2,3,4,5,7,8,11,12,15,21,24,37,43,44,49]
    # wrong_indexs2 = [1, 7, 8, 17, 37, 39, 43]
    # wrong_indexs_1024 = [2,7,8,12,13,15,20,21,23,41,43,44,46]
    # wrong_indexs_4096 = wrong_indexs
    index_buckets = {
        # "bucket1": [ 4, 7,  13, 29,  43, 46],     # [21]
        #  "bucket2": [2, 8, 12, 21, 37],    # [4, 12]
        # "bucket3": indexs3,    # [7]
        "default": []          # All other indices
    }
    
    # Initialize containers for prompts, answers and results
    buckets = {
        bucket_name: {
            "prompts": [],
            "answers": [],
            "results": None,
            "wrong_indices": []
        } for bucket_name in index_buckets.keys()
    }
    
    # Group prompts and answers into buckets
    for index, batch_data in tqdm(enumerate(dataloader)):
        prompts, right_answer = batch_data
        answer_value = right_answer.cpu().item()
        
        # Determine which bucket this prompt belongs to
        bucket_name = "default"
        for name, indices in index_buckets.items():
            if index in indices:
                bucket_name = name
                break
        # Store the prompt and answer in the appropriate bucket
        buckets[bucket_name]["prompts"].append(prompts[0])
        buckets[bucket_name]["answers"].append(answer_value)
    
    # Process each bucket and generate results
    right = 0
    for bucket_name, bucket_data in buckets.items():
        if not bucket_data["prompts"]:
            continue
            
        print(f"Processing bucket: {bucket_name} with {len(bucket_data['prompts'])} prompts")
        
        # Generate text for this bucket
        elapse_time, generated_answers, results = time_generation(
            llm, 
            bucket_data["prompts"], 
            SamplingParams(ignore_eos=False, max_tokens=max_tokens, temperature=0.0, seed=1234),
            checkpoint_file=f'checkpoint_{args.dataset}_{bucket_name}.pkl',
            resume=False
        )
        
        
        # Store results
        bucket_data["results"] = results
        
        # Verify answers and track correct/wrong results
        for i, (gen_answer, true_answer) in enumerate(zip(generated_answers, bucket_data["answers"])):
            # Add prompt to results for reference
            results[i]["prompt"] = bucket_data["prompts"][i]
            
            # Check if answer is correct
            if abs(float(gen_answer) - float(true_answer)) < 1e-6:
                right += 1
                results[i]["answer_correct"] = True
            else:
                bucket_data["wrong_indices"].append(i)
                results[i]["answer_correct"] = False
                results[i]["answer_correct_reason"] = true_answer
                # if bucket_name == "default":  # Only print default bucket wrong answers
                #     print(f"Wrong answer in bucket {bucket_name}, index {i}:", results[i])
    
    # Print summary
    print(f"Total correct answers: {right} out of {sum(len(b['prompts']) for b in buckets.values())}")
    
    # Print wrong indices for each bucket
    for bucket_name, bucket_data in buckets.items():
        if bucket_data["wrong_indices"]:
            print(f"Wrong indices in {bucket_name}: {bucket_data['wrong_indices']}")
    # print(results)
    # with open(f'results_{args.dataset}.pkl', 'wb') as f:
    #     pickle.dump(results, f)
    
    # Assuming aa is a dictionary or list containing tensors
    # def tensor_to_python(obj):
    #     if isinstance(obj, torch.Tensor):
    #         # Move tensor to CPU if it's on GPU
    #         if obj.is_cuda:
    #             obj = obj.cpu()
    #         # Convert to standard Python type (list, float, int)
    #         return obj.tolist()  # For multi-dimensional tensors
    #         # OR use .item() for single-value tensors:
    #         # return obj.item()  # Only for single-element tensors
        
    #     elif isinstance(obj, dict):
    #         # Process dictionaries
    #         return {key: tensor_to_python(value) for key, value in obj.items()}
        
    #     elif isinstance(obj, list):
    #         # Process lists
    #         return [tensor_to_python(item) for item in obj]
        
    #     # Return other types unchanged
    #     return obj

    # # Convert your tensor data to Python native types
    # aa_python = tensor_to_python(metrics_list)

    # # Now you can save to JSON
    # with open('aa_data.json', 'w') as f:
    #     json.dump(aa_python, f)