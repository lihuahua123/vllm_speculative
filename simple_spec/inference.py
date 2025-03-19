import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Dict, Tuple
import time
import numpy as np
from tqdm import tqdm
import json
from transformers import PreTrainedTokenizerBase
def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int, None]]:
    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    # random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or (fixed_output_len is None and output_len < 4):
            # Prune too short sequences.
            continue
        if prompt_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len, None))

    return filtered_dataset
class LLMInference:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.eval()
        self.perf_stats: List[Dict[str, float]] = []

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        batch_size: int = 4,
        **generation_kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of prompts
        """
        all_responses = []
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize the batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Generate responses
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **generation_kwargs
            )

            # Decode the generated responses
            decoded_outputs = self.tokenizer.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            all_responses.extend(decoded_outputs)

        return all_responses

    @torch.no_grad()
    def measure_prefill_time(
        self,
        prompts: List[str],
        batch_size: int,
    ) -> float:
        """Measure prefill time for a batch of prompts"""
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Warm up
        _ = self.model.generate(**inputs, max_new_tokens=1)
        torch.cuda.synchronize()
        
        # Measure prefill time
        start_time = time.perf_counter()
        _ = self.model.generate(**inputs, max_new_tokens=1)
        torch.cuda.synchronize()
        prefill_time = time.perf_counter() - start_time
        
        # Record statistics
        avg_prompt_len = inputs.input_ids.shape[1]
        self.perf_stats.append({
            'batch_size': batch_size,
            'avg_prompt_len': avg_prompt_len,
            'prefill_time': prefill_time
        })
        
        return prefill_time

def collect_performance_data(model_path: str, dataset_path: str):
    inference = LLMInference(model_path)
    
    # Get all prompts from dataset
    dataset = sample_sharegpt_requests(
        dataset_path=dataset_path,
        num_requests=500,  # 从1000减少到100个样本
        tokenizer=inference.tokenizer
    )
    all_prompts = [item[0] for item in dataset]
    
    print("Processing prompts...")
    # Group prompts by length
    prompt_lengths = []
    for prompt in tqdm(all_prompts, desc="Tokenizing"):
        tokens = inference.tokenizer(prompt).input_ids
        prompt_lengths.append(len(tokens))
    
    # Create length buckets with finer granularity
    length_buckets = {}
    bucket_size = 16  # Reduced bucket size for finer granularity
    
    # Ensure we have samples across the full range
    target_lengths = list(range(16, 2049, bucket_size))
    for target_len in target_lengths:
        length_buckets[target_len] = []
    
    print("Organizing prompts into length buckets...")
    # Distribute prompts into buckets
    for prompt, length in tqdm(zip(all_prompts, prompt_lengths), desc="Bucketing", total=len(all_prompts)):
        if length > 1024:  # Skip too long prompts
            continue
        bucket = (length + bucket_size - 1) // bucket_size * bucket_size
        if bucket < 16:  # Ensure minimum length
            bucket = 16
        if bucket in length_buckets:
            length_buckets[bucket].append(prompt)
    
    # If some buckets are empty, generate synthetic prompts
    print("Filling empty buckets with synthetic prompts...")
    empty_buckets = [k for k, v in length_buckets.items() if not v]
    for bucket_len in tqdm(empty_buckets, desc="Synthetic"):
        # Create a synthetic prompt of target length
        synthetic_prompt = "This is a test prompt. " * ((bucket_len + 3) // 4)
        tokens = inference.tokenizer(synthetic_prompt).input_ids
        if len(tokens) >= bucket_len:
            synthetic_prompt = inference.tokenizer.decode(tokens[:bucket_len])
            length_buckets[bucket_len].append(synthetic_prompt)
    
    # Test different batch sizes and prompt lengths
    batch_sizes = [1, 2, 4, 8, 16, 32]  # 减少测试的batch size数量
    num_repeats = 2  # 从3次重复减少到2次
    
    # Calculate total iterations for progress bar
    total_iters = sum(1 for k in length_buckets.keys() if length_buckets[k]) * len(batch_sizes)
    
    print("\nMeasuring performance...")
    with tqdm(total=total_iters, desc="Testing combinations") as pbar:
        for length_bucket in sorted(length_buckets.keys()):
            prompts = length_buckets[length_bucket]
            if not prompts:  # Skip empty buckets
                continue
                
            tqdm.write(f"\nTesting length bucket: {length_bucket}")
            
            # Ensure we have enough prompts by duplicating if necessary
            while len(prompts) < 128:
                prompts.extend(prompts)
            prompts = prompts[:128]  # Limit to max needed
            
            for batch_size in batch_sizes:
                batch_prompts = prompts[:batch_size]
                try:
                    # Take multiple measurements and average
                    times = []
                    for _ in range(num_repeats):
                        prefill_time = inference.measure_prefill_time(
                            batch_prompts,
                            batch_size
                        )
                        times.append(prefill_time)
                    avg_time = sum(times) / len(times)
                    tqdm.write(f"Batch {batch_size}, Length {length_bucket}: {avg_time:.3f}s "
                          f"(min: {min(times):.3f}s, max: {max(times):.3f}s)")
                except RuntimeError as e:
                    tqdm.write(f"Failed with batch {batch_size}: {e}")
                    break
                
                # Add a small delay between measurements
                time.sleep(0.1)
                pbar.update(1)
    
    return inference.perf_stats

def main():
    # model_path = #"alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B"#
    model_path = "/data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B"
    dataset_path = "/data/sharegpt.json"
    
    # Collect performance data
    perf_stats = collect_performance_data(model_path, dataset_path)
    
    # Basic analysis of collected data
    lengths = [stat['avg_prompt_len'] for stat in perf_stats]
    batch_sizes = [stat['batch_size'] for stat in perf_stats]
    times = [stat['prefill_time'] for stat in perf_stats]
    
    print("\nData Collection Summary:")
    print(f"Total samples: {len(perf_stats)}")
    print(f"Prompt length range: {min(lengths)} to {max(lengths)}")
    print(f"Batch sizes tested: {sorted(set(batch_sizes))}")
    print(f"Time range: {min(times):.3f}s to {max(times):.3f}s")
    
    # Save performance data
    import json
    with open(f'perf_stats_{model_path.split("/")[-1]}.json', 'w') as f:
        json.dump(perf_stats, f, indent=2)

if __name__ == "__main__":
    main() 