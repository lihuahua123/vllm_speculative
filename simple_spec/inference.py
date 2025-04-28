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
    spec_len: Optional[int] = None,
    begin_index: Optional[int] = None,
) -> List[Tuple[str, int, int, None]]:
    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]
    if begin_index is not None:
        dataset = dataset[begin_index:]
    # Shuffle the dataset.
    # random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    filtered_dataset2 = []
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
        if prompt_len < 10 or (fixed_output_len is None and output_len < 10):
            # Prune too short sequences.
            continue
        if prompt_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset2.append((prompt, prompt_len, output_len, None))
        prompt_token_ids = prompt_token_ids[:-spec_len]
        filtered_dataset.append((tokenizer.decode(prompt_token_ids), len(prompt_token_ids), output_len, None))
    return filtered_dataset, filtered_dataset2


def main():
    # model_path = #"alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B"#
    model_path = "/data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B"
    dataset_path = "/data/sharegpt.json"
    
    

if __name__ == "__main__":
    main() 