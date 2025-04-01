import json
from typing import Optional
from transformers import PreTrainedTokenizerBase, AutoTokenizer
import random
import numpy as np
import matplotlib.pyplot as plt

def plot_token_length_distribution(dataset_path: str, tokenizer: PreTrainedTokenizerBase):
    # Load and process dataset
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset = [(data["conversations"][0]["value"],
               data["conversations"][1]["value"]) for data in dataset]

    # Get input and output lengths
    input_lengths = []
    output_lengths = []
    
    for prompt, completion in dataset:
        input_len = len(tokenizer(prompt).input_ids)
        output_len = len(tokenizer(completion).input_ids)
        input_lengths.append(input_len)
        output_lengths.append(output_len)

    # Create density plot
    plt.figure(figsize=(12, 6))
    
    plt.hist(input_lengths, bins=50, density=True, alpha=0.5, label='Input Length', color='blue')
    plt.hist(output_lengths, bins=50, density=True, alpha=0.5, label='Output Length', color='red')
    
    plt.title('Distribution of Input and Output Token Lengths')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('./figs/token_length_distribution.png')
    plt.close()

    # Print statistics
    print(f"Input length stats: mean={np.mean(input_lengths):.1f}, median={np.median(input_lengths):.1f}, "
          f"min={min(input_lengths)}, max={max(input_lengths)}")
    print(f"Output length stats: mean={np.mean(output_lengths):.1f}, median={np.median(output_lengths):.1f}, "
          f"min={min(output_lengths)}, max={max(output_lengths)}")

tokenizer = AutoTokenizer.from_pretrained("/data/model/Llama-3.1-8B")
plot_token_length_distribution("/data/sharegpt.json", tokenizer)

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
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    # random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: list[tuple[str, int, int]] = []
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
        if prompt_len < 4 or (fixed_output_len is None and prompt_len + output_len < 256):
            # Prune too short sequences.
            continue
        if prompt_len + output_len > 4096:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len, None))

    return filtered_dataset