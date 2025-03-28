# SPDX-License-Identifier: Apache-2.0
"""Benchmark the latency of processing a single batch of requests."""

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from tqdm import tqdm
import sys

# Add current directory to Python path
current_dir = os.path.dirname('/home/nudt/lirui/vllm_speculative/vllm')
sys.path.append(current_dir)
print("Python path:")
for path in sys.path:
    print(f"  {path}")
try:
    import vllm
    print(f"\nvllm found at: {vllm.__file__}")
except ImportError as e:
    print(f"\nFailed to import vllm: {e}")


from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.sampling_params import BeamSearchParams
from vllm.utils import FlexibleArgumentParser
from typing import List

def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: dict[str, Any]) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={"latency": results["latencies"]},
        extra_info={k: results[k]
                    for k in ["avg_latency", "percentiles"]})
    if pt_records:
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def main(args: argparse.Namespace):
    print(args)

    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))
    assert llm.llm_engine.model_config.max_model_len >= (
        args.input_len +
        args.output_len), ("Please ensure that max_model_len is greater than"
                           " the sum of input_len and output_len.")

    sampling_params = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )
    print(sampling_params)
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_prompts: list[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    dummy_prompt_token_ids1 = np.random.randint(10000,
                                             size=(args.batch_size,
                                                   args.input_len))
    dummy_prompt_token_ids2 = np.random.randint(10000,
                                             size=(args.batch_size,
                                                   args.input_len + 1))
    
    dummy_prompts1: List[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids1.tolist()]
    
    dummy_prompts2: List[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids2.tolist()]

    def llm_generate(prompts):
        if not args.use_beam_search:
            llm.generate(prompts,
                         sampling_params=sampling_params,
                         use_tqdm=False)
        else:
            llm.beam_search(
                prompts,
                BeamSearchParams(
                    beam_width=args.n,
                    max_tokens=1,
                    ignore_eos=True,
                ),
            )

    def run_to_completion(prompts, profile_dir: Optional[str] = None):
        if profile_dir:
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        str(profile_dir)),
            ) as p:
                llm_generate(prompts)
            print(p.key_averages().table(sort_by="self_cuda_time_total"))
        else:
            start_time = time.perf_counter()
            llm_generate(prompts)
            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency

    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(dummy_prompts1, profile_dir=None)
        run_to_completion(dummy_prompts2, profile_dir=None)

    if args.profile:
        profile_dir = args.profile_result_dir
        if not profile_dir:
            profile_dir = (Path(".") / "vllm_benchmark_result" /
                           f"latency_result_{time.time()}")
        print(f"Profiling (results will be saved to '{profile_dir}')...")
        run_to_completion(dummy_prompts1, profile_dir=profile_dir)
        run_to_completion(dummy_prompts2, profile_dir=profile_dir)
        return

    # Benchmark
    latencies1 = []
    latencies2 = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies1.append(run_to_completion(dummy_prompts1, profile_dir=None))
        latencies2.append(run_to_completion(dummy_prompts2, profile_dir=None))
    
    latencies1 = np.array(latencies1)
    latencies2 = np.array(latencies2)
    
    # Calculate average latencies and ratio
    avg_latency1 = np.mean(latencies1)
    avg_latency2 = np.mean(latencies2)
    ratio = avg_latency2 / avg_latency1
    
    print(f"\nResults for batch_size={args.batch_size}, input_len={args.input_len}:")
    print(f"Avg latency for input_len={args.input_len}: {avg_latency1:.4f} seconds")
    print(f"Avg latency for input_len={args.input_len + 1}: {avg_latency2:.4f} seconds")
    print(f"Ratio (len+1/len): {ratio:.4f}")

    # Output JSON results if specified
    if args.output_json:
        results = {
            "batch_size": args.batch_size,
            "input_len": args.input_len,
            "avg_latency_len": avg_latency1,
            "avg_latency_len_plus_1": avg_latency2,
            "ratio": ratio,
            "latencies_len": latencies1.tolist(),
            "latencies_len_plus_1": latencies2.tolist(),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the latency of processing a single batch of "
        "requests till completion.")
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of generated sequences per prompt.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=10,
        help="Number of iterations to run for warmup.",
    )
    parser.add_argument("--num-iters",
                        type=int,
                        default=30,
                        help="Number of iterations to run.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="profile the generation process of a single batch",
    )
    parser.add_argument(
        "--profile-result-dir",
        type=str,
        default=None,
        help=("path to save the pytorch profiler output. Can be visualized "
              "with ui.perfetto.dev or Tensorboard."),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the latency results in JSON format.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=("Do not detokenize responses (i.e. do not include "
              "detokenization time in the latency measurement)"),
    )

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
