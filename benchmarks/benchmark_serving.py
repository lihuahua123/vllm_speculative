# SPDX-License-Identifier: Apache-2.0
r"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""
import argparse
import asyncio
import csv
import json
import gc
import json
import os
import random
import time
import urllib.request
import warnings
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from transformers import PreTrainedTokenizerFast
import numpy as np
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

from benchmark_dataset import (BurstGPTDataset, HuggingFaceDataset,
                               RandomDataset, SampleRequest, ShareGPTDataset,
                               SonnetDataset, VisionArenaDataset, HuggingFaceAlpacaDataset, SpecBenchDataset)
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


def configure_nightjar_logging_endpoint(base_url: str,
                                        export_step_log: Optional[str]) -> None:
    if not export_step_log:
        return
    payload = json.dumps({"path": export_step_log}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/nightjar_logging",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(
                f"Failed to configure nightjar logging: {response.status}")


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]



    
async def get_request(
    input_requests: list[SampleRequest] | list[list[SampleRequest]], 
    request_rate: float | list[float],
    burstiness: float = 1.0,
    enable_trace: bool = False,
) -> AsyncGenerator[SampleRequest, None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness.

    Args:
        input_requests:
            A list of input requests, each represented as a SampleRequest.
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness (optional):
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
    """
    if enable_trace:
        assert len(input_requests) == len(request_rate), f"input_requests: {len(input_requests)} != request_rate:  {len(request_rate)}"
        for index, input_requst in enumerate(input_requests):
            begin_time = time.time()
            theta = 1.0 / (request_rate[index] * burstiness)
            for request in input_requst:
                yield request
                # 按照QPS=1的速率生成请求间隔
                if request_rate == float("inf"):
                    # If the request rate is infinity, then we don't need to wait.
                    continue
                interval = np.random.gamma(shape=burstiness, scale=theta)
                await asyncio.sleep(interval)
            end_time = time.time()
            print(f"generate {index} request {len(input_requst)},qps {request_rate[index]}, time cost: {end_time - begin_time}")
        
    else:
        input_requests: Iterable[SampleRequest] = iter(input_requests)

        # Calculate scale parameter theta to maintain the desired request_rate.
        assert burstiness > 0, (
            f"A positive burstiness factor is expected, but given {burstiness}.")
        theta = 1.0 / (request_rate * burstiness)

        for request in input_requests:
            yield request

            if request_rate == float("inf"):
                # If the request rate is infinity, then we don't need to wait.
                continue

            # Sample the request interval from the gamma distribution.
            # If burstiness is 1, it follows exponential distribution.
            interval = np.random.gamma(shape=burstiness, scale=theta)
            # The next request will be sent after the interval.
            await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
) -> tuple[BenchmarkMetrics, list[int]]:
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    reqs_len = len(outputs)
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if output_len is None or output_len == 0:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                output_len = len(
                    tokenizer(outputs[i].generated_text,
                              add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(goodput_config_dict["ttft"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(goodput_config_dict["tpot"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(goodput_config_dict["e2el"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / reqs_len,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000)
                            for p in selected_percentiles],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000)
                             for p in selected_percentiles],
    )

    return metrics, actual_output_lens



async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: list[SampleRequest],
    logprobs: Optional[int],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    goodput_config_dict: dict[str, float],
    max_concurrency: Optional[int],
    lora_modules: Optional[Iterable[str]],
    enable_trace: bool = False,
    start_index: int = 0,
    strategy_name: Optional[str] = None,
    increase_block_threshold: int = 150,
    decrease_block_threshold: int = 100,
    config_output_len: Optional[int] = None,
    export_step_log: Optional[str] = None,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len, test_mm_content = \
        input_requests[0].prompt, input_requests[0].prompt_len, \
        input_requests[0].expected_output_len, \
            input_requests[0].multi_modal_data

    if backend != "openai-chat" and test_mm_content is not None:
        # multi-modal benchmark is only available on OpenAI Chat backend.
        raise ValueError(
            "Multi-modal content is only supported on 'openai-chat' backend.")
    assert test_mm_content is None or isinstance(test_mm_content, dict)
    test_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        multi_modal_content=test_mm_content,
        ignore_eos=ignore_eos,
    )

    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")

    if lora_modules:
        # For each input request, choose a LoRA module at random.
        lora_modules = iter(
            [random.choice(lora_modules) \
                for _ in range(len(input_requests))])

    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(model=model_id,
                                         model_name=model_name,
                                         prompt=test_prompt,
                                         api_url=base_url + "/start_profile",
                                         prompt_len=test_prompt_len,
                                         output_len=test_output_len,
                                         logprobs=logprobs,
                                         multi_modal_content=test_mm_content,
                                         ignore_eos=ignore_eos)
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started")

    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = (asyncio.Semaphore(max_concurrency)
                 if max_concurrency else None)

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input,
                                      pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input,
                                      pbar=pbar)

   
    
    # start_index = 3 00 # 前300 用来profile了
    # input_requests_list = [input_requests[start_index:start_index+18],input_requests[start_index+18:start_index+20],input_requests[start_index+20:start_index+40],input_requests[start_index+40:]]
    # request_rate_list = [1,0.1,1,0.1]
    outputs_list = []
    request_rate_list =  []#
    input_requests_list = []#[input_requests[start_index:start_index+20],input_requests[start_index+20:start_index+40],input_requests[start_index+40:start_index+540]] # [input_requests] #[input_requests[start_index:start_index+20],input_requests[start_index+20:start_index+40],input_requests[start_index+40:]]
    print(f"enable_trace: {enable_trace}")
    if enable_trace:
        # request_rate_list = np.load('./azureqps.npy') #[1,5,1,10,15,2]
        # for req in request_rate_list:
        #     input_requests_list.append(input_requests[start_index:start_index+req])
        #     start_index += req
        # 这是ok的动态
        request_rate_list = [
            20,
            20,
            20
            #100,
            #20
            ]
        input_requests_list = [
            input_requests[start_index:start_index+20],
            input_requests[start_index+20:start_index+40],
            input_requests[start_index+40:start_index+240],
            #input_requests[start_index+150:start_index+350]
        ]
       
        # # 最后一段请求的每条 output_len 加 200
        # if input_requests_list:
        #     for req in input_requests_list[-1]:
        #         req.expected_output_len += 200

        # request_rate_list = [5,25]
        # input_requests_list = [input_requests[start_index:start_index+100],input_requests[start_index+100:start_index+300]]
        # # request_rate_list = [5,25]
        # input_requests_list = [input_requests[start_index:start_index+10],input_requests[start_index+10:start_index+100]]
 
        # input_requests_list = [input_requests[start_index+120:]]
        # request_rate_list = [2,5,5,25]
        # input_requests_list = [input_requests[start_index:start_index+20],
        #                        input_requests[start_index+20:start_index+120],
        #                        input_requests[start_index+120:start_index+220],
        #                        input_requests[start_index+220:start_index+320]]
        # # request_rate_list = [1,1,2,1,5,10,25,1,1]
        #input_requests_list = [input_requests[start_index:start_index+20],input_requests[start_index+20:start_index+120],input_requests[start_index+120:start_index+320]]
        # for req in request_rate_list[:6]:
        #     input_requests_list.append(input_requests[start_index:start_index+20])
        #     start_index += 20
        # input_requests_list.append(input_requests[start_index:start_index+200])
        # start_index += 200
        # input_requests_list.append(input_requests[start_index:start_index+5])
        # start_index += 5
        # input_requests_list.append(input_requests[start_index:start_index+10])
        # assert len(input_requests_list) == len(request_rate_list)
        print(f"request_rate_list: {request_rate_list}")
    else:
        request_rate_list = [request_rate]
        input_requests_list = [input_requests[start_index:]]
    # 汇总：每个请求的 output_len（expected_output_len）与 ignore_eos
    all_requests_flat = [r for batch in input_requests_list for r in batch]
    output_lens = [r.expected_output_len for r in all_requests_flat]
    print(f"request output_lens (expected_output_len): {output_lens}")
    print(f"ignore_eos: {ignore_eos}")
    # 每条请求的真实 prompt 长度（用当前 tokenizer 计算，与服务端 tokenize 结果一致）
    actual_prompt_lens = []
    for r in all_requests_flat:
        # print(f"r.prompt: {r.prompt}")
        if isinstance(r.prompt, dict) and "prompt_token_ids" in r.prompt:
            actual_prompt_lens.append(len(r.prompt["prompt_token_ids"]))
        else:
            actual_prompt_lens.append(
                len(tokenizer(str(r.prompt), add_special_tokens=False).input_ids))
    print(f"actual prompt len per request (tokenized): {actual_prompt_lens}")
    benchmark_start_time = time.perf_counter()
    begin_time = time.time()
    for index, one_input_requests in enumerate(input_requests_list):
        tasks: list[asyncio.Task] = []
        async for request in get_request(one_input_requests, request_rate_list[index], burstiness, enable_trace=False):
            prompt, prompt_len, output_len, mm_content = request.prompt, \
                request.prompt_len, request.expected_output_len, \
                    request.multi_modal_data
            req_model_id, req_model_name = model_id, model_name
            if lora_modules:
                req_lora_module = next(lora_modules)
                req_model_id, req_model_name = req_lora_module, req_lora_module

            request_func_input = RequestFuncInput(model=req_model_id,
                                                model_name=req_model_name,
                                                prompt=prompt,
                                                api_url=api_url,
                                                prompt_len=prompt_len,
                                                output_len=output_len,
                                                logprobs=logprobs,
                                                multi_modal_content=mm_content,
                                                ignore_eos=ignore_eos)
            tasks.append(
                asyncio.create_task(
                    limited_request_func(request_func_input=request_func_input,
                                        pbar=pbar)))
        end_time = time.time()
        #print(f"send request time cost: {end_time - begin_time}")
        begin_time = time.time()
        outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)
        outputs_list  += outputs
        end_time = time.time()
    #print(f"receive response time cost: {end_time - begin_time}")
    texts = []
    for output in outputs_list:
        # print(f"output.generated_text: {output.generated_text}")
        texts.append(output.generated_text)
    
    with open("texts2.json", "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    
    if profile:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs_list,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
    )

    # 每个请求：实际输出长度 - 预期 output_lens，以数组形式打印
    output_len_diffs = [
        actual - expected
        for actual, expected in zip(actual_output_lens, output_lens)
    ]
    print(f"output_len diff (actual - expected) per request: {output_len_diffs}")

    # Expected total output tokens if every request generated exactly its max_tokens
    expected_total_output_tokens = sum(
        r.expected_output_len for one in input_requests_list for r in one)

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10}".format("Expected total output tokens:",
                                 expected_total_output_tokens))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    if goodput_config_dict:
        print("{:<40} {:<10.2f}".format("SLO Attainment (%):",
                                        metrics.request_goodput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):",
                                    metrics.total_token_throughput))

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "expected_total_output_tokens": expected_total_output_tokens,
        "config_output_len": config_output_len,
        "request_throughput": metrics.request_throughput,
        "request_goodput:":
        metrics.request_goodput if goodput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms")))
        print("{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms")))
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics,
                                f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                            value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT",
                       "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    # Export results to CSV
    export_to_csv(metrics, result, benchmark_duration, model_id, request_rate, 
                  burstiness, goodput_config_dict, selected_percentile_metrics,
                  strategy_name=strategy_name,
                  increase_block_threshold=increase_block_threshold,
                  decrease_block_threshold=decrease_block_threshold)

    return result


def export_to_csv(
    metrics: BenchmarkMetrics,
    result: dict[str, Any],
    benchmark_duration: float,
    model_id: str,
    request_rate: float,
    burstiness: float,
    goodput_config_dict: dict[str, float],
    selected_percentile_metrics: list[str],
    strategy_name: Optional[str] = None,
    csv_file: str = "benchmark_results.csv",
    increase_block_threshold: int = 150,
    decrease_block_threshold: int = 100,
):
    """Export benchmark results to CSV file. Append if file exists, create if not."""
    # Prepare CSV row data（含本次配置的 output_len，便于在 benchmark_results.csv 中区分实验）
    row_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": strategy_name if strategy_name else "",
        "model_id": model_id,
        "request_rate": request_rate if request_rate != float("inf") else "inf",
        "burstiness": burstiness,
        "benchmark_duration_s": f"{benchmark_duration:.2f}",
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "expected_total_output_tokens": result.get("expected_total_output_tokens", ""),
        "output_len": result.get("config_output_len", ""),
        "request_throughput_req_per_s": f"{metrics.request_throughput:.2f}",
        "output_throughput_tok_per_s": f"{metrics.output_throughput:.2f}",
        "total_token_throughput_tok_per_s": f"{metrics.total_token_throughput:.2f}",
        "increase_block_threshold": increase_block_threshold,
        "decrease_block_threshold": decrease_block_threshold,
    }
    
    # Add goodput if available
    if goodput_config_dict:
        row_data["request_goodput"] = f"{metrics.request_goodput:.2f}"
    else:
        row_data["request_goodput"] = ""
    
    # Collect all percentile field names dynamically
    percentile_fields = set()
    
    # Add metric statistics
    metric_fields = ["ttft", "tpot", "itl", "e2el"]
    for metric in metric_fields:
        if metric in selected_percentile_metrics:
            row_data[f"mean_{metric}_ms"] = f"{getattr(metrics, f'mean_{metric}_ms'):.2f}"
            row_data[f"median_{metric}_ms"] = f"{getattr(metrics, f'median_{metric}_ms'):.2f}"
            row_data[f"std_{metric}_ms"] = f"{getattr(metrics, f'std_{metric}_ms'):.2f}"
            # Add percentiles dynamically
            for p, value in getattr(metrics, f"percentiles_{metric}_ms"):
                p_word = str(int(p)) if int(p) == p else str(p)
                field_name = f"p{p_word}_{metric}_ms"
                row_data[field_name] = f"{value:.2f}"
                percentile_fields.add(field_name)
        else:
            # If metric not selected, fill with empty values
            row_data[f"mean_{metric}_ms"] = ""
            row_data[f"median_{metric}_ms"] = ""
            row_data[f"std_{metric}_ms"] = ""
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_file)
    
    # Get existing fieldnames from CSV if file exists
    existing_fieldnames = []
    if file_exists:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_fieldnames = next(reader, [])
        except Exception:
            existing_fieldnames = []
    
    # Build fieldnames list
    base_fieldnames = [
        "timestamp", "strategy", "model_id", "request_rate", "burstiness", 
        "benchmark_duration_s", "completed", "total_input_tokens", 
        "total_output_tokens", "expected_total_output_tokens", "output_len",
        "request_throughput_req_per_s",
        "request_goodput", "output_throughput_tok_per_s", 
        "total_token_throughput_tok_per_s",
        "increase_block_threshold", "decrease_block_threshold",
    ]
    
    # Add metric fields in order
    metric_fieldnames = []
    for metric in metric_fields:
        metric_fieldnames.extend([
            f"mean_{metric}_ms", f"median_{metric}_ms", f"std_{metric}_ms"
        ])
        # Add percentile fields for this metric, sorted by percentile value
        metric_percentile_fields = [f for f in percentile_fields if f.endswith(f"_{metric}_ms")]
        # Sort by extracting percentile number
        def extract_percentile(field_name):
            # Extract number from "p{number}_{metric}_ms"
            try:
                return int(field_name.split('_')[0][1:])
            except:
                return 0
        metric_percentile_fields.sort(key=extract_percentile)
        metric_fieldnames.extend(metric_percentile_fields)
    
    # Combine all fieldnames
    all_fieldnames = base_fieldnames + metric_fieldnames
    
    # If file exists, use existing fieldnames and add any new ones
    if existing_fieldnames:
        # Add any new fields that don't exist
        for field in all_fieldnames:
            if field not in existing_fieldnames:
                existing_fieldnames.append(field)
        fieldnames = existing_fieldnames
    else:
        fieldnames = all_fieldnames
    
    # Ensure all row_data keys are in fieldnames
    for key in row_data.keys():
        if key not in fieldnames:
            fieldnames.append(key)
    
    # Write to CSV
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write data row (only include fields that exist in fieldnames)
        filtered_row = {k: v for k, v in row_data.items() if k in fieldnames}
        # Fill missing fields with empty string
        for field in fieldnames:
            if field not in filtered_row:
                filtered_row[field] = ""
        writer.writerow(filtered_row)
    
    print(f"Benchmark results exported to {csv_file}")


def check_goodput_args(args):
    # Check and parse goodput arguments
    goodput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        goodput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in goodput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. ")
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative.")
    return goodput_config_dict


def parse_goodput(slo_pairs):
    goodput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            "Specify service level objectives for goodput as \"KEY:VALUE\" "
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds.") from err
    return goodput_config_dict


def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: dict[str, Any],
                                     file_name: str) -> None:
    metrics = [
        "median_ttft_ms", "mean_ttft_ms", "std_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "std_tpot_ms", "p99_tpot_ms",
        "median_itl_ms", "mean_itl_ms", "std_itl_ms", "p99_itl_ms"
    ]
    # These raw data might be useful, but they are rather big. They can be added
    # later if needed
    ignored_metrics = ["ttfts", "itls", "generated_texts", "errors"]
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={k: [results[k]]
                 for k in metrics},
        extra_info={
            k: results[k]
            for k in results if k not in metrics and k not in ignored_metrics
        })
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(file_name)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer_mode = args.tokenizer_mode

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    tokenizer = get_tokenizer(tokenizer_id,
                              tokenizer_mode=tokenizer_mode,
                              trust_remote_code=args.trust_remote_code)

    if args.dataset_name is None:
        raise ValueError(
            "Please specify '--dataset-name' and the corresponding "
            "'--dataset-path' if required.")

    if args.dataset_name == "sonnet":
        dataset = SonnetDataset(dataset_path=args.dataset_path)
        # For the "sonnet" dataset, formatting depends on the backend.
        if args.backend == "openai-chat":
            input_requests = dataset.sample(num_requests=args.num_prompts,
                                            input_len=args.sonnet_input_len,
                                            output_len=args.sonnet_output_len,
                                            prefix_len=args.sonnet_prefix_len,
                                            tokenizer=tokenizer,
                                            return_prompt_formatted=False)
        else:
            assert tokenizer.chat_template or tokenizer.default_chat_template, (
                "Tokenizer/model must have chat template for sonnet dataset.")
            input_requests = dataset.sample(num_requests=args.num_prompts,
                                            input_len=args.sonnet_input_len,
                                            output_len=args.sonnet_output_len,
                                            prefix_len=args.sonnet_prefix_len,
                                            tokenizer=tokenizer,
                                            return_prompt_formatted=True)

    elif args.dataset_name == "hf":
        # Choose between VisionArenaDataset
        # and HuggingFaceDataset based on provided parameters.
        dataset_class = (VisionArenaDataset if args.dataset_path
                         == VisionArenaDataset.VISION_ARENA_DATASET_PATH
                         and args.hf_subset is None else HuggingFaceDataset)
        input_requests = dataset_class(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
        ).sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            random_seed=args.seed,
            output_len=args.hf_output_len,
        )

    else:
        # For datasets that follow a similar structure, use a mapping.
        # sharegpt: 若未传 --sharegpt-output-len，则使用 --hf-output-len，便于 run_benchmark_tests 只传 --output-len/--hf-output-len 时生效
        sharegpt_output_len = (args.sharegpt_output_len
                               if args.sharegpt_output_len is not None
                               else args.hf_output_len)
        dataset_mapping = {
            "sharegpt":
            lambda: ShareGPTDataset(random_seed=args.seed,
                                    dataset_path=args.dataset_path).sample(
                                        tokenizer=tokenizer,
                                        num_requests=args.num_prompts,
                                        output_len=sharegpt_output_len,
                                    ),
            "burstgpt":
            lambda: BurstGPTDataset(random_seed=args.seed,
                                    dataset_path=args.dataset_path).
            sample(tokenizer=tokenizer, num_requests=args.num_prompts),
            "alpaca":
            lambda: HuggingFaceAlpacaDataset(
                dataset_path=args.dataset_path,
                dataset_split="train",
                random_seed=args.seed,
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                output_len=args.hf_output_len,
            ),
            "specbench":
            lambda: SpecBenchDataset(
                dataset_path=args.dataset_path,
                random_seed=args.seed,
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                output_len=args.hf_output_len,
            ),
            "random":
            lambda: RandomDataset(dataset_path=args.dataset_path).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.random_prefix_len,
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                range_ratio=args.random_range_ratio,
            )
        }

        try:
            input_requests = dataset_mapping[args.dataset_name]()
        except KeyError as err:
            raise ValueError(f"Unknown dataset: {args.dataset_name}") from err

    # 本次 benchmark 使用的输出长度配置（用于写入 CSV 等）
    config_output_len = None
    if args.dataset_name == "sharegpt":
        config_output_len = sharegpt_output_len
    elif args.dataset_name in ("hf", "alpaca", "specbench"):
        config_output_len = args.hf_output_len
    elif args.dataset_name == "random":
        config_output_len = args.random_output_len
    elif args.dataset_name == "sonnet":
        config_output_len = args.sonnet_output_len
    
    goodput_config_dict = check_goodput_args(args)

    # Parse strategy name
    strategy_name = args.strategy_name
    if strategy_name is None and args.result_filename:
        # Try to parse strategy name from result_filename
        # Format: sub_strategy_speculative_len.json or similar
        filename_base = os.path.splitext(os.path.basename(args.result_filename))[0]
        # Common patterns: "epsilon_greedy_3", "deep_1", "ucb_3", "threshold_2", "nospec_1"
        parts = filename_base.split('_')
        if len(parts) >= 2:
            sub_strategy = parts[0]
            # Map sub_strategy to friendly names
            strategy_mapping = {
                "epsilon_greedy": "Nightjar",
                "ucb": "ucb",
                "threshold": "threshold",
                "nospec": "nospec",
                "smart_spec": "smart_spec",
                "daspec": "daspec",
                "ngram": "ngram",
            }
            if sub_strategy == "deep":
                # For deep, check if there's a number (speculative_len)
                if len(parts) >= 2 and parts[1].isdigit():
                    strategy_name = f"deep-{parts[1]}"
                else:
                    strategy_name = "deep"
            elif sub_strategy in strategy_mapping:
                strategy_name = strategy_mapping[sub_strategy]
            else:
                strategy_name = sub_strategy

    # Avoid GC processing "static" data - reduce pause times.
    gc.collect()
    gc.freeze()

    if args.export_step_log:
        configure_nightjar_logging_endpoint(base_url, args.export_step_log)

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            model_name=model_name,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=args.logprobs,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[
                float(p) for p in args.metric_percentiles.split(",")
            ],
            ignore_eos=args.ignore_eos,
            goodput_config_dict=goodput_config_dict,
            max_concurrency=args.max_concurrency,
            lora_modules=args.lora_modules,
            enable_trace=args.enable_trace,
            start_index=args.start_index,
            strategy_name=strategy_name,
            increase_block_threshold=args.increase_block_threshold,
            decrease_block_threshold=args.decrease_block_threshold,
            config_output_len=config_output_len,
            export_step_log=args.export_step_log,
        ))

    # Save config and results to json
    if args.save_result:
        result_json: dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )
        print(args.save_detailed)
        args.save_detailed = False
        if not args.save_detailed:
            # Remove fields with too many data points
            for field in [
                    "input_lens", "output_lens", "ttfts", "itls",
                    "generated_texts", "errors"
            ]:
                if field in result_json:
                    del result_json[field]

        # Traffic
        result_json["request_rate"] = (args.request_rate if args.request_rate
                                       < float("inf") else "inf")
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        max_concurrency_str = (f"-concurrency{args.max_concurrency}"
                               if args.max_concurrency is not None else "")
        file_name = f"{backend}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  #noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "a", encoding='utf-8') as outfile:
            json.dump(result_json, outfile)
        save_to_pytorch_benchmark_format(args, result_json, file_name)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    # Use 127.0.0.1 here instead of localhost to force the use of ipv4
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "burstgpt", "sonnet", "random", "hf", "alpaca","specbench"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the sharegpt/sonnet dataset. "
                        "Or the huggingface dataset ID if using HF dataset.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=("Number of logprobs-per-token to compute & return as part of "
              "the request. If unspecified, then either (1) if beam search "
              "is disabled, no logprobs are computed & a single dummy "
              "logprob is returned for each token; or (2) if beam search "
              "is enabled 1 logprob per token is computed"),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="When saving the results, whether to include per request "
        "information such as response, error, ttfs, tpots, etc.",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--strategy-name",
        type=str,
        default=None,
        help="Strategy name to be recorded in CSV (e.g., Nightjar, ucb, threshold, deep-1, nospec, etc.). "
        "If not specified, will try to parse from result-filename.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.")
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl,e2el",
        help="Comma-seperated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        "Allowed metric names are \"ttft\", \"tpot\", \"itl\", \"e2el\". "
        "Default value is \"ttft,tpot,itl\".")
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-seperated list of percentiles for selected metrics. "
        "To report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "Default value is \"99\". "
        "Use \"--percentile-metrics\" to select metrics.",
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help="Specify service level objectives for goodput as \"KEY:VALUE\" "
        "pairs, where the key is a metric name, and the value is in "
        "milliseconds. Multiple \"KEY:VALUE\" pairs can be provided, "
        "separated by spaces. Allowed request level metric names are "
        "\"ttft\", \"tpot\", \"e2el\". For more context on the definition of "
        "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve")

    # group for dataset specific arguments
    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help=
        "Number of input tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help=
        "Number of output tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help=
        "Number of prefix tokens per request, used only for sonnet dataset.",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.")
    

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help=
        "Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help=
        "Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random sampling.",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before random "
        " context. The length range of context in a random "
        " request is [random-prefix-len, "
        " random-prefix-len + random-prefix-len * random-range-ratio).")

    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument("--hf-subset",
                          type=str,
                          default=None,
                          help="Subset of the HF dataset.")
    hf_group.add_argument("--hf-split",
                          type=str,
                          default=None,
                          help="Split of the HF dataset.")
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset.",
    )

    parser.add_argument(
        '--tokenizer-mode',
        type=str,
        default="auto",
        choices=['auto', 'slow', 'mistral', 'custom'],
        help='The tokenizer mode.\n\n* "auto" will use the '
        'fast tokenizer if available.\n* "slow" will '
        'always use the slow tokenizer. \n* '
        '"mistral" will always use the `mistral_common` tokenizer. \n*'
        '"custom" will use --tokenizer to select the preregistered tokenizer.')

    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. "
                        "If not specified, the model name will be the "
                        "same as the ``--model`` argument. ")

    parser.add_argument("--lora-modules",
                        nargs='+',
                        default=None,
                        help="A subset of LoRA module names passed in when "
                        "launching the server. For each request, the "
                        "script chooses a LoRA module at random.")
    parser.add_argument("--enable-trace",
                        action="store_true",
                        help="Enable trace mode for the benchmark.")
    parser.add_argument("--start-index",
                        type=int,
                        default=0,
                        help="Start index for the benchmark dataset.")
    parser.add_argument("--increase-block-threshold",
                        type=int,
                        default=150,
                        help="Threshold for free GPU blocks to trigger block number increase.")
    parser.add_argument("--decrease-block-threshold",
                        type=int,
                        default=100,
                        help="Threshold offset for free GPU blocks to trigger block number decrease.")
    parser.add_argument("--export-step-log",
                        type=str,
                        default=None,
                        help="Optional JSONL path on the server host for "
                        "Nightjar step/event logs.")

    args = parser.parse_args()

    main(args)
