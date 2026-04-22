#!/usr/bin/env bash

set -euo pipefail

# Reviewer-facing diagnostic experiment:
#   per method x per load x per dataset acceptance behavior and time breakdown.
#
# The full matrix is summarized as CSV. Only a small set of representative
# Nightjar runs is sent to the trace plotting script so the generated traces are
# readable and directly usable in the response.

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B}"
DRAFT_MODEL_NAME="${DRAFT_MODEL_NAME:-/root/autodl-tmp/deep05b}"
HOST="${HOST:-127.0.0.1}"
PORT_BASE="${PORT_BASE:-8210}"

RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/benchmark_results/acceptance_behavior_diagnostics}"
RUN_LOG="${RUN_LOG:-$RESULT_DIR/acceptance_behavior_diagnostics.log}"
SUMMARY_CSV="${SUMMARY_CSV:-$RESULT_DIR/acceptance_behavior_summary.csv}"
RUN_RESULTS_CSV="${RUN_RESULTS_CSV:-$RESULT_DIR/benchmark_results_acceptance_behavior.csv}"
TRACE_MANIFEST="${TRACE_MANIFEST:-$RESULT_DIR/typical_trace_manifest.json}"
TRACE_ENTRIES_FILE="${TRACE_ENTRIES_FILE:-$RESULT_DIR/typical_trace_entries.jsonl}"
ANALYSIS_OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-$RESULT_DIR/figs}"
ANALYSIS_PREFIX="${ANALYSIS_PREFIX:-acceptance_behavior_typical}"

NUM_PROMPTS="${NUM_PROMPTS:-10}"
START_INDEX="${START_INDEX:-0}"
OUTPUT_LEN="${OUTPUT_LEN:-}"
MAX_SPECULATIVE_LEN="${MAX_SPECULATIVE_LEN:-5}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-4112}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SPECULATIVE_DRAFT_TP_SIZE="${SPECULATIVE_DRAFT_TP_SIZE:-1}"
BURSTINESS="${BURSTINESS:-1.0}"
ENABLE_TRACE="${ENABLE_TRACE:-True}"
SAVE_TRACE="${SAVE_TRACE:-False}"
SEED="${SEED:-42}"
EXPLORE="${EXPLORE:-False}"
FORCE_RERUN="${FORCE_RERUN:-True}"
ANALYZE_AT_END="${ANALYZE_AT_END:-True}"

LOW_RATE="${LOW_RATE:-5}"
MED_RATE="${MED_RATE:-10}"
HIGH_RATE="${HIGH_RATE:-20}"
LOADS="${LOADS:-med:$MED_RATE}"
#LOADS="${LOADS:-low:$LOW_RATE med:$MED_RATE high:$HIGH_RATE}"
DATASETS="${DATASETS:-sharegpt:/root/autodl-tmp/sharegpt.json}"
# DATASETS="${DATASETS:-sharegpt:/root/autodl-tmp/sharegpt.json alpaca:tatsu-lab/alpaca specbench:/root/autodl-tmp/nightjar/vllm_speculative/question_shuffled.jsonl}"

# Format: method_name:sub_strategy:select_strategy:speculative_len:elastic_memory:server_strategy
# Keep the default set short: no speculation, fixed SD, two adaptive baselines,
# and Nightjar.
# nospec:nospec::0:false:ilp
METHODS="${METHODS:-fixed_sd:fixed_draft::5:false:ilp dsd:daspec::5:false:ilp banditspec:ucb::5:false:ilp nightjar:ada_bin_greedy:capacity:5:true:ilp}"

# Representative traces to plot. The full CSV still covers every method/load/dataset.
# Format: dataset:load:method
TRACE_CASES="${TRACE_CASES:-sharegpt:med:fixed_sd sharegpt:med:dsd sharegpt:med:banditspec sharegpt:med:nightjar alpaca:med:nightjar specbench:high:nightjar}"

STATIC_INCREASE_BLOCK_THRESHOLD="${STATIC_INCREASE_BLOCK_THRESHOLD:-999999}"
STATIC_DECREASE_BLOCK_THRESHOLD="${STATIC_DECREASE_BLOCK_THRESHOLD:-999999}"
STATIC_PERSIST_STEPS="${STATIC_PERSIST_STEPS:-999999}"
ELASTIC_INCREASE_BLOCK_THRESHOLD="${ELASTIC_INCREASE_BLOCK_THRESHOLD:-2000}"
ELASTIC_DECREASE_BLOCK_THRESHOLD="${ELASTIC_DECREASE_BLOCK_THRESHOLD:-2000}"
ELASTIC_PERSIST_STEPS="${ELASTIC_PERSIST_STEPS:-3}"

mkdir -p "$RESULT_DIR" "$ANALYSIS_OUTPUT_DIR"
rm -f "$RUN_LOG" "$SUMMARY_CSV" "$RUN_RESULTS_CSV" "$TRACE_MANIFEST" "$TRACE_ENTRIES_FILE"

printf "%s\n" \
    "dataset,load,request_rate,method,sub_strategy,select_strategy,speculative_len,elastic_memory,completed,total_token_throughput,mean_ttft_ms,mean_e2el_ms,num_steps,num_speculative_steps,num_ar_steps,enabled_ratio,mean_acceptance_rate,median_acceptance_rate,mean_gamma_all_steps,mean_gamma_when_enabled,max_gamma,draft_time_ms,verify_time_ms,ar_time_ms,draft_share_pct,verify_share_pct,ar_share_pct,first_disabled_step,result_json,event_log" \
    > "$SUMMARY_CSV"

touch "$TRACE_ENTRIES_FILE"

require_dataset() {
    local dataset_name="$1"
    local dataset_path="$2"
    if [[ "$dataset_name" != "alpaca" && ! -f "$dataset_path" ]]; then
        echo "Missing dataset file for $dataset_name: $dataset_path" >&2
        exit 1
    fi
}

is_trace_case() {
    local dataset_name="$1"
    local load_name="$2"
    local method_name="$3"
    local wanted
    for wanted in $TRACE_CASES; do
        if [[ "$wanted" == "$dataset_name:$load_name:$method_name" ]]; then
            return 0
        fi
    done
    return 1
}

successful_existing_result() {
    local result_json="$1"
    [[ "$FORCE_RERUN" != "True" && "$FORCE_RERUN" != "true" && -f "$result_json" ]]
}

summarize_case() {
    local dataset_name="$1"
    local load_name="$2"
    local request_rate="$3"
    local method_name="$4"
    local sub_strategy="$5"
    local select_strategy="$6"
    local speculative_len="$7"
    local elastic_memory="$8"
    local result_json="$9"
    local event_log="${10}"

    python - "$SUMMARY_CSV" "$dataset_name" "$load_name" "$request_rate" \
        "$method_name" "$sub_strategy" "$select_strategy" "$speculative_len" \
        "$elastic_memory" "$result_json" "$event_log" <<'PY'
import csv
import json
import os
import statistics
import sys

(
    summary_csv,
    dataset,
    load_name,
    request_rate,
    method,
    sub_strategy,
    select_strategy,
    speculative_len,
    elastic_memory,
    result_json,
    event_log,
) = sys.argv[1:12]

completed = throughput = mean_ttft = mean_e2el = ""
benchmark_duration_s = 0.0
if result_json and os.path.exists(result_json):
    with open(result_json, "r", encoding="utf-8") as f:
        result = json.load(f)
    completed = result.get("completed", "")
    throughput = result.get("total_token_throughput", "")
    mean_ttft = result.get("mean_ttft_ms", "")
    mean_e2el = result.get("mean_e2el_ms", "")
    benchmark_duration_s = float(result.get("duration") or 0.0)

steps = []
if event_log and os.path.exists(event_log):
    with open(event_log, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event_type") == "speculative_step":
                steps.append(record)

gammas = [int(step.get("proposal_length_gamma", 0) or 0) for step in steps]
spec_steps = [step for step, gamma in zip(steps, gammas) if gamma > 0]
ar_steps = [step for step, gamma in zip(steps, gammas) if gamma == 0]
acceptances = [float(step.get("acceptance_rate", 0.0) or 0.0) for step in spec_steps]

draft_time = sum(float(step.get("draft_time_ms", 0.0) or 0.0) for step in spec_steps)
verify_time = sum(
    float(step.get("scoring_time_ms", 0.0) or 0.0)
    + float(step.get("verify_time_ms", 0.0) or 0.0)
    for step in spec_steps
)
ar_time = sum(float(step.get("step_total_time_ms", 0.0) or 0.0) for step in ar_steps)

# Pure no-speculation runs do not emit per-step speculative_step records in the
# current server path. Treat the benchmark wall-clock duration as AR-only time
# so the diagnostic table does not show an empty 0/0/0 breakdown.
if not steps and sub_strategy == "nospec" and benchmark_duration_s > 0:
    ar_time = benchmark_duration_s * 1000.0
total_time = draft_time + verify_time + ar_time

def mean(values):
    return statistics.fmean(values) if values else 0.0

def median(values):
    return statistics.median(values) if values else 0.0

def pct(part, whole):
    return part / whole * 100.0 if whole > 0 else 0.0

first_disabled_step = ""
for index, gamma in enumerate(gammas):
    if gamma == 0:
        first_disabled_step = index
        break

row = [
    dataset,
    load_name,
    request_rate,
    method,
    sub_strategy,
    select_strategy,
    speculative_len,
    elastic_memory,
    completed,
    throughput,
    mean_ttft,
    mean_e2el,
    len(steps),
    len(spec_steps),
    len(ar_steps),
    f"{(len(spec_steps) / len(steps)):.4f}" if steps else "0.0000",
    f"{mean(acceptances):.4f}",
    f"{median(acceptances):.4f}",
    f"{mean([float(g) for g in gammas]):.4f}",
    f"{mean([float(g) for g in gammas if g > 0]):.4f}",
    max(gammas) if gammas else 0,
    f"{draft_time:.4f}",
    f"{verify_time:.4f}",
    f"{ar_time:.4f}",
    f"{pct(draft_time, total_time):.2f}",
    f"{pct(verify_time, total_time):.2f}",
    f"{pct(ar_time, total_time):.2f}",
    first_disabled_step,
    result_json,
    event_log,
]

with open(summary_csv, "a", encoding="utf-8", newline="") as f:
    csv.writer(f).writerow(row)

print(
    f"{dataset}/{load_name}/{method}: throughput={throughput}, "
    f"acceptance={row[16]}, enabled_ratio={row[15]}, "
    f"draft/verify/ar={row[24]}/{row[25]}/{row[26]}%"
)
PY
}

append_trace_entry() {
    local dataset_name="$1"
    local load_name="$2"
    local request_rate="$3"
    local method_name="$4"
    local event_log="$5"
    local result_json="$6"

    python - "$TRACE_ENTRIES_FILE" "$event_log" "$dataset_name" "$load_name" \
        "$request_rate" "$method_name" "$result_json" <<'PY'
import json
import sys

entries_file, event_log, dataset, load_name, request_rate, method, result_json = sys.argv[1:8]
entry = {
    "path": event_log,
    "label": f"{method}_{dataset}_{load_name}",
    "dataset": dataset,
    "load": f"{load_name} ({request_rate} req/s)",
    "method": method,
    "request_rate": float(request_rate),
    "result_json": result_json,
}
with open(entries_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
PY
}

run_one() {
    local dataset_name="$1"
    local dataset_path="$2"
    local load_name="$3"
    local request_rate="$4"
    local method_name="$5"
    local sub_strategy="$6"
    local select_strategy="$7"
    local speculative_len="$8"
    local elastic_memory="$9"
    local server_strategy="${10}"
    local port="${11}"

    local case_dir="$RESULT_DIR/${dataset_name}/${load_name}/${method_name}"
    local event_log="$case_dir/nightjar_events.jsonl"
    local result_json="$case_dir/result.json"
    local case_log="$case_dir/run.log"
    mkdir -p "$case_dir"
    rm -f "$event_log" "$case_log"

    if successful_existing_result "$result_json"; then
        echo "[acceptance-diagnostics] reuse $dataset_name/$load_name/$method_name" >> "$RUN_LOG"
        summarize_case "$dataset_name" "$load_name" "$request_rate" "$method_name" \
            "$sub_strategy" "$select_strategy" "$speculative_len" "$elastic_memory" \
            "$result_json" "$event_log"
        if is_trace_case "$dataset_name" "$load_name" "$method_name"; then
            append_trace_entry "$dataset_name" "$load_name" "$request_rate" \
                "$method_name" "$event_log" "$result_json"
        fi
        return
    fi

    local increase_threshold="$STATIC_INCREASE_BLOCK_THRESHOLD"
    local decrease_threshold="$STATIC_DECREASE_BLOCK_THRESHOLD"
    local persist_steps="$STATIC_PERSIST_STEPS"
    if [[ "$elastic_memory" == "true" ]]; then
        increase_threshold="$ELASTIC_INCREASE_BLOCK_THRESHOLD"
        decrease_threshold="$ELASTIC_DECREASE_BLOCK_THRESHOLD"
        persist_steps="$ELASTIC_PERSIST_STEPS"
    fi

    local effective_spec_len="$speculative_len"
    if [[ "$effective_spec_len" == "0" ]]; then
        effective_spec_len="$MAX_SPECULATIVE_LEN"
    fi

    export NIGHTJAR_EVENT_LOG_PATH="$event_log"

    local before_file after_file produced_json
    before_file="$(mktemp)"
    after_file="$(mktemp)"
    find "$case_dir" -maxdepth 1 -type f -name 'benchmark_*.json' | sort > "$before_file"

    local cmd=(
        python "$ROOT_DIR/run_benchmark_tests.py"
        --strategy "$server_strategy"
        --sub-strategy "$sub_strategy"
        --explore "$EXPLORE"
        --save-trace "$SAVE_TRACE"
        --model "$MODEL_NAME"
        --draft-model "$DRAFT_MODEL_NAME"
        --host "$HOST"
        --port "$port"
        --dataset-name "$dataset_name"
        --dataset-path "$dataset_path"
        --speculative-len "$effective_spec_len"
        --num-prompts "$NUM_PROMPTS"
        --request-rates "$request_rate"
        --start-index "$START_INDEX"
        --num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE"
        --enable-trace "$ENABLE_TRACE"
        --burstiness "$BURSTINESS"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --max-model-len "$MAX_MODEL_LEN"
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
        --speculative-draft-tensor-parallel-size "$SPECULATIVE_DRAFT_TP_SIZE"
        --increase-block-threshold "$increase_threshold"
        --decrease-block-threshold "$decrease_threshold"
        --persist-steps "$persist_steps"
        --export-step-log "$event_log"
        --result-dir "$case_dir"
        --seed "$SEED"
    )

    if [[ -n "$OUTPUT_LEN" ]]; then
        cmd+=(--output-len "$OUTPUT_LEN")
    fi
    if [[ -n "$select_strategy" ]]; then
        cmd+=(--select-strategy "$select_strategy")
    fi

    {
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start dataset=$dataset_name load=$load_name rate=$request_rate method=$method_name sub_strategy=$sub_strategy elastic=$elastic_memory port=$port"
        echo "cmd=${cmd[*]}"
        BENCHMARK_RESULTS_CSV="$RUN_RESULTS_CSV" "${cmd[@]}"
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done dataset=$dataset_name load=$load_name method=$method_name"
    } >> "$case_log" 2>&1
    cat "$case_log" >> "$RUN_LOG"

    find "$case_dir" -maxdepth 1 -type f -name 'benchmark_*.json' | sort > "$after_file"
    produced_json="$(comm -13 "$before_file" "$after_file" | tail -n 1)"
    if [[ -z "$produced_json" ]]; then
        produced_json="$(find "$case_dir" -maxdepth 1 -type f -name 'benchmark_*.json' | sort | tail -n 1)"
    fi
    rm -f "$before_file" "$after_file"

    if [[ -n "$produced_json" && -f "$produced_json" ]]; then
        cp "$produced_json" "$result_json"
    else
        echo "No result JSON produced for $dataset_name/$load_name/$method_name" >&2
        exit 1
    fi

    summarize_case "$dataset_name" "$load_name" "$request_rate" "$method_name" \
        "$sub_strategy" "$select_strategy" "$speculative_len" "$elastic_memory" \
        "$result_json" "$event_log"

    if is_trace_case "$dataset_name" "$load_name" "$method_name"; then
        append_trace_entry "$dataset_name" "$load_name" "$request_rate" \
            "$method_name" "$event_log" "$result_json"
    fi
}

{
    echo "[acceptance-diagnostics] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[acceptance-diagnostics] MODEL_NAME=$MODEL_NAME"
    echo "[acceptance-diagnostics] DRAFT_MODEL_NAME=$DRAFT_MODEL_NAME"
    echo "[acceptance-diagnostics] DATASETS=$DATASETS"
    echo "[acceptance-diagnostics] LOADS=$LOADS"
    echo "[acceptance-diagnostics] METHODS=$METHODS"
    echo "[acceptance-diagnostics] TRACE_CASES=$TRACE_CASES"
} > "$RUN_LOG"

port_offset=0
for dataset in $DATASETS; do
    dataset_name="${dataset%%:*}"
    dataset_path="${dataset#*:}"
    require_dataset "$dataset_name" "$dataset_path"

    for load in $LOADS; do
        load_name="${load%%:*}"
        request_rate="${load#*:}"

        for method in $METHODS; do
            method_name="${method%%:*}"
            rest="${method#*:}"
            sub_strategy="${rest%%:*}"
            rest="${rest#*:}"
            select_strategy="${rest%%:*}"
            rest="${rest#*:}"
            speculative_len="${rest%%:*}"
            rest="${rest#*:}"
            elastic_memory="${rest%%:*}"
            server_strategy="${rest#*:}"
            if [[ "$server_strategy" == "$rest" ]]; then
                server_strategy="ilp"
            fi

            run_one "$dataset_name" "$dataset_path" "$load_name" "$request_rate" \
                "$method_name" "$sub_strategy" "$select_strategy" \
                "$speculative_len" "$elastic_memory" "$server_strategy" \
                "$((PORT_BASE + port_offset))"
            port_offset=$((port_offset + 1))
        done
    done
done

python - "$TRACE_ENTRIES_FILE" "$TRACE_MANIFEST" <<'PY'
import json
import sys

entries_file, manifest = sys.argv[1:3]
entries = []
with open(entries_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            entries.append(json.loads(line))
with open(manifest, "w", encoding="utf-8") as f:
    json.dump(entries, f, indent=2, ensure_ascii=False)
print(f"trace_entries={len(entries)}")
PY

if [[ "$ANALYZE_AT_END" == "True" || "$ANALYZE_AT_END" == "true" ]]; then
    python "$ROOT_DIR/exps/plot_dynamic_behavior_report.py" \
        --manifest "$TRACE_MANIFEST" \
        --output-dir "$ANALYSIS_OUTPUT_DIR" \
        --prefix "$ANALYSIS_PREFIX" \
        >> "$RUN_LOG" 2>&1
fi

echo "Acceptance behavior diagnostics finished."
echo "Summary CSV: $SUMMARY_CSV"
echo "Benchmark CSV: $RUN_RESULTS_CSV"
echo "Trace manifest: $TRACE_MANIFEST"
echo "Typical trace figures: $ANALYSIS_OUTPUT_DIR"
echo "Run log: $RUN_LOG"
