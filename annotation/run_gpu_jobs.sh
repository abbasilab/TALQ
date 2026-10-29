#!/usr/bin/env bash
# GPU jobs of the TALQ annotation with vLLM (run_vllm_annotation.py), for a machine with several
# GPUs (NGPU, default 8).  None of these jobs is needed for the results of the manuscript, which are
# computed from the stored model outputs (export_llm_outputs.py).
#
# Copy the annotation/ folder, including requests/ (written by steps 3 and 4) with a copy of
# config/symptoms_job_overrides.yaml (the symptom titles shown to the model), to the GPU machine, then:
#     pip install "vllm>=0.6" transformers bitsandbytes pandas pyarrow pyyaml
#     huggingface-cli login            # Llama and Gemma are gated models
#     bash run_gpu_jobs.sh <job>       # see the job names below
# Every job can be resumed: running it again skips the inputs that are already written.
# Copy each resulting outputs/<name>/ folder to data/llm_outputs/<name>/ of the repository.
#
# Every job uses --sampling annotation, the sampling settings of the stored outputs (temperature 0.6,
# top-p 0.9, top-k 50, at most 384 new tokens), for the other models as well.
set -euo pipefail
cd "$(dirname "$0")"
NGPU=${NGPU:-8}
mkdir -p outputs                  # the per-GPU logs outputs/<name>.gpu<i>.log are written here

shard8 () {   # run one model data-parallel over $NGPU GPUs: shard8 <model> <input> <out> <reps> [extra args]
  local model=$1 input=$2 out=$3 reps=$4; shift 4
  for ((g=0; g<NGPU; g++)); do    # process g handles shard g of the inputs on GPU g
    CUDA_VISIBLE_DEVICES=$g python run_vllm_annotation.py --model "$model" --input "$input" \
      --out "$out" --replicates "$reps" --shard $g --n-shards $NGPU "$@" > "$out.gpu$g.log" 2>&1 &
  done
  wait
}

case "${1:-help}" in
  # A. The cohort windows with Llama-3.1-8B-Instruct and the settings of the stored outputs: 4-bit NF4,
  #    annotation sampling, prompts beginning with two BOS tokens, 5 samples per window and symptom.
  #    An independent second annotation of the cohort, e.g. to measure agreement between runs.
  cohort)
    shard8 meta-llama/Llama-3.1-8B-Instruct requests/windows.parquet outputs/llama31_8b_vllm 5 \
      --quantization bnb4 --double-bos --sampling annotation ;;

  # B. Other models on the same 9,913 windows (2 samples each).
  qwen)    shard8 Qwen/Qwen2.5-7B-Instruct           requests/windows.parquet outputs/qwen25_7b  2 \
             --sampling annotation ;;
  gemma)   shard8 google/gemma-3-12b-it              requests/windows.parquet outputs/gemma3_12b 2 \
             --sampling annotation ;;
  mistral) shard8 mistralai/Mistral-7B-Instruct-v0.3 requests/windows.parquet outputs/mistral_7b 2 \
             --sampling annotation ;;
  llama70)
    # two tensor-parallel copies of 4 GPUs each
    for g in 0 1; do
      CUDA_VISIBLE_DEVICES=$((4*g)),$((4*g+1)),$((4*g+2)),$((4*g+3)) python run_vllm_annotation.py \
        --model meta-llama/Llama-3.3-70B-Instruct --tp 4 --input requests/windows.parquet \
        --out outputs/llama33_70b --replicates 2 --shard $g --n-shards 2 --sampling annotation \
        > outputs/llama33_70b.gpu$g.log 2>&1 &
    done; wait ;;

  # C. Every post of the human-annotated validation datasets, Llama-3.1-8B-Instruct with the settings of
  #    the stored outputs (2 samples).  Step 14 reads data/llm_outputs/<model>_val_<dataset>/.
  validation)
    for f in requests/validation_*.parquet; do ds=$(basename "$f" .parquet); ds=${ds#validation_}
      shard8 meta-llama/Llama-3.1-8B-Instruct requests/validation_$ds.parquet outputs/llama31_8b_val_$ds 2 \
        --quantization bnb4 --double-bos --sampling annotation
    done ;;
  # the same datasets with the other models of B
  validation_all_models)
    for m in Qwen/Qwen2.5-7B-Instruct google/gemma-3-12b-it mistralai/Mistral-7B-Instruct-v0.3; do
      tag=$(basename "$m" | tr 'A-Z.' 'a-z_')       # e.g. qwen2_5-7b-instruct
      for f in requests/validation_*.parquet; do ds=$(basename "$f" .parquet); ds=${ds#validation_}
        shard8 "$m" requests/validation_$ds.parquet outputs/${tag}_val_$ds 2 --sampling annotation
      done
    done ;;

  # quick check on one GPU: renders prompts and prints token-length statistics, generates nothing
  dryrun)  python run_vllm_annotation.py --model meta-llama/Llama-3.1-8B-Instruct \
             --input requests/windows.parquet --out outputs/_dry --dry-run ;;
  *) echo "usage: bash run_gpu_jobs.sh" \
          "{cohort|qwen|gemma|mistral|llama70|validation|validation_all_models|dryrun}";;
esac
