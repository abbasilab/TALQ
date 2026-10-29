# Annotation tools

The analysis (`scripts/`) starts from the stored outputs of two annotation jobs run with
Llama-3.1-8B-Instruct. The tools in this folder export those outputs from the job database and
annotate inputs again with vLLM on GPUs. Neither is needed to run the analysis once the exported
outputs are in `data/llm_outputs/`.

| File | Purpose |
|---|---|
| `export_llm_outputs.py` | export the stored outputs of annotation jobs from MongoDB to Parquet |
| `run_vllm_annotation.py` | run the TALQ questionnaire with vLLM (batched, prefix caching, resumable) |
| `run_gpu_jobs.sh` | run `run_vllm_annotation.py` data-parallel over several GPUs |
| `requests/` | inputs of `run_vllm_annotation.py`, written by steps 3 and 4 (they hold post text) |

## Export the stored model outputs

Run `export_llm_outputs.py` on a machine that can reach the MongoDB server of the job queue
(`pip install pymongo pandas pyarrow`). Connection settings and credentials are read only from
environment variables:

| Variable | Meaning |
|---|---|
| `MONGO_HOST`, `MONGO_PORT` | server (default `localhost`, `27017`) |
| `MONGO_USER`, `MONGO_PASS` | credentials |
| `MONGO_TLS_CA_FILE`, `MONGO_TLS_CLIENT_FILE`, `MONGO_TLS_CLIENT_FILE_PASS` | TLS files (optional) |
| `MONGO_DB` | database of the job queue (default `redaf_langjobs`; or `--db`) |

```bash
python3 export_llm_outputs.py --list            # jobs in the database
python3 export_llm_outputs.py --out llm_export \
    --job-id 5976943c-4594-46b5-a71a-f1809cf1bfb3 \
    --job-id 7f390e2d-314e-4023-b3af-0800aa9ecc1a
```

| Job | Content | Used by |
|---|---|---|
| `5976943c-4594-46b5-a71a-f1809cf1bfb3` (`P.SYMPTOM_JOB`) | the 52-symptom battery, five samples per window and symptom (about 2.55 million stored responses, the bot's included) | step 5 |
| `7f390e2d-314e-4023-b3af-0800aa9ecc1a` (`P.INVENTORY_JOB`) | BDI-II and PHQ-9 answered by the same model, five samples per item (1,479,190 stored responses) | step 18 |

For each job the script writes `llm_export/<job_id>/part-*.parquet` (one row per stored request; the
model's `response` verbatim, without the large fields `posts`, `posts_text` and `prompt`, which steps
1–3 rebuild from the corpus), `sample_full_docs.jsonl` (300 complete documents, for checks of the rebuilt
windows), `collections.json` (collection names, sizes, fields, request status) and a `manifest.json`
with the SHA-256 of every part. Copy `llm_export/` to `data/llm_outputs/`, where the analysis looks for
it (`P.LLM_EXPORT`).

The inventory job stored the same placeholder `speaker_id` in every document. The export therefore
adds, by default, `posts_speaker_id` (author of the window's first post) and `n_posts`. The export
analyzed in the manuscript does not have these columns, and step 18 links windows to their authors by
the window start (see `scripts/18_inventories.py`).

## Annotation with vLLM

`run_vllm_annotation.py` sends every input (a two-week window, or one post of a validation dataset)
with every symptom question to the model, with the prompts of the stored outputs, and writes the raw
answers to `outputs/<name>/part-*.parquet`. Its inputs in `requests/` are written by the numbered
scripts:

| File | Rows | Content |
|---|---|---|
| `windows.parquet` | 9,913 | windows of the 492 annotated accounts (491 cohort members and the bot AutoModerator); 52 symptoms × 5 samples give 2,577,380 requests (step 3) |
| `validation_irf.parquet` | 3,522 | interpersonal risk factors: thwarted belongingness, perceived burdensomeness (step 4) |
| `validation_lost.parquet` | 3,250 | LoST: low self-esteem (step 4) |
| `validation_severity.parquet` | 3,553 | depression severity on four levels (step 4) |
| `validation_depemo.parquet` | 6,037 | DepressionEmo, weak labels; not reported in the manuscript (step 4) |
| `symptoms.yaml`, `task_symptom.yaml` | | the battery and the prompt templates (copies of `config/`, step 3) |

Settings that make the prompts and sampling those of the stored outputs:

- **Sampling.** `--sampling annotation` (the default) uses the default generation configuration of the
  Llama-3.1-8B-Instruct checkpoint, which generated the stored outputs: temperature 0.6, top-p 0.9,
  top-k 50, no repetition penalty, at most 384 new tokens (Methods). `config/llm_local.yaml` lists
  temperature 0.1, top-k 30 and a repetition penalty of 1.03, but these values were not applied to
  generation (see `src/slm/params.py`); `--sampling configured` uses them.
- **Two BOS tokens.** The chat template starts the rendered text with the beginning-of-sequence token,
  and the annotation job tokenized that text with special tokens, which adds a second one.
  `--double-bos` gives the same token sequence.
- **Symptom titles.** One title shown to the model differs from `config/symptoms.yaml`
  (`Indiciciveness` for indecisiveness, `config/symptoms_job_overrides.yaml`). The runner applies the
  titles of that file (`--symptom-overrides`; it looks in `requests/` and then in `../config/`). When
  only the `annotation/` folder is copied to a GPU machine, copy that file into `requests/` as well.
- **Quantization.** `--quantization bnb4` loads the weights in 4-bit NF4 with bitsandbytes.
- **Context length.** The longest window has about 66,000 tokens and 0.4% of the windows exceed 16,000.
  Prompts longer than a model's context are skipped and recorded (`finish_reason`
  `skipped_too_long`); this is why Gemma-3 12B (128,000 tokens) is used rather than Gemma-2 (8,000).

```bash
pip install "vllm>=0.6" transformers bitsandbytes pandas pyarrow pyyaml
huggingface-cli login                 # Llama and Gemma are gated models
bash run_gpu_jobs.sh dryrun           # renders prompts and prints token-length statistics
bash run_gpu_jobs.sh <job>
```

| Job | What it runs | Rough time on 8 × A100-80GB |
|---|---|---|
| `cohort` | the cohort windows with Llama-3.1-8B-Instruct and the settings of the stored outputs (NF4, `--sampling annotation`, `--double-bos`, 5 samples) | 8–12 h |
| `validation` | every post of the validation datasets with the same settings, 2 samples | 3–6 h |
| `qwen`, `gemma`, `mistral` | the cohort windows with Qwen2.5-7B, Gemma-3-12B or Mistral-7B, 2 samples | 2–4 h each |
| `llama70` | the cohort windows with Llama-3.3-70B, 2 samples, two copies on 4 GPUs each | 10–20 h |
| `validation_all_models` | the validation datasets with the models of `qwen`, `gemma`, `mistral` | 3–6 h each |

Every job can be resumed by running it again. Copy each `outputs/<name>/` folder to
`data/llm_outputs/<name>/`: step 5 parses runs on the cohort windows, and step 14 reads
`data/llm_outputs/<model>_val_<dataset>/`.
