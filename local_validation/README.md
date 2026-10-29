# Reduced validation protocol on a single local computer

This kit scores the posts of three human-annotated Reddit datasets, and a calibration set of cohort
windows, with Llama-3.1-8B-Instruct (4-bit, Q4_K_M) served on the same computer by llama.cpp (or
Ollama). It needs Python 3 (standard library only) and a model server; it was tested on macOS with an
Apple M5 processor. The results are analyzed by `scripts/14b_validation_local.py` and
`scripts/14c_model_comparison_local.py` (Methods, "Validation against human annotations" and
"Additional language models"; Fig. 1d and Extended Data Fig. 3).

## What it runs

The prompts and sampling settings are those of the cohort annotation: the Llama-3.1-Instruct chat
template with two BOS tokens at the start of the prompt, temperature 0.6, top-k 50, top-p 0.9 (applied
in this order, as by the transformers sampler of the annotation), no repetition penalty and at most 384
new tokens. Each request gets one sample, with a seed derived from its key. Only the symptoms mapped to
the labels in advance and the control symptoms are scored (`config/validation_mapping.yaml`; its
SHA-256 is recorded in `config/validation_mapping.record.json`).

| Set | Items | Symptoms scored per item |
|---|---|---|
| Interpersonal risk factors: thwarted belongingness, perceived burdensomeness | posts balanced over the four label combinations | the 9 symptoms mapped to the two labels and 3 control symptoms |
| LoST: low self-esteem | posts balanced over the two levels | 5 mapped symptoms and 3 control symptoms |
| Depression severity (minimum, mild, moderate, severe) | posts balanced over the four levels | the 20 items of criterion A |
| Calibration | cohort windows | 8 symptoms, compared with the cohort ratings of the same windows |

| Preset | Posts per dataset | Calibration windows | Requests | Llama-3.1-8B, Apple M5 (about 760 requests per hour) |
|---|---|---|---|---|
| `smoke` | 5 | 5 | 240 | about 20 min |
| `quick` | 80 | 40 | 3,520 | 4–5 h |
| `standard` (the protocol of the manuscript) | 300 | 80 | 12,640 | about 17 h |
| `extended` | 800 | 150 | 33,200 | about 2 days |

Items are taken in the order of their rank, so each preset contains the previous one and a partial run
still covers every dataset.

## Steps

Commands are given from the root of the repository.

0. Create the request files in `requests/` (after steps 3 and 4 of the pipeline; see
   [Input files](#input-files)):
   ```
   cd scripts
   python3 04b_prepare_local_validation.py
   ```
   Copy `local_validation/` to the computer that runs the model if it is a different one.
1. If the Ollama app is running, quit it (menu bar: Ollama > Quit Ollama).
2. Terminal 1, the model server (keep it open):
   ```
   cd local_validation
   PARALLEL=8 bash start_server.sh
   ```
   Wait for `server is listening`. When Ollama is installed, `start_server.sh` uses the `llama-server`
   bundled with it and, if Ollama has downloaded `llama3.1:8b-instruct-q4_K_M`, that model file. The
   run of the manuscript used this file, Ollama blob
   `sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29`.
   Otherwise the model is downloaded from Hugging Face (`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`,
   Q4_K_M). Without Ollama, install llama.cpp first (`brew install llama.cpp`).
3. Terminal 2, a short check, then the protocol:
   ```
   cd local_validation
   python3 run_validation.py --preset smoke --workers 8
   python3 run_validation.py --preset standard --workers 8
   ```
   On macOS, prefix these commands with `caffeinate -i` to keep the computer awake.

`PARALLEL=8` and `--workers 8` suit 24 GB of memory or more; with 16 GB, leave both out (4 slots).
A run can be stopped with Ctrl-C at any time; the same command resumes it. Progress and an estimate of
the remaining time are printed every 50 requests. Each request is appended to
`outputs/<model-tag>.jsonl`, here `outputs/llama31_8b_q4km_llamacpp.jsonl`. Then, from `scripts/`:

```
python3 14b_validation_local.py
python3 14c_model_comparison_local.py       # when a second model was run
```

## A second model

The same posts are scored by Qwen3.6-35B-A3B (mixture of experts, 3 billion active parameters,
4-bit UD-IQ4_XS from `unsloth/Qwen3.6-35B-A3B-GGUF`, 17.7 GB download) with the same prompts and
sampling settings, its own chat template and its thinking mode switched off (otherwise every answer
starts with a long reasoning text). The standard preset took about 14 hours on the Apple M5. Each model
writes its own output file, here `outputs/qwen36_35b_a3b_iq4xs_llamacpp.jsonl`.

```
MODEL=qwen3.6-35b bash start_server.sh                                 # terminal 1
python3 run_validation.py --model qwen3.6-35b --preset smoke             # terminal 2
python3 run_validation.py --model qwen3.6-35b --preset standard
```

The model file goes to `~/models` (change with `MODEL_DIR=...`); the download resumes if it stops.
Close other large applications, since the model uses most of the memory. The kit also knows
Qwen3.8-27B (`qwen3.8-27b`, dense, 4-bit UD-Q4_K_M, 16.5 GB), which is several times slower. If the run
script says that the server runs a different model, restart the server with the matching `MODEL=`. The
run stops by itself if the model keeps thinking before it answers.

### If the server cannot load Qwen

Recent Qwen models need a recent llama.cpp. If `start_server.sh` reports that the model could not be
loaded, install the current version (`brew install llama.cpp`, or `brew upgrade llama.cpp`) and start
it again. Without Homebrew, use Ollama: `MODEL=qwen3.6:35b bash start_ollama.sh` in terminal 1 (this
downloads Ollama's copy of the model, about 24 GB, which may run partly on the CPU), and
`python3 run_validation.py --model qwen3.6-35b --backend ollama --preset smoke` in terminal 2.

## Through Ollama

`bash start_ollama.sh` serves the model through Ollama's API, and `run_validation.py` detects it.
The caller cannot set the sampler order through Ollama: `start_ollama.sh` selects Ollama's own engine,
whose sampler applies top-k, the temperature and top-p (equivalent to the order of the cohort
annotation), whereas Ollama's llama.cpp-based engine applies top-k and top-p before the temperature.
Answers go to `outputs/llama31_8b_q4km_ollama.jsonl` (other models: `outputs/<tag>_ollama_chat.jsonl`),
so the two routes are never mixed. The results of the manuscript come from the llama.cpp route.

## Input files

`scripts/04b_prepare_local_validation.py` writes the request files from the outputs of steps 3 and 4
(`data/processed/validation_labels.parquet`, `annotation/requests/validation_<dataset>.parquet`,
`data/processed/windows.parquet`) and `config/`. `requests/` holds post text and is not tracked in git;
`outputs/` neither.

| File | Content |
|---|---|
| `requests/posts.jsonl` | 800 posts per dataset, one JSON object per line: `item_id`, `dataset`, `rank`, `posts_text` (the post rendered as a one-post window, as in `annotation/requests/validation_<dataset>.parquet`) and `symptoms` (the symptoms to score) |
| `requests/calibration_windows.jsonl` | 150 cohort windows (`calib_000` to `calib_149`, rank in this order), the same fields with dataset `calibration` and eight symptoms (sadness, anxiety, initial insomnia, worthlessness, suicidal ideation, loneliness, loss of interest, feeling happy) |
| `requests/symptoms.json` | the battery of `config/symptoms.yaml` with the titles shown to the model (`config/symptoms_job_overrides.yaml`) |
| `requests/task_symptom.json` | the prompt templates of `config/task_symptom.yaml` |

**Posts.** The script draws the posts with one `numpy.random.RandomState(20261001)`, used in turn for
the interpersonal risk factors, LoST and severity. For each combination of label values (in sorted
order), the posts with that combination (in the row order of `data/processed/validation_labels.parquet`)
are permuted and the first 800 / (number of combinations) are kept. Rank r is post number floor(r / c) of
combination r mod c, where c is the number of combinations, so that every preset is balanced. This gives
exactly the posts scored for the manuscript.

**Calibration windows.** `data/processed/calibration_window_map.parquet` records the cohort window
(`speaker_id`, `window_start`, `window_index`) of each calibration item. The record of the 150 windows
used in the manuscript identifies accounts and is not distributed with the repository; it is available
from the authors.

- With the record in `data/processed/`, the script rebuilds `calibration_windows.jsonl` from it (post
  text from `data/processed/windows.parquet`), which gives exactly the calibration windows of the
  manuscript. The procedure that drew them cannot be regenerated: they were not drawn from all windows
  (none has more than 5,773 characters of post text, consistent with a cap that keeps each prompt within
  the 4,096-token context of a server slot).
- Without the record, the script draws a new set and writes its record: 150 windows without replacement,
  with the same random generator continued after the posts, among the windows of accounts other than the
  bot whose post text has at most 6,000 characters. The calibration results then differ from those of
  the manuscript.

## Differences from the cohort annotation

| | Cohort annotation | This kit |
|---|---|---|
| Runtime | transformers 4.44.2 and bitsandbytes on an NVIDIA GPU | llama.cpp (or Ollama) on the local GPU |
| Quantization | NF4, 4 bits | Q4_K_M GGUF, 4 bits |
| Model, prompts, chat template, sampling settings | Llama-3.1-8B-Instruct | the same |
| Samples per request | 5 | 1 |

The calibration windows measure how closely the two deployments agree on identical inputs.
