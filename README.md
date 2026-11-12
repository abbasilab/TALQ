# TALQ: language model questionnaires for dense measurement of mental states

Code for the manuscript

> Levesque, M. E., Poskanzer, K., Woolley, J. & Abbasi-Asl, R. *TALQ, a reproducible language model questionnaire for dense
> measurement of depressive and anxious states* (2026).

TALQ (text assessment with language model questionnaires) uses a large language model
(Llama-3.1-8B-Instruct) as a structured rater. For every two-week window
of a person's posts and comments on Reddit it rates each of 52 symptoms compiled from depression rating
scales (0 to 100, with a justification and a confidence), five times per window and symptom. The code
covers every step from the public corpora to the figures: cohort selection, prompt construction,
annotation, parsing, validation against human-annotated datasets, agreement with inventories completed by
the same model, factorization into 12 replicable mental health factors, psychometrics, lagged dynamics
within individuals, discrete states, mixtures of Markov chains, course types over the year and the
sensitivity analyses.

## Contents

| Folder | What it holds |
|---|---|
| `src/slm/` | Library: corpus reading, cohort and windows, prompts, response parsing, staNMF, psychometrics, panel VAR, states, Markov mixtures, growth mixture model of course types, lexicons, figure style and editable PowerPoint export |
| `scripts/` | Numbered pipeline steps (table below) |
| `config/` | Symptom battery, prompt templates, model configuration of the annotation, the mappings of validation labels and inventory items to symptoms specified in advance, with their SHA-256 records |
| `annotation/` | Export of stored model outputs from a MongoDB job database, and a vLLM runner that renders the same prompts on GPUs ([README](annotation/README.md)) |
| `local_validation/` | Self-contained kit (Python standard library) that runs the validation protocol on one local computer through llama.cpp or Ollama ([README](local_validation/README.md)) |
| `data/` | Download script and checksums of the corpora ([README](data/README.md)); everything else under `data/` is created by the pipeline and is not part of the repository |
| `tests/` | A simulated stand-in for the model outputs, to test the pipeline end to end |
| `third_party/empath/` | The Empath lexicon (Fast et al., 2016) |
| `run_all.sh` | Runs the pipeline |

## Installation

Python 3.11 on Linux or macOS.

```bash
python3 -m pip install -r requirements.txt
```

The versions in `requirements.txt` are the ones used for every reported result. Annotation on GPUs needs
`requirements-gpu.txt` instead (vLLM).

## Data

* **Reddit corpora.** The ConvoKit corpora of r/depression and r/Anxiety. Download them with
  `python3 data/raw/download_convokit.py`; `data/raw/download_manifest.json` lists the URLs and SHA-256
  checksums of the archives that were analyzed.
* **Human-annotated validation datasets.** Step 4 clones the four public repositories into
  `data/external/` (commits in [data/README.md](data/README.md)).
* **Stored model outputs.** The ratings analyzed in the manuscript come from annotation job
  `5976943c-4594-46b5-a71a-f1809cf1bfb3` (52 symptoms) and job `7f390e2d-314e-4023-b3af-0800aa9ecc1a`
  (BDI-II and PHQ-9). They are not part of this repository (see the data availability statement of the
  manuscript). Place them in `data/llm_outputs/<job>/` as parquet parts with the schema written by
  `annotation/export_llm_outputs.py`, or generate new ratings with `annotation/run_vllm_annotation.py`.
* **Local validation outputs.** Model responses of the local validation runs go to
  `local_validation/outputs/` (see its README).

## Running the pipeline

```bash
bash run_all.sh data        # steps 1-4, 4b, 13, 15 (about 10 min)
bash run_all.sh main        # steps 5-12, sensitivity analyses, 14, 14b-14e, 18, 20, figures (about 11 h)
bash run_all.sh figures     # figures, Table 1 and the numbers quoted in the text only
QUICK=1 bash run_all.sh synthetic   # pipeline test on simulated ratings (under an hour)
```

Every step writes a log to `results/logs/`. `RESUME=1` skips steps that finished, `NJOBS` sets the number
of parallel workers. Random steps use fixed seeds, so results are identical across runs on the same
software versions. The main annotation run is called `llama31_8b`; its processed tables are in
`data/processed/llama31_8b/` and its results in `results/llama31_8b/`.

| Step | Script | Manuscript | Output |
|---|---|---|---|
| 1 | `01_ingest_corpus.py` | Methods, data source | index of all 4,547,682 utterances |
| 2 | `02_select_cohort.py` | Methods, cohort | posting rates, cohort of 491 users (plus one bot, excluded from analysis), texts of the cohort |
| 3 | `03_build_windows.py` | Methods, windows | 27 windows of 14 days per user (9,886 with text), requests for the annotation |
| 4 | `04_prepare_validation.py` | Methods, validation | human-annotated posts rendered as one-post windows, labels, mapping of labels to symptoms |
| 4b | `04b_prepare_local_validation.py` | Methods, validation | request files of the local validation kit |
| 5 | `05_parse_collate.py` | Methods, questionnaire and symptom profiles; ED Fig. 2 | parsed ratings, window-by-symptom tables, confidence filter |
| 6 | `06_nmf.py` | Fig. 2, Table 1, ED Fig. 5 | staNMF stability sweeps, 12-factor model, K = 10 and K = 18 solutions |
| 7 | `07_psychometrics.py` | Fig. 2d, ED Fig. 6 | weighted α and item discrimination (test–retest κ and ICC are also computed but not reported) |
| 8 | `08_var_population.py` | Fig. 3, ED Fig. 7 | panel VAR with half-panel jackknife, Granger tests, impulse and step responses, null calibration |
| 9 | `09_umap.py` | Fig. 4a | UMAP embedding of the factor profiles |
| 10, 10b, 10c | `10_states_dpmm.py`, `10b_alt_states.py`, `10c_name_states.py` | Fig. 4, ED Fig. 8 | Dirichlet process mixture sweep, BIC selection (11 states), eight-state sensitivity solution, state labels |
| 11 | `11_regimes.py` | Fig. 5d,e, ED Fig. 9 | mixtures of Markov chains (1–20 regimes, AICc), burden, transition matrices |
| 12 | `12_robustness.py` | Fig. 1c, 2c, 5a–c, ED Figs 5, 9, 10 | replicate reliability, split-half and confidence-threshold replication, held-out likelihood and shuffle nulls, Gaussian HMM, stability across halves of the year |
| 12c | `12c_community_replication.py` | ED Fig. 5b | refits within each community |
| 13 | `13_lexicon_features.py` | ED Fig. 3 | VADER, Empath and style features |
| 14, 14b, 14c, 14d, 14e | `14_validation_analysis.py`, `14b_validation_local.py`, `14c_model_comparison_local.py`, `14d_validation_predictive.py`, `14e_validation_single_symptom.py` | Fig. 1d, ED Fig. 3 | discrimination of human labels by single symptoms, calibration of the local deployment, comparison of two models, cross-validated prediction of the human labels from all scored symptoms (lasso and neural network), best single symptom and paired differences between predictions |
| 15 | `15_cohort_description.py` | ED Fig. 1 | cohort description |
| 18 | `18_inventories.py` | ED Fig. 4 | BDI-II and PHQ-9 answered by the same model versus the battery |
| 20 | `20_course_types.py` | Fig. 6 | change over months (semivariogram), reliability of level, variability, course, inertia and instability, growth mixture model of course types, their reproducibility across interleaved halves and description by quarter of the year |
| 21 | `21_figures_main.py` | Figs 1–6, Table 1 | figures as PDF and SVG (vector), PNG and TIFF (600 dpi) and editable PPTX |
| 22 | `22_figures_extended.py` | Extended Data Figs 1–10 | figures as PDF and SVG (vector), PNG and TIFF (600 dpi) and editable PPTX |
| 23 | `23_manuscript_numbers.py` | text | every number quoted in the manuscript, with its source |
| 99 | `99_pseudonymize.py` | | a release copy of the ratings with pseudonymous user identifiers and no text |

`src/slm/params.py` collects every analysis parameter with the section of the Methods that reports it.
In every figure and in Table 1, each factor carries the color of its clinical family (`FAMILY` and
`FAMILY_COLORS` in `src/slm/plotting.py`).

## Annotation and local validation

The stored ratings were generated with transformers 4.44.2 and 4-bit NormalFloat weights
(`config/llm_local.yaml`). `annotation/run_vllm_annotation.py` renders the identical prompts (including the
symptom title and the token sequence shown to the model, see `config/symptoms_job_overrides.yaml`) and
samples with the same settings, so that new models or new data can be annotated in the same way.
`local_validation/` runs the reduced validation protocol of the manuscript on a single computer, which
keeps the posts on that computer.

## Data protection

The corpora are public, but the cohort are people writing about their mental health. Account names are
the keys of the corpora and therefore appear in the processed data and in some intermediate tables. Do
not share `data/` or `results/` as they are. `scripts/99_pseudonymize.py` writes a release copy with salted
hashes instead of account names, without post text or text written by the model. The figures contain no
account names or text.

# Citation

```bibtex
@article{levesque2026talq,
  title   = {TALQ: A reproducible language model questionnaire for dense measurement of depressive and anxious states},
  author  = {Levesque, Maxine and Poskanzer, Kira and Woolley, Joshua and Abbasi-Asl, Reza},
  journal = {medRxiv},
  year    = {2026}
}
```
