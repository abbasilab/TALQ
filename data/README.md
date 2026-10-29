# Data

Only the download script and the checksums of the corpora are part of the repository. Everything else
in this folder is downloaded or created by the pipeline, and is excluded from version control because it
holds account names, post text or model outputs.

## Reddit corpora (`raw/`)

The ConvoKit subreddit corpora of r/depression and r/Anxiety (Chang et al., 2020), which contain every
post and comment from the creation of each community to 31 October 2018.

```bash
cd data/raw && python3 download_convokit.py
```

The script resolves the download URLs as ConvoKit does, resumes partial downloads, checks that each
archive opens, and records URL, size and SHA-256 in `download_manifest.json`. The archives analyzed in
the manuscript (retrieved on 1 October 2026) were:

| File | Bytes | SHA-256 |
|---|---|---|
| `depression.corpus.zip` | 702,851,424 | `3b4e2bac45df2649e94c372c1bee8e5d210fbc65174c65c2c54a3e15ea815868` |
| `Anxiety.corpus.zip` | 242,961,641 | `512898716c76cf156fc2f0e9f7ec033775520ff4a3de03a32a9918ddd1ac181f` |

## Validation datasets (`external/`)

Step 4 (`scripts/04_prepare_validation.py`) clones these public repositories with `git clone --depth 1`
and prints the commit of each clone. The commits used for the manuscript:

| Folder | Repository | Commit | Labels |
|---|---|---|---|
| `Irf/` | github.com/drmuskangarg/Irf | `1e46a3d955cfd1e457d2869f1b6c2954aaaf7971` | thwarted belongingness, perceived burdensomeness |
| `LoST/` | github.com/drmuskangarg/LoST | `b1a0f44023e8ed0a4cc0ba361afed17ed5aa9e46` | low self-esteem |
| `Depression_Severity_Dataset/` | github.com/usmaann/Depression_Severity_Dataset | `e21dbbf2c61f952628df18b037c81875504b9428` | depression severity (four levels) |
| `DepressionEmo/` | github.com/abuBakarSiddiqurRahman/DepressionEmo | `f5b5fdad8ad0014a634310e7fae8261bbc91591a` | eight emotions (weak labels, secondary analysis) |

## Stored model outputs (`llm_outputs/`)

Parquet parts of the two annotation jobs, as written by `annotation/export_llm_outputs.py`, in
`llm_outputs/llm_export/<job>/` or directly in `llm_outputs/<job>/`:

* `5976943c-4594-46b5-a71a-f1809cf1bfb3`: the 52-symptom questionnaire (read by step 5),
* `7f390e2d-314e-4023-b3af-0800aa9ecc1a`: BDI-II and PHQ-9 (read by step 18).

`llm_outputs/_synthetic/` holds the simulated stand-in written by `tests/make_synthetic_llm_export.py`.

## Created by the pipeline

| Path | Step | Content |
|---|---|---|
| `interim/utterance_index.parquet` | 1 | every utterance of both corpora, without text |
| `interim/cohort_texts.parquet` | 2 | all utterances of the cohort, with text |
| `processed/cohort.csv` | 2 | accounts with at least 20 utterances, posting activity, cohort membership |
| `processed/windows.parquet` | 3 | one row per window with text |
| `processed/validation_labels.parquet` | 4 | labels of the validation posts |
| `processed/calibration_window_map.parquet` | 4b | the cohort windows of the calibration set |
| `processed/lexicon_*.parquet` | 13 | lexicon and length features |
| `processed/user_description.parquet` | 15 | per-user description |
| `processed/inventories.parquet` | 18 | BDI-II and PHQ-9 answers linked to cohort windows |
| `processed/<run>/` | 5-12 | parsed ratings, window-by-symptom tables, factor scores, states, UMAP |
