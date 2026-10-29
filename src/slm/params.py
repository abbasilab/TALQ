"""
Analysis parameters and file locations, collected in one place.

Every numbered script imports this module as ``P``.  Parameters are grouped by pipeline stage
and annotated with the section of the Methods that describes them.  Changing a value here
changes the analysis, so values that the manuscript reports are marked "(Methods)".

Directory layout (created by the scripts as needed)::

    config/                         symptom battery, prompts, pre-specified mappings
    data/raw/                       ConvoKit corpora (download with data/raw/download_convokit.py)
    data/external/                  human-annotated validation datasets (step 4 clones them)
    data/interim/                   utterance index and cohort texts (step 1)
    data/processed/                 cohort, windows, lexicon features (steps 2-4, 13, 15)
    data/processed/<run>/           parsed ratings, factors, states of one annotation run (steps 5-12)
    data/llm_outputs/<job>/         stored model outputs, as exported by annotation/export_llm_outputs.py
    results/<run>/                  numbers behind the figures and text, as JSON and CSV
    figures/                        figures (step 21: main text, step 22: Extended Data)

A *run* is one complete set of model ratings of the cohort windows.  The main analysis uses the
run ``llama31_8b`` (Llama-3.1-8B-Instruct, five samples per window and symptom).  The run
``synthetic`` is a simulated stand-in used only to test the pipeline (tests/).
"""

from pathlib import Path

# --------------------------------------------------------------------------------------------
# File locations
# --------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]       # repository root (two levels above src/slm/)
CONFIG = ROOT / "config"
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
LLM_OUTPUTS = DATA / "llm_outputs"
# Stored model outputs: data/llm_outputs/llm_export/<job>/ when written by
# annotation/export_llm_outputs.py, or the same parquet parts placed in data/llm_outputs/<job>/.
LLM_EXPORT = (LLM_OUTPUTS / "llm_export") if (LLM_OUTPUTS / "llm_export").is_dir() else LLM_OUTPUTS
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

MAIN_RUN = "llama31_8b"                          # annotation run analyzed in the manuscript
SYMPTOM_JOB = "5976943c-4594-46b5-a71a-f1809cf1bfb3"     # job id of the stored symptom ratings
INVENTORY_JOB = "7f390e2d-314e-4023-b3af-0800aa9ecc1a"   # job id of the stored BDI-II / PHQ-9 answers


def processed_dir(run: str) -> Path:
    """Folder with the processed tables of one annotation run (ratings, factors, states)."""
    return PROCESSED / run


def results_dir(run: str) -> Path:
    """Folder with the result files (JSON, CSV, NPZ) of one annotation run."""
    return RESULTS / run


# --------------------------------------------------------------------------------------------
# Corpus and cohort (Methods, "Cohort and observation windows")
# --------------------------------------------------------------------------------------------
SUBREDDITS = ["depression", "Anxiety"]           # the two ConvoKit subreddit corpora
EXCLUDE_SPEAKERS = ["[deleted]"]                 # placeholder name of deleted accounts
MIN_POSTS = 20                                   # accounts need at least 20 utterances (Methods)
RATE_KERNEL_WIDTH_DAYS = 28.0                    # half-width of the Epanechnikov kernel (Methods)
RATE_EVAL_STEP_DAYS = 1.0                        # the posting rate is evaluated once per day
ACTIVE_THRESH_PER_DAY = 0.2                      # a day is active if the rate is >= 0.2 per day
MIN_ACTIVE_DAYS = 12 * 30                        # 360 active days in total, not necessarily contiguous
# Accounts run by software rather than people.  AutoModerator met the activity criteria; its
# windows were annotated with the others, and its ratings are discarded when parsing (step 5).
BOT_SPEAKERS = ["AutoModerator"]

# Time zone of the window boundaries (Methods: "Window boundaries were defined in Pacific
# time").  Times are converted to naive wall-clock times in this zone, so a boundary moves by
# one hour in UTC when daylight saving time starts or ends.  The stored model outputs were
# generated for windows built this way, and src/slm/windows.py rebuilds them identically.
WINDOW_TZ = "America/Los_Angeles"

# --------------------------------------------------------------------------------------------
# Windows and the language model questionnaire (Methods, "Language model questionnaire")
# --------------------------------------------------------------------------------------------
N_REPLICATES = 5                                 # samples per window and symptom
WINDOW_DAYS = 14.0                               # window length
ACTIVE_DELAY_DAYS = 1.0                          # the first window starts one day before onset
ACTIVE_DURATION_DAYS = 52 * 7 + 1                # 365 days after onset, giving 27 windows
EXPECTED_N_REQUESTS = 2_577_380                  # requests submitted (all 492 accounts, bot included)

# Sampling settings.  The annotation configuration (config/llm_local.yaml) lists temperature
# 0.1 and top-k 30, but those values were passed to `from_pretrained` (transformers 4.44.2),
# which stores them on the model configuration without using them for generation.  Generation
# therefore used the checkpoint's own defaults below, as stated in the Methods.  Both sets are
# kept so that annotation/run_vllm_annotation.py can apply either one.
GEN_CONFIGURED = dict(do_sample=True, temperature=0.1, top_k=30, repetition_penalty=1.03,
                      max_new_tokens=384)
GEN_LLAMA31_DEFAULT = dict(do_sample=True, temperature=0.6, top_p=0.9, top_k=50,
                           repetition_penalty=1.0, max_new_tokens=384)

# --------------------------------------------------------------------------------------------
# Symptom profiles (Methods, "Symptom profiles")
# --------------------------------------------------------------------------------------------
N_POINTS_MIN = 200                               # users need >= 200 rated window-symptom pairs
CONFIDENCE_THRESH = 40.0                         # keep windows whose mean confidence > 40 for every symptom

# --------------------------------------------------------------------------------------------
# Nonnegative matrix factorization (Methods, "Latent mental factors")
# --------------------------------------------------------------------------------------------
NMF_K_RANGE_SWEEP = (3, 25)                      # broad stability sweep over K
NMF_N_REPLICATES = 100                           # initializations of the final model
NMF_TOL = 1e-5                                   # coordinate-descent tolerance
NMF_MAX_ITER = 10_000
NMF_K = 12                                       # number of factors of the main analysis

# --------------------------------------------------------------------------------------------
# Psychometric properties (Methods, "Psychometric properties")
# --------------------------------------------------------------------------------------------
ALPHA_CONF_QUANTILES = (0.1, 0.9)                # confidence strata for alpha (bottom / top 10 %)
DISCRIM_TOP_FRAC = 0.5                           # items that carry 50 % of a factor's weight
DISCRIM_CONF_QUANTILES = (0.25, 0.75)            # confidence strata for item discrimination
KAPPA_N_BINS = 3                                 # test-retest kappa on tertile bins
KAPPA_DT_MAX_DAYS = 14.01                        # adjacent windows only (exactly 14 days apart)
KAPPA_CONF_QUANTILES = (0.25, 0.75)              # confidence strata for kappa (quartiles)

# --------------------------------------------------------------------------------------------
# UMAP embedding (Methods, "Discrete mental states")
# --------------------------------------------------------------------------------------------
UMAP_SEED = 219_879_005
UMAP_CONFIG = dict(n_neighbors=20, min_dist=0.08, local_connectivity=1,
                   metric="correlation")

# --------------------------------------------------------------------------------------------
# Dirichlet process Gaussian mixture of factor profiles (Methods, "Discrete mental states")
# --------------------------------------------------------------------------------------------
DPMM_N_COMPONENTS_MAX = 100                      # truncation level of the variational mixture
# The 14 hyperparameter settings ("steps") interpolate between these keypoints, from the least
# regularized (step 0) to the most regularized (step 13); see slm.states.dpmm_param_sweep.
DPMM_PARAM_KEYPOINTS = {
    "weight_concentration_prior": [1e-8, 1e-10, 1e-12, 1e-14],
    "mean_precision_prior": [0.001, 0.1, 1.0, 3.0],
    "reg_covar": [10.0, 50.0, 100.0, 150.0],
}
DPMM_PARAM_KINDS = {"weight_concentration_prior": "log", "mean_precision_prior": "log",
                    "reg_covar": "linear"}       # interpolation scale of each hyperparameter
# DPMM_N_INTERP[i] = number of equal steps from keypoint i to keypoint i + 1 (on a log scale for "log"
# kinds): 6 + 5 + 2 + 1 = 14 steps, with the keypoints at steps 0, 6, 11 and 13.
DPMM_N_INTERP = [6, 5, 2]
DPMM_STEP_USED = 5                               # step selected by BIC in the main analysis (11 states)
DPMM_FIT = dict(covariance_type="full", tol=1e-3, max_iter=1000, n_init=10)
DPMM_TRUNC_FRAC = 0.9                            # keep the largest components covering 90 % of windows
DPMM_SEED = 918_293_448

# --------------------------------------------------------------------------------------------
# Mixtures of Markov chains (Methods, "Dynamical regimes")
# --------------------------------------------------------------------------------------------
SEQ_GAP_MAX_DAYS = 4 * 14                        # drop users with any gap longer than 57 days
SEQ_LEN_MIN = 10                                 # users need >= 10 confident windows
EM_N_ITER = 20                                   # iterations of hard-assignment EM
EM_N_INIT = 1000                                 # random restarts in the sweep (10,000 for the final fit)
N_REGIMES_RANGE = (1, 20)                        # numbers of regimes compared by AICc

# --------------------------------------------------------------------------------------------
# Vector autoregression (Methods, "Factor dynamics within individuals")
# --------------------------------------------------------------------------------------------
IRF_HORIZON = 12                                 # responses over 12 windows (24 weeks)
IRF_IMPULSE_SCALE = 50.0                         # impulse of 50 points
STEP_SCALE = 25.0                                # sustained change of 25 points
FDR_Q = 0.05                                     # Benjamini-Hochberg false discovery rate

# Display names of the models used for the validation (output tags of
# local_validation/run_validation.py)
MODEL_NAMES = {
    "llama31_8b_q4km_llamacpp": "Llama-3.1-8B-Instruct (Q4_K_M)",
    "llama31_8b_q4km_ollama": "Llama-3.1-8B-Instruct (Q4_K_M, Ollama)",
    "qwen36_35b_a3b_iq4xs_llamacpp": "Qwen3.6-35B-A3B (UD-IQ4_XS)",
    "qwen38_27b_q4km_llamacpp": "Qwen3.8-27B (UD-Q4_K_M)",
    "qwen3_6_35b_ollama_chat": "Qwen3.6-35B-A3B (Ollama)",
    "qwen3_8_27b_ollama_chat": "Qwen3.8-27B (Ollama)",
}
