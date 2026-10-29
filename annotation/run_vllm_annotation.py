#!/usr/bin/env python3
"""
TALQ symptom annotation with vLLM on GPUs.

Runs the questionnaire of the manuscript (Methods, "Language model questionnaire"; prompts in
Supplementary Note 1) as a batched vLLM job: for every input (a two-week window of a cohort member, or
one post of a validation dataset) and every symptom, the model receives the system message with the
posts and the user message with the symptom's title, description and 0-100 rubric, and returns
`--replicates` samples.  Automatic prefix caching is enabled: all symptom questions about an input
share the long system message that holds its posts, so the posts are processed once per input
rather than once per request.

The prompt text is the one of the stored outputs: the same templates (requests/task_symptom.yaml), the
same rendering of the posts (each post in single quotes, posts separated by blank lines), the symptom
titles as shown to the model (one title differs from config/symptoms.yaml, see --symptom-overrides),
system and user roles, and the model's own chat template with the generation prompt appended.

INPUTS (written by the numbered scripts into annotation/requests/)
    requests/windows.parquet             speaker_id, window_index, window_start, n_posts, posts_text:
                                         the 9,913 windows of the 492 annotated accounts (step 3)
    requests/validation_<name>.parquet   item_id, dataset, posts_text: one post per item (step 4)
    requests/symptoms.yaml, requests/task_symptom.yaml   the battery and the prompts (step 3)

USAGE
    pip install "vllm>=0.6" transformers bitsandbytes pandas pyarrow pyyaml
    # Llama-3.1-8B-Instruct with the settings of the stored outputs, 5 samples per request:
    python3 run_vllm_annotation.py --model meta-llama/Llama-3.1-8B-Instruct \\
        --input requests/windows.parquet --replicates 5 --quantization bnb4 --double-bos \\
        --out outputs/llama31_8b_vllm
    # a validation dataset (any model):
    python3 run_vllm_annotation.py --model meta-llama/Llama-3.1-8B-Instruct \\
        --input requests/validation_irf.parquet --replicates 2 --out outputs/llama31_8b_val_irf

    Data parallelism over several GPUs: one process per GPU with CUDA_VISIBLE_DEVICES=<i> ...
    --shard <i> --n-shards <n>; shards are disjoint sets of inputs (run_gpu_jobs.sh does this).
    Running the same command again resumes: inputs already written by that shard are skipped.

SAMPLING (--sampling)
    annotation  (default) the settings that generated the stored outputs: the default generation
                configuration of the Llama-3.1-8B-Instruct checkpoint, temperature 0.6, top-p 0.9,
                top-k 50, no repetition penalty, at most 384 new tokens (Methods).  The annotation
                configuration (config/llm_local.yaml) lists other values, but they were not applied
                to generation (see GEN_CONFIGURED / GEN_LLAMA31_DEFAULT in src/slm/params.py).  The
                same values are used for every model, so that models are compared at equal settings.
    configured  the values listed in config/llm_local.yaml (temperature 0.1, top-k 30, repetition
                penalty 1.03)
    model       no sampling settings are passed, so vLLM's SamplingParams defaults apply
                (temperature 1, no top-k or top-p truncation)
    greedy      temperature 0

TOKENIZATION (--double-bos)
    The chat template starts the rendered text with the beginning-of-sequence token
    (<|begin_of_text|> for Llama-3.1).  In the annotation job the rendered text was then tokenized with
    special tokens, which adds a second one, so every stored output was generated from a prompt that
    begins with two BOS tokens.  --double-bos gives this token sequence; without it, the prompt is
    the rendered template alone (for Llama-3.1, a single BOS token).

QUANTIZATION (--quantization bnb4)
    loads the weights in 4-bit NF4 with bitsandbytes through vLLM's loader, as for the stored outputs
    (which were generated with transformers 4.44.2 and bitsandbytes NF4).

OUTPUT  <out>/part-<shard>-<time in ms>.parquet, one row per (input, symptom, sample): the input's key
columns, symptom_shortcode, symptom_title, replicate (sample 0..n-1), response (raw text),
n_prompt_tokens, n_output_tokens, finish_reason, model and sampling.  Prompts longer than the model's
context minus --max-new-tokens are not generated; they get one row with replicate -1 and finish_reason
"skipped_too_long", which the analysis drops.  <out>/run_config.json records the settings.  Copy
<out>/ to data/llm_outputs/<name>/ to analyze it (step 5 for cohort windows; step 14 reads
data/llm_outputs/<model>_val_<dataset>/).
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent


def default_overrides():
    """Path of the file with the symptom titles as shown to the model in the stored outputs:
    requests/symptoms_job_overrides.yaml if present, else config/symptoms_job_overrides.yaml of the
    repository, else None."""
    for p in [HERE / "requests" / "symptoms_job_overrides.yaml",
              HERE.parent / "config" / "symptoms_job_overrides.yaml"]:
        if p.exists():
            return str(p)
    return None


def load_symptoms(path, overrides=None):
    """The symptom battery (list of dicts: shortcode, title, description, omit, scale, ...) with
    `prompt_title`, the title shown to the model: `title`, unless the overrides file gives a
    prompt_title for the symptom (as slm.prompts.load_symptoms does)."""
    symptoms = yaml.safe_load(open(path))
    ov = (yaml.safe_load(open(overrides)) or {}) if overrides else {}
    for s in symptoms:
        s["prompt_title"] = ov.get(s["shortcode"], {}).get("prompt_title", s["title"])
    return symptoms


def symptom_desc(spec):
    """Description shown with the symptom; empty for the 12 symptoms marked `omit`."""
    return "" if spec["omit"] else spec["description"]


def scale_string(scale):
    """The 0-100 rubric as lines "- <score>: <anchor text>"."""
    return "\n".join([f"- {list(x)[0]}: {x[list(x)[0]]}" for x in scale])


def messages_for(posts_text, spec, task):
    """System message (instructions and the posts) and user message (the symptom question)."""
    sys_msg = task["system_prompt"].format(posts_text=posts_text)
    usr_msg = task["question_prompt"].format(symptom_title=spec["prompt_title"],
                                             symptom_description=symptom_desc(spec),
                                             symptom_scale_text=scale_string(spec["scale"]))
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}]


def render(tok, msgs, chat_kwargs):
    """Render the messages with the model's chat template, generation prompt appended (a string).

    Templates without a system role (e.g. Gemma, some Mistral versions) raise an error; the system
    message is then prepended to the user message.
    """
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **chat_kwargs)
    except Exception:  # templates without a system role
        merged = [{"role": "user", "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}]
        return tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True, **chat_kwargs)


def sampling_kwargs(name, model_id):
    """Keyword arguments of vllm.SamplingParams for the --sampling choice `name` (see the module
    docstring); `model_id` is not used, since the annotation settings apply to every model."""
    if name == "configured":
        return dict(temperature=0.1, top_k=30, top_p=1.0, repetition_penalty=1.03)
    if name == "annotation":
        return dict(temperature=0.6, top_p=0.9, top_k=50, repetition_penalty=1.0)
    if name == "greedy":
        return dict(temperature=0.0)
    return {}                                     # "model": vLLM's defaults


def main():
    """Parse the arguments, then render, generate and write the outputs chunk by chunk."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="Hugging Face model id, e.g. meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--input", required=True,
                    help="requests/windows.parquet or requests/validation_<name>.parquet")
    ap.add_argument("--out", required=True, help="output folder (created; parts are added to it)")
    ap.add_argument("--symptoms", default=str(HERE / "requests" / "symptoms.yaml"),
                    help="symptom battery (YAML)")
    ap.add_argument("--symptom-overrides", default=default_overrides(),
                    help="YAML with the symptom titles as shown to the model in the stored outputs "
                         "(default: requests/ or config/symptoms_job_overrides.yaml)")
    ap.add_argument("--task", default=str(HERE / "requests" / "task_symptom.yaml"),
                    help="prompt templates (system_prompt, question_prompt)")
    ap.add_argument("--replicates", type=int, default=5, help="samples per input and symptom")
    ap.add_argument("--sampling", default="annotation",
                    choices=["annotation", "configured", "model", "greedy"],
                    help="sampling settings (see SAMPLING above)")
    ap.add_argument("--max-new-tokens", type=int, default=384, help="maximum tokens generated per sample")
    ap.add_argument("--max-model-len", type=int, default=0,
                    help="0 = model's native context (Llama-3.1: 131072; Qwen2.5-7B: 32768; Gemma-2: 8192). "
                         "The longest window is ~66k tokens; longer prompts are skipped and logged")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (GPUs per model copy)")
    ap.add_argument("--gpu-mem", type=float, default=0.90, help="share of GPU memory vLLM may use")
    ap.add_argument("--quantization", default=None, choices=[None, "bnb4", "fp8", "awq"],
                    help="bnb4: 4-bit NF4 with bitsandbytes (as for the stored outputs); fp8, awq: vLLM's")
    ap.add_argument("--shard", type=int, default=0, help="index of this process's shard of the inputs")
    ap.add_argument("--n-shards", type=int, default=1, help="number of shards (processes)")
    ap.add_argument("--windows-per-chunk", type=int, default=200,
                    help="inputs per generation batch and output part")
    ap.add_argument("--symptom-subset", default=None, help="comma-separated shortcodes (default all 52)")
    ap.add_argument("--chat-template-kwargs", default="{}", help='JSON, e.g. {"enable_thinking": false}')
    ap.add_argument("--seed", type=int, default=20240926, help="seed of the vLLM engine")
    ap.add_argument("--dry-run", action="store_true",
                    help="render prompts and print token-length statistics only")
    ap.add_argument("--double-bos", action="store_true",
                    help="tokenize the rendered prompt with special tokens, so that it begins with two BOS "
                         "tokens like the prompts of the stored outputs (see TOKENIZATION above)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    # ---- battery, prompts and inputs
    symptoms = load_symptoms(args.symptoms, args.symptom_overrides)
    if args.symptom_subset:
        keep = set(args.symptom_subset.split(","))
        symptoms = [s for s in symptoms if s["shortcode"] in keep]
    task = yaml.safe_load(open(args.task))
    chat_kwargs = json.loads(args.chat_template_kwargs)

    df = pd.read_parquet(args.input)
    # key of an input: its identifying columns joined by "|"; inputs sorted by key, then dealt to shards
    key_cols = [c for c in ["speaker_id", "window_index", "window_start", "item_id", "dataset"] if c in df]
    df["_key"] = df[key_cols].astype(str).agg("|".join, axis=1)
    df = df.sort_values("_key").reset_index(drop=True)
    df = df.iloc[args.shard::args.n_shards].reset_index(drop=True)

    # ---- resume: inputs already written by this shard are skipped
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    done = set()
    for p in out.glob(f"part-{args.shard:03d}-*.parquet"):
        try:
            done |= set(pd.read_parquet(p, columns=["_key"])["_key"].unique())
        except Exception:  # noqa: BLE001  (a part left incomplete by an interrupted write)
            pass
    todo = df[~df["_key"].isin(done)].reset_index(drop=True)
    print(f"[info] {len(df)} inputs in shard, {len(done)} done, {len(todo)} to do; "
          f"{len(symptoms)} symptoms x {args.replicates} replicates")

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.dry_run:                     # prompt lengths of the first 200 inputs, first symptom
        lens = []
        for _, r in todo.head(200).iterrows():
            for s in symptoms[:1]:  # noqa: B007
                lens.append(len(tok(render(tok, messages_for(r["posts_text"], s, task), chat_kwargs),
                                    add_special_tokens=False)["input_ids"]))
        print(pd.Series(lens).describe()); return

    # ---- engine
    from vllm import LLM, SamplingParams

    llm_kw = dict(model=args.model, tensor_parallel_size=args.tp, enable_prefix_caching=True,
                  gpu_memory_utilization=args.gpu_mem, seed=args.seed)
    if args.max_model_len:
        llm_kw["max_model_len"] = args.max_model_len
    if args.quantization == "bnb4":
        llm_kw.update(quantization="bitsandbytes", load_format="bitsandbytes")
    elif args.quantization in ("fp8", "awq"):
        llm_kw.update(quantization=args.quantization)
    llm = LLM(**llm_kw)
    skw = sampling_kwargs(args.sampling, args.model)
    max_len = llm.llm_engine.model_config.max_model_len
    limit = max_len - args.max_new_tokens          # longest prompt that leaves room for the answer
    print(f"[info] max_model_len = {max_len}; prompts longer than {limit} tokens are skipped")

    meta = {"model": args.model, "sampling": args.sampling, "sampling_kwargs": skw,
            "replicates": args.replicates, "max_new_tokens": args.max_new_tokens,
            "quantization": args.quantization, "vllm_seed": args.seed, "double_bos": args.double_bos,
            "symptom_overrides": args.symptom_overrides}
    (out / "run_config.json").write_text(json.dumps(meta, indent=2))

    # ---- generation, one chunk of inputs at a time (one output part per chunk)
    n_chunks = (len(todo) + args.windows_per_chunk - 1) // args.windows_per_chunk
    t0 = time.time()
    for ci in range(n_chunks):
        chunk = todo.iloc[ci * args.windows_per_chunk:(ci + 1) * args.windows_per_chunk]
        prompts, keys, sps = [], [], []
        skipped = []
        for _, r in chunk.iterrows():
            for s in symptoms:
                text = render(tok, messages_for(r["posts_text"], s, task), chat_kwargs)
                n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
                if n_tok > limit:
                    skipped.append((r["_key"], s["shortcode"], n_tok)); continue
                # a seed per request (input and symptom), so results do not depend on batching or sharding
                seed = int(hashlib.sha1(f"{r['_key']}|{s['shortcode']}".encode()).hexdigest()[:8], 16)
                sps.append(SamplingParams(n=args.replicates, max_tokens=args.max_new_tokens, seed=seed,
                                          **skw))
                prompts.append(text)
                keys.append({**{c: r[c] for c in key_cols}, "_key": r["_key"],
                             "symptom_shortcode": s["shortcode"], "symptom_title": s["prompt_title"],
                             "n_prompt_tokens": n_tok})
        # the rendered text already starts with BOS; --double-bos lets the tokenizer add a second one
        tok_prompts = [{"prompt_token_ids": tok(p, add_special_tokens=args.double_bos)["input_ids"]}
                       for p in prompts]
        res = llm.generate(tok_prompts, sps, use_tqdm=True) if tok_prompts else []
        rows = []
        for k, rr in zip(keys, res):
            for rep, o in enumerate(rr.outputs):       # the n samples of one request
                rows.append({**k, "replicate": rep, "response": o.text,
                             "n_output_tokens": len(o.token_ids), "finish_reason": o.finish_reason,
                             "model": args.model, "sampling": args.sampling})
        for key, sc, n_tok in skipped:
            rows.append({"_key": key, "symptom_shortcode": sc, "replicate": -1, "response": None,
                         "n_prompt_tokens": n_tok, "finish_reason": "skipped_too_long",
                         "model": args.model, "sampling": args.sampling})
        pd.DataFrame(rows).to_parquet(out / f"part-{args.shard:03d}-{int(time.time()*1000)}.parquet",
                                      index=False, compression="zstd")
        el = time.time() - t0
        print(f"[chunk {ci+1}/{n_chunks}] {len(rows)} rows, {len(skipped)} skipped, "
              f"{el/60:.1f} min elapsed, ETA {el/(ci+1)*(n_chunks-ci-1)/60:.1f} min", flush=True)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
