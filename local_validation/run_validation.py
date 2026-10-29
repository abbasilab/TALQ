#!/usr/bin/env python3
"""
Reduced validation protocol on a single local computer (Python standard library only).

Scores the posts of three human-annotated Reddit datasets and a small calibration set of cohort
windows with a model served on the same computer, one sample per request (Methods, "Validation
against human annotations" and "Additional language models"; analyzed by scripts/14b_validation_local.py
and scripts/14c_model_comparison_local.py).  Tested on macOS with an Apple M5 processor.

Reference model: Llama-3.1-8B-Instruct, 4-bit (Q4_K_M), with the prompts and sampling settings of the
cohort annotation: the Llama-3.1-Instruct chat template rendered as for the annotation job, so that the
prompt begins with two BOS tokens, and the checkpoint's default sampling settings (temperature 0.6,
top-k 50, top-p 0.9, no repetition penalty, at most 384 new tokens).

Other models are scored with the same prompts and sampling settings, their own chat template applied
by the server, and their thinking mode switched off:
    MODEL=qwen3.6-35b bash start_server.sh                       (terminal 1)
    python3 run_validation.py --model qwen3.6-35b --preset smoke  (terminal 2)

Two local servers are supported (the script detects which one is running):
    llama.cpp  bash start_server.sh    (llama-server; the sampler order is set explicitly)
    Ollama     bash start_ollama.sh    (Ollama's API; Ollama decides the sampler order)

    python3 run_validation.py --preset smoke      # check, about 20 minutes for Llama-3.1-8B
    python3 run_validation.py --preset standard   # the protocol of the manuscript, resumable
On macOS, prefix the commands with `caffeinate -i` to keep the computer awake.

Presets (posts per dataset / calibration windows):
    smoke 5 / 5   quick 80 / 40   standard 300 / 80   extended 800 / 150
Items are taken in the order of their rank, so each preset contains the previous one.  Every request
is appended to outputs/<model-tag>.jsonl as soon as it finishes; running a command again skips the
requests already done.

Inputs (requests/, written by scripts/04b_prepare_local_validation.py; they hold post text and are
not tracked in git):
    posts.jsonl               validation posts: item_id, dataset, rank, posts_text, symptoms to score
    calibration_windows.jsonl cohort windows: the same fields, dataset "calibration"
    symptoms.json             the battery, with the symptom titles as shown to the model
    task_symptom.json         the prompt templates (system_prompt, question_prompt)
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REQ = os.path.join(HERE, "requests")
# (posts per dataset, calibration windows) of each preset; scripts/14b_validation_local.py has a copy
PRESETS = {"smoke": (5, 5), "quick": (80, 40), "standard": (300, 80), "extended": (800, 150)}
DEFAULT_URL = {"llamacpp": "http://127.0.0.1:8080", "ollama": "http://127.0.0.1:11434"}
NUM_CTX = 4096          # per request; the longest prompt is ~2,100 tokens + 384 new tokens
# Models: output file (tag), display name, how the prompt is sent, and the Ollama tag of the fallback route.
# "chat": the server applies the model's own chat template; "no_think": thinking mode off.
# "file" is checked against the model the llama.cpp server reports, so results of two models never mix.
MODELS = {
    "llama3.1-8b": {"tag": "llama31_8b_q4km_llamacpp", "name": "Llama-3.1-8B-Instruct (Q4_K_M)",
                    "chat": False, "no_think": False, "file": None, "ollama": "llama3.1:8b-instruct-q4_K_M"},
    "qwen3.6-35b": {"tag": "qwen36_35b_a3b_iq4xs_llamacpp", "name": "Qwen3.6-35B-A3B (UD-IQ4_XS)",
                    "chat": True, "no_think": True, "file": "Qwen3.6-35B-A3B", "ollama": "qwen3.6:35b"},
    "qwen3.8-27b": {"tag": "qwen38_27b_q4km_llamacpp", "name": "Qwen3.8-27B (UD-Q4_K_M)", "chat": True,
                    "no_think": True, "file": "Qwen3.8-27B", "ollama": "qwen3.8:27b"},
}

ap = argparse.ArgumentParser(description="Reduced validation protocol on a single local computer.")
ap.add_argument("--preset", choices=list(PRESETS), default="smoke",
                help="number of posts per dataset and of calibration windows (see the module docstring)")
ap.add_argument("--model", choices=list(MODELS), default="llama3.1-8b",
                help="which model the server runs (start it with MODEL=<same name> bash start_server.sh)")
ap.add_argument("--backend", choices=["auto", "ollama", "llamacpp"], default="auto",
                help="model server; auto: llama.cpp if it answers on its port, else Ollama")
ap.add_argument("--server", default=None, help="server URL (default: the backend's local port)")
ap.add_argument("--ollama-model", default=None, help="Ollama tag (default: the --model's tag)")
ap.add_argument("--workers", type=int, default=4,
                help="parallel requests; must not exceed the server's parallel slots (PARALLEL)")
ap.add_argument("--model-tag", default=None, help="output file name (default depends on the backend)")
ap.add_argument("--datasets", default="calibration,irf,lost,severity",
                help="comma-separated datasets to score (calibration = the cohort windows)")
ap.add_argument("--replicates", type=int, default=1, help="samples per item and symptom")
ap.add_argument("--max-new-tokens", type=int, default=384, help="maximum tokens generated per answer")
ap.add_argument("--no-double-bos", action="store_true",
                help="Llama prompt with a single BOS token (by default it begins with two, as in the "
                     "cohort annotation)")
ap.add_argument("--chat", action="store_true",
                help="other models: send the system and user messages and let the server apply the model's "
                     "own chat template (Ollama /api/chat, llama.cpp /v1/chat/completions)")
ap.add_argument("--no-think", action="store_true",
                help="switch the model's thinking mode off (implied for the Qwen models)")
ap.add_argument("--dry-run", action="store_true",
                help="print one rendered prompt and the request count, then exit")
args = ap.parse_args()
MODEL = MODELS[args.model]
if MODEL["chat"]:
    args.chat = True
if MODEL["no_think"]:
    args.no_think = True
if args.ollama_model is None:
    args.ollama_model = MODEL["ollama"]

n_posts, n_cal = PRESETS[args.preset]
symptoms = {s["shortcode"]: s for s in json.load(open(os.path.join(REQ, "symptoms.json")))}
task = json.load(open(os.path.join(REQ, "task_symptom.json")))


def load(path, n):
    """Items of a request file (one JSON object per line) whose rank is below n."""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["rank"] < n:
                rows.append(r)
    return rows


# ---- the items of the preset and their requests ---------------------------------------------------
wanted = set(args.datasets.split(","))
items = []
if "calibration" in wanted:
    items += load(os.path.join(REQ, "calibration_windows.jsonl"), n_cal)
items += [r for r in load(os.path.join(REQ, "posts.jsonl"), n_posts) if r["dataset"] in wanted]
# interleave datasets so a partial run still covers all of them
items.sort(key=lambda r: (r["rank"], r["dataset"]))


def scale_string(scale):
    """The 0-100 rubric as lines "- <score>: <anchor text>"."""
    return "\n".join("- {}: {}".format(list(x)[0], x[list(x)[0]]) for x in scale)


def render(posts_text, spec):
    """Llama-3.1-Instruct chat template (system + user, generation prompt), rendered as for the cohort
    annotation: the checkpoint's template with its default date lines; message contents are stripped,
    as the template does."""
    sys_msg = task["system_prompt"].format(posts_text=posts_text).strip()
    usr_msg = task["question_prompt"].format(
        symptom_title=spec["title"], symptom_description="" if spec["omit"] else spec["description"],
        symptom_scale_text=scale_string(spec["scale"])).strip()
    text = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n" + sys_msg + "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n" + usr_msg + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")
    # Both servers add the model's BOS themselves; keeping the literal <|begin_of_text|> gives the two
    # BOS tokens of the cohort annotation's prompts.  --no-double-bos removes the literal one.
    return text[len("<|begin_of_text|>"):] if args.no_double_bos else text


# one request per item, symptom and sample; key "<item_id>|<symptom>|<sample>"
requests = []
for r in items:
    for sc in r["symptoms"]:
        for rep in range(args.replicates):
            key = "{}|{}|{}".format(r["item_id"], sc, rep)
            requests.append((key, r, sc, rep))

if args.dry_run:
    print("preset {}: {} items, {} requests".format(args.preset, len(items), len(requests)))
    k, r, sc, rep = requests[0]
    print("---- example prompt ({} / {}; model {}) ----".format(r["item_id"], sc, args.model))
    if args.chat:      # the server applies the model's chat template to these two messages
        sys_m = task["system_prompt"].format(posts_text=r["posts_text"]).strip()
        spec = symptoms[sc]
        usr_m = task["question_prompt"].format(
            symptom_title=spec["title"], symptom_description="" if spec["omit"] else spec["description"],
            symptom_scale_text=scale_string(spec["scale"])).strip()
        print("[system]\n" + sys_m[:2000] + "\n[user]\n" + usr_m)
        print("---- thinking mode: {} ----".format("off" if args.no_think else "model default"))
    else:
        print(render(r["posts_text"], symptoms[sc])[:3000])
    sys.exit(0)


def probe(url):
    """HTTP status of a GET (None if nothing is listening)."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:  # noqa: BLE001
        return None


# ---- which server, which output file --------------------------------------------------------------
backend = args.backend
if backend == "auto" and args.server:
    if probe(args.server + "/api/version") == 200:
        backend = "ollama"
    elif probe(args.server + "/health") is not None:
        backend = "llamacpp"
    else:
        sys.exit("No model server answers at {}.".format(args.server))
if backend == "auto":
    if probe(DEFAULT_URL["llamacpp"] + "/health") is not None:
        backend = "llamacpp"
    elif probe(DEFAULT_URL["ollama"] + "/api/version") is not None:
        backend = "ollama"
    else:
        sys.exit("No model server is running. In another Terminal window start one first:\n"
                 "    bash start_ollama.sh     (Ollama)    or    bash start_server.sh    (llama.cpp)")
server = args.server or DEFAULT_URL[backend]
# output file per model and route, so that results of different models or servers never mix
if args.model_tag:
    model_tag = args.model_tag
elif backend == "llamacpp":
    model_tag = MODEL["tag"]
elif args.model == "llama3.1-8b" and not args.chat:
    model_tag = "llama31_8b_q4km_ollama"
else:
    model_tag = "".join(c if c.isalnum() else "_" for c in args.ollama_model) + "_ollama_chat"

# ---- resume: requests already answered are skipped ------------------------------------------------
out_dir = os.path.join(HERE, "outputs"); os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, model_tag + ".jsonl")
done = set()
if os.path.exists(out_path):
    with open(out_path) as f:
        for line in f:
            try:
                rec_ = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            # answers given in thinking mode (before thinking could be switched off) are redone;
            # the analysis keeps the last answer per request
            if args.no_think and ((rec_.get("reasoning_chars") or 0) > 0
                                  or re.search(r"<think>\s*\S", rec_.get("response") or "")):
                continue
            done.add(rec_["key"])
todo = [q for q in requests if q[0] not in done]

# memory size (macOS reports it through sysctl; elsewhere it is printed as nan)
try:
    mem_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"],
                                         stderr=subprocess.DEVNULL).decode()) / 2 ** 30
except Exception:  # noqa: BLE001
    mem_gb = float("nan")
print("backend {} at {}   machine {} ({}, {:.0f} GB)".format(
    backend, server, platform.machine(), platform.mac_ver()[0] or platform.system(), mem_gb), flush=True)
print("preset {}: {} items, {} requests ({} already done, {} to go) -> {}".format(
    args.preset, len(items), len(requests), len(requests) - len(todo), len(todo), out_path), flush=True)
if not todo:
    print("nothing to do"); sys.exit(0)


def post(path, payload, timeout=900):
    """POST a JSON payload to the server and return the decoded JSON answer."""
    req = urllib.request.Request(server + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ---- wait until the server has the model ready -------------------------------------------------
if backend == "llamacpp":
    for wait in range(180):              # the first start downloads and loads the model (HTTP 503 meanwhile)
        st = probe(server + "/health")
        if st == 200:
            break
        if wait == 0:
            print("waiting for llama-server to finish loading the model ...", flush=True)
        if st is None and wait >= 3:
            sys.exit("llama-server is not reachable at {}. Start it with bash start_server.sh first."
                     .format(server))
        time.sleep(10)
    else:
        sys.exit("llama-server did not become ready within 30 minutes")
    try:
        with urllib.request.urlopen(server + "/props", timeout=10) as resp:
            props = json.loads(resp.read().decode())
        n_slots = int(props.get("total_slots") or args.workers)
    except Exception:  # noqa: BLE001
        props, n_slots = {}, args.workers
    # the served model file must be the one of --model (and not the file of another model)
    served = str(props.get("model_path") or props.get("model_alias") or "")
    other = [m["file"] for k, m in MODELS.items() if m["file"] and k != args.model]
    if served and ((MODEL["file"] and MODEL["file"].lower() not in served.lower())
                   or any(o.lower() in served.lower() for o in other)):
        sys.exit("The server is running {}, not {}.\n"
                 "Restart it in window 1 with:  MODEL={} bash start_server.sh"
                 .format(os.path.basename(served), args.model, args.model))
    if args.workers > n_slots:
        print("llama-server has {} slots: using --workers {}".format(n_slots, n_slots), flush=True)
        args.workers = n_slots
else:
    with urllib.request.urlopen(server + "/api/tags", timeout=10) as resp:
        tags = [m.get("name") for m in json.loads(resp.read().decode()).get("models", [])]
    if args.ollama_model not in tags:
        sys.exit("Ollama does not have {} yet. Run:  ollama pull {}"
                 .format(args.ollama_model, args.ollama_model))
    print("loading the model in Ollama ...", flush=True)
    post("/api/generate", {"model": args.ollama_model, "prompt": "", "keep_alive": "30m",
                           "options": {"num_ctx": NUM_CTX}})


def messages(posts_text, spec):
    """System and user messages, for servers that apply the model's own chat template (--chat)."""
    sys_msg = task["system_prompt"].format(posts_text=posts_text).strip()
    usr_msg = task["question_prompt"].format(
        symptom_title=spec["title"], symptom_description="" if spec["omit"] else spec["description"],
        symptom_scale_text=scale_string(spec["scale"])).strip()
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}]


# Sampling settings of the cohort annotation (the checkpoint's defaults): temperature 0.6, top-k 50,
# top-p 0.9, nothing else (min-p 0 and repetition penalty 1 switch those samplers off).
# Ollama: only what changes its defaults.  typical_p = 1 and zero presence/frequency penalties are
# no-ops (recent Ollama versions reject typical_p outright).  Any other option a given Ollama
# version rejects is dropped at run time and recorded in every output line ("dropped_options").
OLLAMA_SAMPLING = {"temperature": 0.6, "top_k": 50, "top_p": 0.9, "min_p": 0.0, "repeat_penalty": 1.0}
# llama.cpp: the sampler chain is given explicitly, so only these three samplers act, in the order
# temperature, top-k, top-p of the transformers sampler of the cohort annotation
LLAMACPP_SAMPLING = {"temperature": 0.6, "top_k": 50, "top_p": 0.9, "min_p": 0.0, "repeat_penalty": 1.0,
                     "samplers": ["temperature", "top_k", "top_p"]}
DROPPED = set()                        # Ollama options rejected by the running version
DROP_LOCK = threading.Lock()
_REFUSAL = re.compile(r"support|unknown|invalid|unrecogni", re.I)    # wording of a rejected option


def ollama_options(seed):
    """Ollama "options" of one request: sampling, answer length, seed and context size."""
    o = {**OLLAMA_SAMPLING, "num_predict": args.max_new_tokens, "seed": seed, "num_ctx": NUM_CTX}
    return {k: v for k, v in o.items() if k not in DROPPED}


def rejected_option(body):
    """Name of the sampling option an Ollama error message refers to, if it is one we sent."""
    b = (body or "").lower()
    if not _REFUSAL.search(b):
        return None
    for name in OLLAMA_SAMPLING:
        if re.search(r"\b{}\b".format(name), b):    # already dropped by another worker: just retry
            return name
    return None


def complete_chat(msgs, seed, slot):
    """One answer from the messages, the server applying the model's chat template.

    Ollama: /api/chat (thinking switched off with "think": false when --no-think); llama.cpp:
    /v1/chat/completions on server slot `slot` (thinking off through the template's enable_thinking).
    Returns the fields recorded for every request (answer, token counts, stop type, timings, and the
    length of any reasoning text).
    """
    if backend == "ollama":
        payload = {"model": args.ollama_model, "messages": msgs, "stream": False, "keep_alive": "30m",
                   "options": ollama_options(seed)}
        if args.no_think:
            payload["think"] = False
        res = post("/api/chat", payload)
        ms = lambda k: None if res.get(k) is None else res[k] / 1e6      # noqa: E731  (ns -> ms)
        msg = res.get("message") or {}
        return {"response": msg.get("content", ""), "n_prompt_tokens": res.get("prompt_eval_count"),
                "n_output_tokens": res.get("eval_count"), "n_cached_tokens": None,
                "stop_type": res.get("done_reason"), "truncated": None, "server_model": res.get("model"),
                "prompt_ms": ms("prompt_eval_duration"), "predicted_ms": ms("eval_duration"),
                "reasoning_chars": len(msg.get("thinking") or "")}
    payload = {"messages": msgs, "max_tokens": args.max_new_tokens, **LLAMACPP_SAMPLING,
               "seed": seed, "cache_prompt": True, "id_slot": slot, "stream": False}
    if args.no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    res = post("/v1/chat/completions", payload)
    ch = (res.get("choices") or [{}])[0]
    u = res.get("usage") or {}
    msg = ch.get("message") or {}
    return {"response": msg.get("content", ""), "n_prompt_tokens": u.get("prompt_tokens"),
            "n_output_tokens": u.get("completion_tokens"), "n_cached_tokens": None,
            "stop_type": ch.get("finish_reason"), "truncated": None, "server_model": res.get("model"),
            "prompt_ms": (res.get("timings") or {}).get("prompt_ms"),
            "predicted_ms": (res.get("timings") or {}).get("predicted_ms"),
            "reasoning_chars": len(msg.get("reasoning_content") or "")}


def complete(prompt, seed, slot):
    """One completion of a rendered prompt (llama.cpp /completion on slot `slot`, or Ollama
    /api/generate in raw mode, without a template); returns the fields recorded for every request."""
    if backend == "llamacpp":
        res = post("/completion", {"prompt": prompt, "n_predict": args.max_new_tokens, **LLAMACPP_SAMPLING,
                                   "seed": seed, "cache_prompt": True, "id_slot": slot, "stream": False})
        t = res.get("timings") or {}
        return {"response": res.get("content", ""), "n_prompt_tokens": res.get("tokens_evaluated"),
                "n_output_tokens": res.get("tokens_predicted"), "n_cached_tokens": res.get("tokens_cached"),
                "stop_type": res.get("stop_type"), "truncated": res.get("truncated"),
                "server_model": res.get("model"),
                "prompt_ms": t.get("prompt_ms"), "predicted_ms": t.get("predicted_ms")}
    res = post("/api/generate", {
        "model": args.ollama_model, "prompt": prompt, "raw": True, "stream": False, "keep_alive": "30m",
        "options": ollama_options(seed)})
    ms = lambda k: None if res.get(k) is None else res[k] / 1e6      # Ollama reports nanoseconds
    return {"response": res.get("response", ""), "n_prompt_tokens": res.get("prompt_eval_count"),
            "n_output_tokens": res.get("eval_count"), "n_cached_tokens": None,
            "stop_type": res.get("done_reason"), "truncated": None, "server_model": res.get("model"),
            "prompt_ms": ms("prompt_eval_duration"), "predicted_ms": ms("eval_duration")}


# ---- run: worker threads, one server slot each ---------------------------------------------------
lock = threading.Lock()
stats = {"n": 0, "tok": 0, "fail": 0, "t0": time.time(), "think": 0}
THINK_TAG = re.compile(r"<think>\s*\S", re.I)
if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
    with open(out_path, "rb") as f:          # a run stopped mid-line: start on a fresh line
        f.seek(-1, os.SEEK_END)
        needs_nl = f.read(1) != b"\n"
    if needs_nl:
        with open(out_path, "a") as f:
            f.write("\n")
fout = open(out_path, "a")


def run_item_group(slot, group):
    """All requests of one item go to the same server slot, so the post's prefix stays cached.

    Each request gets a seed of its own (first 32 bits of the SHA-1 of its key), so answers do not
    depend on the order or parallelism of the run.  Failed requests are retried up to five times
    with increasing waits; a request that still fails is left for the next run.
    """
    for key, r, sc, rep in group:
        seed = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
        res, t0 = None, time.time()
        attempt, n_drop = 0, 0
        while attempt < 5:
            try:
                if args.chat:
                    res = complete_chat(messages(r["posts_text"], symptoms[sc]), seed, slot)
                else:
                    res = complete(render(r["posts_text"], symptoms[sc]), seed, slot)
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                name = rejected_option(body) if backend == "ollama" and e.code == 400 else None
                if name and n_drop < len(OLLAMA_SAMPLING):   # this Ollama version does not take the option
                    n_drop += 1
                    with DROP_LOCK:
                        if name not in DROPPED:
                            DROPPED.add(name)
                            print("Ollama does not accept the option '{}' ({}); continuing without it".format(
                                name, body.strip()[:120]), flush=True)
                    continue
                if 400 <= e.code < 500:              # a request the server rejects: stop and show why
                    print("the server rejected the request (HTTP {}): {}".format(e.code, body[:500]),
                          flush=True)
                    os._exit(2)
                attempt += 1                         # server error (5xx): wait and retry
                if attempt == 5:
                    print("giving up on {} ({}); it will be retried on the next run".format(key, e),
                          flush=True)
                time.sleep(5 * attempt)
                continue
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as e:
                attempt += 1
                if attempt == 5:
                    print("giving up on {} ({}); it will be retried on the next run".format(key, e),
                          flush=True)
                time.sleep(5 * attempt)
        if res is None:
            with lock:
                stats["fail"] += 1
            continue
        # one output line per request: request description, then the server's answer and counts
        rec = {"key": key, "item_id": r["item_id"], "dataset": r["dataset"], "symptom_shortcode": sc,
               "replicate": rep, "seed": seed, "backend": backend, "model_tag": model_tag,
               # "llama31_original": the Llama-3.1 template rendered by render(), with two BOS tokens
               # unless --no-double-bos; "server": the server applied the model's own template
               "double_bos": (not args.no_double_bos) and not args.chat,
               "chat_template": "server" if args.chat else "llama31_original",
               "max_new_tokens": args.max_new_tokens,
               "slot": slot, "workers": args.workers, "t_start": round(t0, 3), "t_end": round(time.time(), 3),
               "dropped_options": sorted(DROPPED) if backend == "ollama" else [],
               "model": args.model, "model_name": MODEL["name"], "no_think": bool(args.no_think)}
        rec.update(res)
        with lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            stats["n"] += 1; stats["tok"] += rec["n_output_tokens"] or 0
            if (rec.get("reasoning_chars") or 0) > 0 or THINK_TAG.search(rec.get("response") or ""):
                stats["think"] += 1
            # thinking that cannot be switched off would make the run many times longer: stop early
            if args.no_think and stats["n"] >= 8 and stats["think"] > stats["n"] // 4:
                print("The model is still thinking before it answers ({} of the first {} answers), "
                      "so the run would take many times longer. Stopping.\nUpdate the server "
                      "(brew upgrade llama.cpp, or the Ollama app) and start it again with start_server.sh."
                      .format(stats["think"], stats["n"]), flush=True)
                os._exit(3)
            if stats["n"] % 50 == 0 or stats["n"] in (5, len(todo)):
                el = time.time() - stats["t0"]
                rate = stats["n"] / el
                eta = (len(todo) - stats["n"]) / max(rate, 1e-9)
                print("{:>7,}/{:,} requests  {:5.1f} req/min  {:5.0f} tok/s  elapsed {:6.1f} min  "
                      "ETA {:5.1f} h".format(stats["n"], len(todo), 60 * rate, stats["tok"] / el, el / 60,
                                             eta / 3600), flush=True)


# requests grouped by item, in the order of `todo`; workers take the next item from a shared queue
groups = {}
for q in todo:
    groups.setdefault(q[1]["item_id"], []).append(q)
glist = list(groups.values())
it = iter(glist)
it_lock = threading.Lock()


def worker(w):
    """Worker w always uses server slot w and takes the next item from a shared queue."""
    while True:
        with it_lock:
            g = next(it, None)
        if g is None:
            return
        run_item_group(w, g)


ex = cf.ThreadPoolExecutor(max_workers=args.workers)
futures = [ex.submit(worker, w) for w in range(args.workers)]
try:
    for f in futures:
        f.result()
except KeyboardInterrupt:
    with lock:
        fout.flush()
        print("\nstopped after {} requests; run the same command again to resume".format(stats["n"]),
              flush=True)
        os._exit(1)
ex.shutdown()
fout.close()
print("finished {} requests in {:.1f} min ({} failed and will be retried on the next run): {}".format(
    stats["n"], (time.time() - stats["t0"]) / 60, stats["fail"], out_path), flush=True)
