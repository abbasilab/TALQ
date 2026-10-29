#!/usr/bin/env bash
# Local llama.cpp server (llama-server) for the reduced validation protocol, run_validation.py.
# Written for macOS (Metal GPU; the script uses BSD `stat` and the Ollama app bundle) and tested on macOS
# with an Apple M5 processor.  Start it in one terminal and keep that terminal open; run
# run_validation.py in a second terminal.
#
#   bash start_server.sh                       Llama-3.1-8B-Instruct, 4-bit (Q4_K_M): the reference model
#   MODEL=qwen3.6-35b bash start_server.sh     Qwen3.6-35B-A3B, 4-bit (UD-IQ4_XS, 17.7 GB download)
#   MODEL=qwen3.8-27b bash start_server.sh     Qwen3.8-27B, 4-bit (UD-Q4_K_M, 16.5 GB download)
#
#   PARALLEL=8 ...    parallel slots (default 4; 8 for Llama with 24 GB of memory or more, then --workers 8)
#   MODEL_DIR=...     where the Qwen model files are kept (default ~/models; the download resumes if stopped)
#   MODEL_PATH=...    use this GGUF file instead
#   CTX_PER_SLOT=...  context per slot in tokens (default 4096: longest prompt about 2,100 + 384 new tokens)
#
# Llama: llama.cpp lets the request set the sampler order (temperature, then top-k, then top-p, the order
# of the transformers sampler that generated the cohort annotation).  When Ollama is installed, no other
# install is needed: Ollama ships llama.cpp's llama-server, and the GGUF file Ollama downloaded for
# llama3.1:8b-instruct-q4_K_M is used directly (find_ollama_model.py).  Otherwise install llama.cpp
# (brew install llama.cpp); the model is then downloaded from Hugging Face.
# Qwen: the server applies the model's own chat template (--jinja) and its thinking mode is switched
# off, so that each answer is the questionnaire JSON only.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${MODEL:-llama3.1-8b}
MODEL_DIR=${MODEL_DIR:-$HOME/models}
CTX_PER_SLOT=${CTX_PER_SLOT:-4096}
OLLAMA_TAG=${OLLAMA_TAG:-llama3.1:8b-instruct-q4_K_M}
HF_FILE=""
case "$MODEL" in
  llama3.1-8b)
    DEFAULT_PARALLEL=4; QWEN=0 ;;
  qwen3.6-35b)
    HF_REPO=unsloth/Qwen3.6-35B-A3B-GGUF; HF_FILE=Qwen3.6-35B-A3B-UD-IQ4_XS.gguf; HF_BYTES=17730509792
    DEFAULT_PARALLEL=4; QWEN=1 ;;
  qwen3.8-27b)
    HF_REPO=unsloth/Qwen3.8-27B-GGUF; HF_FILE=Qwen3.8-27B-UD-Q4_K_M.gguf; HF_BYTES=16464440224
    DEFAULT_PARALLEL=4; QWEN=1 ;;
  *)
    echo "Unknown MODEL=$MODEL. Use llama3.1-8b (default), qwen3.6-35b or qwen3.8-27b."; exit 1 ;;
esac
PARALLEL=${PARALLEL:-$DEFAULT_PARALLEL}
LOG="llama_server_${MODEL}.log"
[ "$MODEL" = "llama3.1-8b" ] && LOG=llama_server.log

# ---- the llama-server binary: on the PATH, else the one inside the Ollama app ----------------------
SERVER=$(command -v llama-server || true)
if [ -z "$SERVER" ] && [ -x /Applications/Ollama.app/Contents/Resources/llama-server ]; then
  SERVER=/Applications/Ollama.app/Contents/Resources/llama-server
fi
if [ -z "$SERVER" ]; then
  echo "llama-server not found. Install Ollama (https://ollama.com/download) or: brew install llama.cpp"
  exit 1
fi
if curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
  echo "Something is already listening on port 8080 (another llama-server?)."
  echo "Stop it first (Ctrl-C in its window)."; exit 1
fi

# ---- model file -----------------------------------------------------------------------------------
if [ -n "${MODEL_PATH:-}" ]; then
  MODEL_ARGS=(--model "$MODEL_PATH")
elif [ "$QWEN" = 1 ]; then
  # download (resumable) from Hugging Face; complete when the size matches HF_BYTES
  mkdir -p "$MODEL_DIR"
  MODEL_PATH="$MODEL_DIR/$HF_FILE"
  have=0; [ -f "$MODEL_PATH" ] && have=$(stat -f %z "$MODEL_PATH")
  if [ "$have" != "$HF_BYTES" ]; then
    need_kb=$(( (HF_BYTES - have) / 1024 + 2 * 1024 * 1024 ))     # remaining download + 2 GB margin
    free_kb=$(df -k "$MODEL_DIR" | awk 'NR==2 {print $4}')
    if [ "$free_kb" -lt "$need_kb" ]; then
      echo "Not enough free disk space in $MODEL_DIR:" \
           "need about $(( need_kb / 1024 / 1024 )) GB, have $(( free_kb / 1024 / 1024 )) GB."
      echo "Free some space, or choose another folder with MODEL_DIR=/path/to/folder."; exit 1
    fi
    echo "Downloading $HF_FILE ($(( HF_BYTES / 1000000000 )) GB) from Hugging Face to $MODEL_DIR."
    echo "This takes a while; if it stops, run the same command again and it resumes."
    curl -L --fail -C - --retry 5 --retry-delay 10 --progress-bar -o "$MODEL_PATH" \
      "https://huggingface.co/$HF_REPO/resolve/main/$HF_FILE"
    have=$(stat -f %z "$MODEL_PATH")
    if [ "$have" != "$HF_BYTES" ]; then
      echo "The download is incomplete ($have of $HF_BYTES bytes). Run the same command again to resume."
      exit 1
    fi
  fi
  echo "model: $MODEL_PATH"
  MODEL_ARGS=(--model "$MODEL_PATH")
else
  # the GGUF file Ollama downloaded for $OLLAMA_TAG (manifest -> model layer -> blob)
  BLOB=$(python3 find_ollama_model.py "$OLLAMA_TAG" 2>/dev/null || true)
  if [ -n "$BLOB" ]; then
    MODEL_ARGS=(--model "$BLOB")
    echo "model: $OLLAMA_TAG, the file Ollama downloaded ($BLOB)"
  else
    MODEL_ARGS=(-hf bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M)
    echo "model: bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M (downloaded on the first start)"
  fi
fi

# ---- options supported by this llama-server version -----------------------------------------------
echo "server: $SERVER"
"$SERVER" --version 2>&1 | head -n 2 || true
HELP=$("$SERVER" --help 2>&1 || true)
EXTRA=()
# Each worker keeps its post's prefix in its own slot, so the extra host-memory prompt cache of
# recent llama-server versions (several GB by default) is not needed: switch it off where supported.
if grep -q -- "--cache-ram" <<<"$HELP"; then EXTRA+=(--cache-ram 0); fi
if [ "$QWEN" = 1 ]; then
  # the model's own chat template; thinking off (also switched off in every request)
  if grep -q -- "--jinja" <<<"$HELP"; then EXTRA+=(--jinja); fi
  if grep -q -- "--reasoning-budget" <<<"$HELP"; then EXTRA+=(--reasoning-budget 0); fi
fi

# ---- start: all layers on the GPU, PARALLEL slots of CTX_PER_SLOT tokens, local port 8080 ---------
echo "When it prints 'server is listening', start run_validation.py in a second Terminal window."
set +e
"$SERVER" "${MODEL_ARGS[@]}" --parallel "$PARALLEL" --ctx-size $((PARALLEL * CTX_PER_SLOT)) \
  --n-gpu-layers 99 --host 127.0.0.1 --port 8080 ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e
# exit status 130 = stopped with Ctrl-C; any other failure: explain the usual causes from the log
if [ "$status" != 0 ] && [ "$status" != 130 ]; then
  echo
  if grep -q -i -E "unknown model architecture|unsupported model|failed to load model|error loading model" \
       "$LOG"; then
    echo "This llama-server could not load $MODEL. It is probably too old for this model."
    echo "Install the current llama.cpp with:  brew install llama.cpp   (or: brew upgrade llama.cpp)"
    echo "and run this command again. Without Homebrew, see 'If the server cannot load Qwen' in README.md."
  elif grep -q -i -E "out of memory|failed to allocate|insufficient memory" "$LOG"; then
    echo "Not enough GPU memory for $MODEL with $PARALLEL slots. Quit other apps, or try PARALLEL=2."
  else
    echo "llama-server stopped (exit status $status); the last lines of $LOG show why."
  fi
  exit "$status"
fi
