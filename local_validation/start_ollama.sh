#!/usr/bin/env bash
# Local Ollama server for the reduced validation protocol, an alternative to start_server.sh (llama.cpp).
# Written for macOS (it can use the Ollama app's binary) and tested on macOS with an Apple M5 processor.
# Model: Llama-3.1-8B-Instruct, 4-bit (Q4_K_M), Ollama tag llama3.1:8b-instruct-q4_K_M
# (~4.9 GB, downloaded the first time); another tag with MODEL=<tag>.
#
# Quit the Ollama menu-bar app first (menu bar: Ollama > Quit Ollama). This script starts
# a server with the settings the run needs and writes its log to ollama_server.log.
#   bash start_ollama.sh               4 parallel slots (default)
#   PARALLEL=8 bash start_ollama.sh    with 24 GB of memory or more; then use --workers 8
# Answers obtained through Ollama are written to their own output file (llama31_8b_q4km_ollama.jsonl);
# the analysis of the manuscript used the llama.cpp route (start_server.sh).
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${MODEL:-llama3.1:8b-instruct-q4_K_M}

# the ollama binary: on the PATH, else the one inside the Ollama app
OLLAMA=$(command -v ollama || true)
if [ -z "$OLLAMA" ] && [ -x /Applications/Ollama.app/Contents/Resources/ollama ]; then
  OLLAMA=/Applications/Ollama.app/Contents/Resources/ollama
fi
if [ -z "$OLLAMA" ]; then
  echo "Ollama was not found. Install it from https://ollama.com/download,"
  echo "or use bash start_server.sh (llama.cpp)."
  exit 1
fi
if curl -s -m 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  echo "An Ollama server is already running (usually the menu-bar app)."
  echo "Quit it (menu bar: Ollama > Quit Ollama), then run bash start_ollama.sh again."
  exit 1
fi

export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=${PARALLEL:-4}
export OLLAMA_CONTEXT_LENGTH=4096      # tokens per request, as NUM_CTX in run_validation.py
export OLLAMA_KEEP_ALIVE=30m
# Ollama's own sampler applies top-k, temperature, then top-p.  This is equivalent to the order of the
# annotation's sampler (temperature, top-k, top-p), since the temperature does not change which tokens
# are the k most likely.
export OLLAMA_NEW_ENGINE=1

# start the server in the background, wait until it answers, then download the model if needed
"$OLLAMA" --version || true
"$OLLAMA" serve > ollama_server.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT INT TERM
for i in $(seq 1 60); do
  curl -s -m 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
  sleep 1
done
"$OLLAMA" pull "$MODEL"
echo
echo "Ollama is ready with $MODEL ($OLLAMA_NUM_PARALLEL parallel slots)."
echo "Keep this window open and start run_validation.py in a second Terminal window."
echo "(Server log: local_validation/ollama_server.log. Ctrl-C here stops the server.)"
wait $PID
