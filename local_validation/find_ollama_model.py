#!/usr/bin/env python3
"""Print the path of the GGUF file that Ollama downloaded for a model tag (or nothing).

Used by start_server.sh to serve, with llama.cpp, the same model file that Ollama distributes, without
downloading it again.  Ollama keeps a manifest per tag (<models>/manifests/registry.ollama.ai/library/
<name>/<version>); its layer with a media type ending in ".model" names the blob (sha256-<digest>) that
holds the GGUF weights.  The models folder is $OLLAMA_MODELS, by default ~/.ollama/models.

usage: python3 find_ollama_model.py [tag]      (default tag: llama3.1:8b-instruct-q4_K_M)
"""
import json
import os
import sys

tag = sys.argv[1] if len(sys.argv) > 1 else "llama3.1:8b-instruct-q4_K_M"
name, _, ver = tag.partition(":")                    # "llama3.1", "8b-instruct-q4_K_M"
root = os.path.expanduser(os.environ.get("OLLAMA_MODELS", "~/.ollama/models"))
man = os.path.join(root, "manifests", "registry.ollama.ai", "library", name, ver or "latest")
try:
    layers = json.load(open(man)).get("layers", [])
except (OSError, ValueError):                        # no such tag downloaded: print nothing
    sys.exit(0)
for layer in layers:
    if layer.get("mediaType", "").endswith(".model"):
        path = os.path.join(root, "blobs", layer["digest"].replace(":", "-"))
        if os.path.exists(path):
            print(path)
        break
