#!/usr/bin/env python3
"""
Download the two ConvoKit subreddit corpora analyzed in the manuscript (r/depression and r/Anxiety;
Chang et al. 2020, ConvoKit) into this folder (Methods, "Data source" and "Data availability").

Run it from this folder, on a computer with unrestricted internet access:

    cd data/raw
    python3 download_convokit.py

Only the Python standard library is used.  The script resolves the download URL the same way as
ConvoKit's own `convokit.download("subreddit-<name>")` (through the server's
subreddit-groupings.txt), downloads each zip with a progress read-out (resuming a partial file if
one exists), checks that the zip opens and contains `utterances.jsonl`, and writes
`download_manifest.json` (URL, size in bytes, SHA-256 and archive members), which documents the
exact files used.  The manifest distributed with the code lists the checksums of the archives
analyzed in the manuscript; running the script rewrites it, so a new download can be compared with
the distributed version (for example with `git diff download_manifest.json`).

If the download fails, the URLs are printed so that the files can be fetched with a browser or with
`curl -L -C - -O <url>` instead.

Outputs (in data/raw/)
  depression.corpus.zip, Anxiety.corpus.zip, download_manifest.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import zipfile

BASE = "https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/"
SUBREDDITS = ["depression", "Anxiety"]
CHUNK = 1 << 20  # 1 MiB


def fetch_groupings():
    """
    Read the server's list of subreddit groupings (subreddit-groupings.txt).

    Returns:
        list of non-empty lines, each either a subreddit name or a range "first~-~last" of the
        names stored together in one folder.
    """
    url = BASE + "subreddit-groupings.txt"
    with urllib.request.urlopen(url, timeout=120) as r:
        text = r.read().decode("utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def find_group(name, groups):
    """
    Grouping that holds subreddit `name`: same rule as convokit.util.get_subreddit_info (the first
    grouping that equals the name or whose range contains it in string order).

    Returns:
        the grouping string, or None if no grouping matches.
    """
    for g in groups:
        bounds = g.split("~-~")
        if len(bounds) == 1:
            if bounds[0] == name:
                return g
        elif bounds[0] <= name <= bounds[1]:
            return g
    return None


def download(url, dest):
    """
    Download `url` to the file `dest`, resuming a partial download when the server allows it.

    An existing partial file is continued with an HTTP Range request; if the server answers with
    the whole file (status 200 instead of 206), the file is written from the start.  Progress is
    printed in MB.
    """
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=300) as r:
        status = getattr(r, "status", 200)
        total = r.headers.get("Content-Length")
        total = int(total) + (have if status == 206 else 0) if total else None
        mode = "ab" if (have and status == 206) else "wb"     # append only to a partial content reply
        if mode == "wb":
            have = 0
        t0 = time.time()
        with open(dest, mode) as f:
            while True:
                buf = r.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
                have += len(buf)
                if total:
                    pct = 100.0 * have / total
                    rate = have / max(time.time() - t0, 1e-6) / 1e6
                    sys.stdout.write(f"\r  {have/1e6:9.1f} / {total/1e6:9.1f} MB ({pct:5.1f}%)"
                                     f"  {rate:5.1f} MB/s")
                else:
                    sys.stdout.write(f"\r  {have/1e6:9.1f} MB")
                sys.stdout.flush()
    print()


def sha256(path):
    """SHA-256 hex digest of a file, read in chunks of 1 MiB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(CHUNK), b""):
            h.update(buf)
    return h.hexdigest()


def main():
    """Download both corpora into the folder of this script and write download_manifest.json."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    print("Fetching ConvoKit subreddit groupings ...")
    groups = fetch_groupings()
    manifest = {"source": BASE, "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "files": []}
    for name in SUBREDDITS:
        g = find_group(name, groups)
        if g is None:
            print(f"!! Could not locate a grouping for r/{name}; please download manually via ConvoKit.")
            continue
        url = f"{BASE}corpus-zipped/{g}/{name}.corpus.zip"
        dest = f"{name}.corpus.zip"
        print(f"r/{name}: {url}")
        try:
            download(url, dest)
        except Exception as e:  # noqa: BLE001
            print(f"!! Download failed ({e}). Fetch manually:\n   curl -L -C - -o {dest} '{url}'")
            continue
        with zipfile.ZipFile(dest) as z:
            names = z.namelist()
        utt = [n for n in names if n.endswith("utterances.jsonl")]
        print(f"  zip OK: {len(names)} members; utterances file: {utt[:1]}")
        manifest["files"].append({"subreddit": name, "url": url, "file": dest,
                                  "bytes": os.path.getsize(dest), "sha256": sha256(dest),
                                  "members": names})
    with open("download_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Done. Wrote download_manifest.json")


if __name__ == "__main__":
    main()
