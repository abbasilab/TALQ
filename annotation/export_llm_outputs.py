#!/usr/bin/env python3
"""
Export the stored model outputs of annotation jobs from the MongoDB job database to Parquet files.

The annotation jobs of the study (the symptom battery, P.SYMPTOM_JOB, and the BDI-II / PHQ-9
inventories, P.INVENTORY_JOB in src/slm/params.py) were run by a job queue that wrote one document
per request (window, item, sample) to MongoDB.  This script streams the output collection of each
requested job and writes it in chunks of compressed Parquet, which the analysis reads from
data/llm_outputs/ (step 5 for the symptom job, step 18 for the inventory job).

WHERE TO RUN
    On a machine that can reach the MongoDB server, with pymongo, pandas and pyarrow installed.
    Connection settings and credentials are read only from environment variables:
        MONGO_HOST (default localhost), MONGO_PORT (default 27017), MONGO_USER, MONGO_PASS,
        MONGO_TLS_CA_FILE, MONGO_TLS_CLIENT_FILE, MONGO_TLS_CLIENT_FILE_PASS (TLS, optional),
        MONGO_DB (database of the job queue, default redaf_langjobs; or --db).
    If the job queue's Python package (langjobs) can be imported, it is used to find the output
    collection of a job and its request status counts; otherwise collections are matched by name.

    python3 export_llm_outputs.py --list                 # jobs (or collections) in the database
    python3 export_llm_outputs.py --out llm_export \\
        --job-id 5976943c-4594-46b5-a71a-f1809cf1bfb3 --job-id 7f390e2d-314e-4023-b3af-0800aa9ecc1a

    Copy the resulting folder to data/llm_outputs/ of the repository, so that the parts are found
    at data/llm_outputs/llm_export/<job_id>/ (P.LLM_EXPORT).

WHAT IT WRITES (in --out)
    <job_id>/part-00000.parquet ...   one row per stored request; the large fields `posts`,
                                      `posts_text` and `prompt` are dropped (steps 1-3 rebuild the
                                      posts of every window from the corpus); the model's `response`
                                      is kept verbatim.  By default each row also gets
                                      `posts_speaker_id` (author of the window's first post) and
                                      `n_posts`, because some jobs store a placeholder speaker_id.
    <job_id>/sample_full_docs.jsonl   a random sample of complete documents, prompt and posts
                                      included, for checks of the rebuilt windows and prompts
                                      (step 18 compares their post text with the rebuilt windows)
    <job_id>/collections.json         output collection, matching collections with their sizes,
                                      field names, request status counts, job metadata
    manifest.json                     every written file with its row count and SHA-256
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

DEFAULT_JOBS = ["5976943c-4594-46b5-a71a-f1809cf1bfb3"]   # the symptom job (P.SYMPTOM_JOB)
BIG_FIELDS = ["posts", "posts_text", "prompt"]             # large text fields left out of the parts


# --------------------------------------------------------------------------- #
# Connection helpers
# --------------------------------------------------------------------------- #

def get_jobs_db(db_name):
    """Return the pymongo Database `db_name` of the job queue.

    Host, port, user name, password and the optional TLS files are taken only from the environment
    variables listed in the module docstring; nothing is read from files in the working directory.
    """
    from pymongo import MongoClient

    env = os.environ
    kw = dict(port=int(env.get("MONGO_PORT", 27017)),
              username=env.get("MONGO_USER"), password=env.get("MONGO_PASS"))
    if "MONGO_TLS_CA_FILE" in env:
        kw.update(tls=True, tlsCAFile=env["MONGO_TLS_CA_FILE"],
                  tlsCertificateKeyFile=(os.path.expanduser(env["MONGO_TLS_CLIENT_FILE"])
                                         if "MONGO_TLS_CLIENT_FILE" in env else None),
                  tlsCertificateKeyFilePassword=env.get("MONGO_TLS_CLIENT_FILE_PASS"))
    client = MongoClient(env.get("MONGO_HOST", "localhost"), **kw)
    print(f"[info] jobs database = {db_name}")
    return client[db_name]


def find_output_collection(db, job_id):
    """Return (output collection of the job, sorted names of all collections that mention the job id).

    The job queue's own lookup (langjobs.Job(...).output_collection) is used when the package can be
    imported.  Otherwise the output collection is the collection whose name contains the job id
    (with or without hyphens) and starts with "output"; if there are several, the largest.
    """
    names = db.list_collection_names()
    keys = {job_id, job_id.replace("-", "")}
    matching = sorted(n for n in names if any(k in n for k in keys))
    # 1) ask the job queue itself, if importable
    try:
        import langjobs

        job = langjobs.Job(None, db, id=job_id)
        coll = job.output_collection
        print(f"[info] langjobs output collection: {coll.name}")
        return coll, matching
    except Exception as e:  # noqa: BLE001
        print(f"[warn] langjobs lookup failed ({type(e).__name__}: {e}); matching by name")
    # 2) match by name
    outs = [n for n in matching if n.lower().startswith("output")]
    if not outs:
        raise SystemExit(f"No output collection found for job {job_id}. Matching collections: {matching}")
    if len(outs) > 1:
        print(f"[warn] several output-like collections {outs}; using the largest")
        outs.sort(key=lambda n: db[n].estimated_document_count(), reverse=True)
    return db[outs[0]], matching


# --------------------------------------------------------------------------- #
# Writing helpers
# --------------------------------------------------------------------------- #

def _clean(v):
    """Make a MongoDB value storable in Parquet: ObjectId -> str, dict/list/tuple -> JSON string."""
    try:
        from bson import ObjectId
    except Exception:  # noqa: BLE001
        ObjectId = ()  # type: ignore
    if isinstance(v, ObjectId):
        return str(v)
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, default=str)
    return v


def write_chunk(rows, path_stem):
    """Write a list of documents to `<path_stem>.parquet` (zstd, else gzip compression; csv.gz if
    Parquet cannot be written).  Returns (path written, number of rows)."""
    import pandas as pd

    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(_clean)
    try:
        path = Path(f"{path_stem}.parquet")
        df.to_parquet(path, compression="zstd", index=False)
    except Exception as e:  # noqa: BLE001  (no pyarrow / zstd)
        try:
            path = Path(f"{path_stem}.parquet")
            df.to_parquet(path, compression="gzip", index=False)
        except Exception:  # noqa: BLE001
            print(f"[warn] parquet unavailable ({e}); writing csv.gz")
            path = Path(f"{path_stem}.csv.gz")
            df.to_csv(path, index=False, compression="gzip")
    return path, len(df)


def sha256(path, chunk=1 << 20):
    """SHA-256 hex digest of a file, read in chunks of 1 MiB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Main export
# --------------------------------------------------------------------------- #

def export_job(db, job_id, out, chunk_rows, n_sample, speaker_from_posts=True):
    """Export one job to out/<job_id>/ (parts, sample of complete documents, collections.json).

    Arguments: db (pymongo Database), job_id (str), out (Path), chunk_rows (rows per Parquet part),
    n_sample (complete documents to sample), speaker_from_posts (add posts_speaker_id and n_posts).
    Returns a list of (path, number of rows) of the parts written.
    """
    coll, matching = find_output_collection(db, job_id)
    jdir = out / job_id
    jdir.mkdir(parents=True, exist_ok=True)

    info = {"job_id": job_id, "output_collection": coll.name, "matching_collections": {}}
    for n in matching:
        try:
            info["matching_collections"][n] = db[n].estimated_document_count()
        except Exception as e:  # noqa: BLE001
            info["matching_collections"][n] = f"count failed: {e}"
    one = coll.find_one()
    info["output_fields"] = sorted(one.keys()) if one else []

    # request status counts (when the job queue's package is available)
    try:
        import langjobs

        job = langjobs.Job(None, db, id=job_id)
        st = job.request_status_table
        info["request_status_counts"] = {str(langjobs.RequestStatus(k)): int(v)
                                         for k, v in st.groupby("status_code").size().items()}
    except Exception as e:  # noqa: BLE001
        info["request_status_counts"] = f"unavailable: {e}"

    # job metadata (when the job queue's package is available)
    try:
        import langjobs

        for j in langjobs.Job.list(db):
            if j.get("job_id") == job_id:
                info["job_metadata"] = {k: str(v) for k, v in j.items()}
    except Exception:  # noqa: BLE001
        pass

    # a random sample of complete documents (prompt and posts included) for checks of the rebuilt
    # windows and prompts ($sample spreads the check over many windows; fall back to the first documents)
    with open(jdir / "sample_full_docs.jsonl", "w") as f:
        try:
            docs = coll.aggregate([{"$sample": {"size": n_sample}}], allowDiskUse=True)
            n_written = 0
            for doc in docs:
                f.write(json.dumps(doc, default=str) + "\n"); n_written += 1
            if n_written == 0:
                raise RuntimeError("empty $sample")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] $sample failed ({e}); using the first {n_sample} documents")
            for doc in coll.find({}).limit(n_sample):
                f.write(json.dumps(doc, default=str) + "\n")

    # streamed, chunked export of everything except the large text fields
    total = coll.estimated_document_count()
    print(f"[info] exporting ~{total:,} documents from {coll.name} ...")
    proj = {k: 0 for k in BIG_FIELDS}
    if speaker_from_posts:
        # Some jobs store a placeholder speaker_id (the inventory job, 7f390e2d-..., has the same value
        # in every document); the posts carry the author.  Keep the first post's speaker_id and the
        # number of posts while still dropping the large text fields.
        pipeline = [{"$addFields": {"posts_speaker_id": {"$arrayElemAt": ["$posts.speaker_id", 0]},
                                    "n_posts": {"$size": {"$ifNull": ["$posts", []]}}}},
                    {"$project": proj}]
        cursor = coll.aggregate(pipeline, allowDiskUse=True, batchSize=5000)
    else:
        cursor = coll.find({}, projection=proj, batch_size=5000, no_cursor_timeout=True)
    rows, files, i_part, n_done, t0 = [], [], 0, 0, time.time()
    try:
        for doc in cursor:
            doc["_id"] = str(doc.get("_id"))
            rows.append(doc)
            if len(rows) >= chunk_rows:
                p, n = write_chunk(rows, jdir / f"part-{i_part:05d}")
                files.append((p, n)); i_part += 1; n_done += n; rows = []
                el = time.time() - t0
                print(f"  {n_done:>10,} / {total:,} rows  ({el/60:5.1f} min)")
        if rows:
            p, n = write_chunk(rows, jdir / f"part-{i_part:05d}")
            files.append((p, n)); n_done += n
    finally:
        cursor.close()
    info["rows_exported"] = n_done
    with open(jdir / "collections.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"[info] job {job_id}: {n_done:,} rows in {len(files)} files")
    return files


def main():
    """Parse the arguments, connect, then list the jobs or export every requested job and the manifest."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="llm_export", help="output directory")
    ap.add_argument("--job-id", action="append", dest="job_ids",
                    help="job id to export (repeatable; default: the symptom job)")
    ap.add_argument("--db", default=os.environ.get("MONGO_DB", "redaf_langjobs"),
                    help="name of the job queue's database (default: $MONGO_DB, else redaf_langjobs)")
    ap.add_argument("--chunk-rows", type=int, default=250_000, help="rows per Parquet part")
    ap.add_argument("--n-sample", type=int, default=300,
                    help="complete documents written to sample_full_docs.jsonl per job")
    ap.add_argument("--list", action="store_true", help="list jobs in the database and exit")
    ap.add_argument("--no-speaker-from-posts", action="store_true",
                    help="do not add posts_speaker_id / n_posts (plain find() export)")
    args = ap.parse_args()

    db = get_jobs_db(args.db)
    if args.list:
        try:
            import langjobs

            for j in langjobs.Job.list(db):
                print(j)
        except Exception:  # noqa: BLE001
            for n in sorted(db.list_collection_names()):
                print(n, db[n].estimated_document_count())
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "jobs": {}}
    for job_id in (args.job_ids or DEFAULT_JOBS):
        files = export_job(db, job_id, out, args.chunk_rows, args.n_sample,
                           speaker_from_posts=not args.no_speaker_from_posts)
        manifest["jobs"][job_id] = [{"file": str(p.relative_to(out)), "rows": n, "sha256": sha256(p)}
                                    for p, n in files]
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done. Copy '{out}/' to data/llm_outputs/ of the repository.")


if __name__ == "__main__":
    main()
