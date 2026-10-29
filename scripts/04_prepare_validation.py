#!/usr/bin/env python3
"""
Step 4 - human-annotated Reddit datasets for the validation (Methods, "Validation against human
annotations"; Fig. 1d, Extended Data Fig. 3; Supplementary Table 2).

Datasets (cloned from their public GitHub repositories with `git clone --depth 1` into
data/external/; the commit of each clone is printed):
  irf       Garg et al. 2023, interpersonal risk factors: thwarted belongingness (tbe) and perceived
            burdensomeness (pbu), human labels; 3,522 posts.  github.com/drmuskangarg/Irf
  lost      Garg et al. 2023, LoST, low self-esteem; 3,250 posts with text and label.
            github.com/drmuskangarg/LoST
  severity  Naseem et al. 2022, depression severity on four levels (minimum, mild, moderate,
            severe -> 0-3); 3,553 posts.  github.com/usmaann/Depression_Severity_Dataset
  depemo    Rahman et al. 2024, DepressionEmo, eight depression-related emotions; 6,037 posts.  Its
            labels come from a majority vote of zero-shot models checked by humans, so they are
            treated as weak labels (secondary; not part of the reduced protocol).
            github.com/abuBakarSiddiqurRahman/DepressionEmo
  redsm5    (optional) Bao et al. 2025, ReDSM5, posts annotated at the sentence level by a licensed
            psychologist for the nine DSM-5 criterion A symptoms; gated (agreement form,
            huggingface.co/datasets/irlab-udc/redsm5), used only if
            data/external/ReDSM5/redsm5_{posts,annotations}.csv exist.  A post is labeled with a
            symptom if any of its sentences is annotated as present (status 1).
The manuscript reports the first three datasets.  Every text is rendered like a window with a single
utterance ("'<text>'", slm.windows.posts_text), so the prompt is the one used for the cohort
windows.

The pre-specified mapping of labels to battery symptoms (with the control symptoms of the reduced
protocol) is written to config/validation_mapping.yaml.  It was specified before any model output on
these posts existed, and its SHA-256 is recorded in config/validation_mapping.record.json and
checked by step 14b; the dictionary below is written byte-identical to that file.

Inputs
  the four public repositories (cloned on first run); data/external/ReDSM5/ (optional)
Outputs
  annotation/requests/validation_<name>.parquet   item_id, dataset, posts_text (one file per dataset)
  data/processed/validation_labels.parquet         item_id, dataset and the label columns
  config/validation_mapping.yaml                   pre-specified label -> symptom mapping

Usage (from scripts/):
  python3 04_prepare_validation.py
"""
import argparse
import subprocess

import _setup  # noqa: F401
import pandas as pd
import yaml

from slm import params as P
from slm.windows import posts_text

# --help prints this description; the step has no options
argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

# ---- step: clone the public datasets (once) and print their commits ----
EXT = P.DATA / "external"
REPOS = {"Irf": "drmuskangarg/Irf", "LoST": "drmuskangarg/LoST",
         "Depression_Severity_Dataset": "usmaann/Depression_Severity_Dataset",
         "DepressionEmo": "abuBakarSiddiqurRahman/DepressionEmo"}
for d, r in REPOS.items():
    if not (EXT / d).exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", f"https://github.com/{r}", str(EXT / d)],
                       check=True)
    rev = subprocess.run(["git", "-C", str(EXT / d), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(d, rev)

# ---- step: one request row and one label row per post ----
REDSM5_SYMPTOMS = ["DEPRESSED_MOOD", "ANHEDONIA", "APPETITE_CHANGE", "SLEEP_ISSUES", "PSYCHOMOTOR",
                   "FATIGUE", "WORTHLESSNESS", "COGNITIVE_ISSUES", "SUICIDAL_THOUGHTS"]
rows, labels = [], []
# IRF: train/val/test files; item id = split + the row number stored in the file's unnamed column
irf = pd.concat([pd.read_csv(EXT / "Irf" / f"{s}_data.csv").assign(split=s)
                 for s in ["train", "val", "test"]], ignore_index=True)
for i, r in irf.iterrows():
    iid = f"irf_{r['split']}_{r['Unnamed: 0']}"
    rows.append({"item_id": iid, "dataset": "irf", "posts_text": posts_text([str(r["text"])])})
    labels.append({"item_id": iid, "dataset": "irf", "tbe": int(r["belong"]), "pbu": int(r["burden"])})

# LoST: posts without text or label are dropped; item id = row number after dropping
lost = pd.read_csv(EXT / "LoST" / "LoSTdataset.csv").dropna(subset=["text", "self"]).reset_index(drop=True)
for i, r in lost.iterrows():
    iid = f"lost_{i}"
    rows.append({"item_id": iid, "dataset": "lost", "posts_text": posts_text([str(r["text"])])})
    labels.append({"item_id": iid, "dataset": "lost", "low_self_esteem": int(r["self"])})

# depression severity: four ordered levels coded 0-3
sev = pd.read_csv(EXT / "Depression_Severity_Dataset" / "Reddit_depression_dataset.csv")
lev = {"minimum": 0, "mild": 1, "moderate": 2, "severe": 3}
for i, r in sev.iterrows():
    iid = f"severity_{i}"
    rows.append({"item_id": iid, "dataset": "severity", "posts_text": posts_text([str(r["text"])])})
    labels.append({"item_id": iid, "dataset": "severity", "severity": lev[r["label"]]})

# DepressionEmo: one binary column per emotion, named emo_<first word of the emotion>
emo_names = ["anger", "brain dysfunction (forget)", "emptiness", "hopelessness", "loneliness",
             "sadness", "suicide intent", "worthlessness"]
for split in ["train", "val", "test"]:
    de = pd.read_json(EXT / "DepressionEmo" / "Dataset" / f"{split}.json", lines=True)
    for _, r in de.iterrows():
        iid = f"depemo_{r['id']}"
        rows.append({"item_id": iid, "dataset": "depemo", "posts_text": posts_text([str(r["text"])])})
        lab = {"item_id": iid, "dataset": "depemo"}
        for e in emo_names:
            lab["emo_" + e.split(" ")[0]] = int(e in r["emotions"])
        labels.append(lab)

# ReDSM5 (optional): post-level label = any sentence annotated present (status 1) for the symptom
redsm5 = EXT / "ReDSM5"
if (redsm5 / "redsm5_posts.csv").exists() and (redsm5 / "redsm5_annotations.csv").exists():
    rp = pd.read_csv(redsm5 / "redsm5_posts.csv")
    ra = pd.read_csv(redsm5 / "redsm5_annotations.csv")
    pos = ra[ra["status"] == 1].groupby(["post_id", "DSM5_symptom"]).size().unstack(fill_value=0) > 0
    for _, r in rp.iterrows():
        iid = f"redsm5_{r['post_id']}"
        rows.append({"item_id": iid, "dataset": "redsm5", "posts_text": posts_text([str(r["text"])])})
        lab = {"item_id": iid, "dataset": "redsm5"}
        for c in REDSM5_SYMPTOMS:
            in_table = r["post_id"] in pos.index and c in pos     # post and symptom occur in `pos`
            lab["dsm_" + c.lower()] = int(bool(pos.loc[r["post_id"], c])) if in_table else 0
        labels.append(lab)
    print("ReDSM5:", len(rp), "posts")
else:
    print("ReDSM5 not found (optional; gated dataset) - skipped")

# ---- step: write the request and label tables (first row of a duplicated item id kept) ----
req = pd.DataFrame(rows).drop_duplicates("item_id")
lab = pd.DataFrame(labels).drop_duplicates("item_id")
out = P.ROOT / "annotation" / "requests"
for name, g in req.groupby("dataset"):
    g.reset_index(drop=True).to_parquet(out / f"validation_{name}.parquet", index=False)
    print(name, len(g))
lab.to_parquet(P.PROCESSED / "validation_labels.parquet", index=False)

# ---- step: the pre-specified mapping (config/validation_mapping.yaml) ----
# label -> battery symptoms whose mean is the score of the label ("inverse_symptoms" are reversed;
# "aggregate: max" takes the maximum instead); "factor" names the related factor.  The text and
# structure below are those of the pre-specified file and must stay unchanged, because its SHA-256
# was recorded before any model output on these posts existed.  The control lists of irf and lost
# are one shared object, which yaml writes as an anchor and an alias, exactly as in that file.
_controls = ["insomnia_early", "appetite_decrease", "gi_dysfunction"]   # pre-specified controls
mapping = {
    "irf": {"tbe": {"symptoms": ["loneliness", "others_dislike", "others_unfriendly", "ip_sensitivity"],
                    "factor": "Social strain", "direction": 1},
            "pbu": {"symptoms": ["worthlessness", "self_punishment", "guilt", "si"],
                    "inverse_symptoms": ["feeling_needed"], "factor": "Negative self-reflection",
                    "direction": 1}},
    "lost": {"low_self_esteem": {"symptoms": ["worthlessness", "self_dislike", "self_criticalness",
                                              "self_failure"], "inverse_symptoms": ["feeling_good"],
                                 "factor": "Negative self-reflection", "direction": 1}},
    "severity": {"severity": {"symptoms": ["sadness", "interest_loss", "pleasure_loss",
                                           "appetite_decrease", "appetite_increase",
                                           "weight_decrease", "weight_increase",
                                           "insomnia_early", "insomnia_middle", "insomnia_late",
                                           "hypersomnia", "agitation", "retardation", "fatigue",
                                           "energy_loss", "worthlessness", "guilt",
                                           "concentration_dysfunction", "indiciciveness", "si"],
                              "note": "DSM-5 MDD criterion A items; composite = mean", "ordinal": True}},
    "depemo": {"emo_sadness": {"symptoms": ["sadness"]}, "emo_hopelessness": {"symptoms": ["pessimism"]},
               "emo_loneliness": {"symptoms": ["loneliness"]},
               "emo_worthlessness": {"symptoms": ["worthlessness"]},
               "emo_emptiness": {"symptoms": ["feeling_inability"]}, "emo_suicide": {"symptoms": ["si"]},
               "emo_anger": {"symptoms": ["irritability"]},
               "emo_brain": {"symptoms": ["concentration_dysfunction"]}},
    # ReDSM5: DSM-5 criterion A symptoms; a criterion is met by any of its listed
    # manifestations, so the composite is the maximum over the mapped items.
    "redsm5": {
        "dsm_depressed_mood": {"symptoms": ["sadness", "pessimism", "feeling_inability", "crying"],
                               "aggregate": "max"},
        "dsm_anhedonia": {"symptoms": ["interest_loss", "pleasure_loss"], "aggregate": "max"},
        "dsm_appetite_change": {"symptoms": ["appetite_decrease", "appetite_increase", "weight_decrease",
                                             "weight_increase"], "aggregate": "max"},
        "dsm_sleep_issues": {"symptoms": ["insomnia_early", "insomnia_middle", "insomnia_late",
                                          "hypersomnia"],
                             "aggregate": "max", "factor": "Sleep disturbance"},
        "dsm_psychomotor": {"symptoms": ["agitation", "retardation"], "aggregate": "max"},
        "dsm_fatigue": {"symptoms": ["fatigue", "energy_loss"], "aggregate": "max"},
        "dsm_worthlessness": {"symptoms": ["worthlessness", "guilt"], "aggregate": "max",
                              "factor": "Negative self-reflection"},
        "dsm_cognitive_issues": {"symptoms": ["concentration_dysfunction", "indiciciveness"],
                                 "aggregate": "max"},
        "dsm_suicidal_thoughts": {"symptoms": ["si"], "aggregate": "max", "factor": "Self-harm tendency"},
    },
    # the reduced protocol of the local validation: mapped symptoms plus three control symptoms
    # without a conceptual link to the labels (the note is kept verbatim, see above)
    "_reduced_protocol": {
        "note": ("Reduced on-device protocol (Apple-silicon laptop, llama.cpp): only the symptoms mapped "
                 "above are scored, plus three pre-specified control symptoms for the IRF and LoST "
                 "datasets that are not conceptually related to the labels (discriminant check). "
                 "Severity uses its 20 criterion-A items. Factor-score validation (needs all 52 items) "
                 "is not part of the reduced protocol."),
        "controls": {"irf": _controls, "lost": _controls, "severity": []},
    },
}
with open(P.CONFIG / "validation_mapping.yaml", "w") as f:
    yaml.safe_dump(mapping, f, sort_keys=False)
print("labels:", lab.groupby("dataset").size().to_dict())
