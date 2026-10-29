"""
Symptom battery and prompt construction for the language model questionnaire (Methods, "Language
model questionnaire"; Supplementary Note 1; Supplementary Table 1).

The battery (config/symptoms.yaml) lists the 52 symptoms compiled by Fried.  Each entry has a
number, a shortcode (used as column name everywhere), a display title, a short title, a
description, a flag `omit` and a rubric (`scale`).  The task file (config/task_symptom.yaml) holds
two templates:

    system_prompt    casts the model as a mental health professional and contains {posts_text},
                     the utterances of the window (slm.windows.posts_text);
    question_prompt  the questionnaire item: {symptom_title}, {symptom_description} (left empty for
                     the 12 symptoms with omit: True, so 40 of 52 symptoms have a description) and
                     {symptom_scale_text}, the rubric with one line per anchor ("- 0: Symptom is
                     entirely absent", "- 100: Symptom is present every day"); it ends with the
                     request for a JSON object with the fields reasoning, rating and confidence.

chat_messages() returns the system and user message of one request.  The annotation job passed
these two messages to the model's chat template with a generation prompt appended; the rendered
text, which already begins with the begin-of-text (BOS) token, was then tokenized with special
tokens, so each prompt began with two BOS tokens.  The GPU runner (annotation/run_vllm_annotation.py)
and the local validation runner (local_validation/run_validation.py) build the same messages with
their own copies of this code, because they run without this package, and handle the
tokenization (their double-BOS options).

Symptom title shown to the model: the annotation job showed item 17 (indecisiveness) under the
misspelled title "Indiciciveness" (config/symptoms_job_overrides.yaml).  load_symptoms() keeps both
spellings: `title` for tables and figures, `prompt_title` for prompts.  slm.parse maps either
spelling to the shortcode, so that prompts and stored model outputs match.

Used by most analysis scripts (load_symptoms gives the battery order of the symptom columns), for
example `python3 05_parse_collate.py --run llama31_8b`.
"""

from __future__ import annotations

from functools import lru_cache

import yaml

from . import params as P


@lru_cache(maxsize=None)
def load_symptoms() -> list[dict]:
    """
    The symptom battery in battery order (config/symptoms.yaml), read once and cached.

    Returns:
        list of 52 dicts with the keys number, shortcode, title, title_short, omit, description,
        scale, plus prompt_title: the title shown to the model, taken from
        config/symptoms_job_overrides.yaml where an entry exists (one item, 'Indiciciveness') and
        equal to `title` otherwise.  The list is shared between callers; do not modify it.
    """
    with open(P.CONFIG / "symptoms.yaml") as f:
        syms = yaml.safe_load(f)
    ov_path = P.CONFIG / "symptoms_job_overrides.yaml"
    ov = yaml.safe_load(open(ov_path)) if ov_path.exists() else {}
    for s in syms:
        s["prompt_title"] = (ov or {}).get(s["shortcode"], {}).get("prompt_title", s["title"])
    return syms


@lru_cache(maxsize=None)
def load_task(name: str = "symptom") -> dict:
    """
    Prompt templates of a task (config/task_<name>.yaml), read once and cached.

    Returns:
        dict with the keys system_prompt and question_prompt.
    """
    with open(P.CONFIG / f"task_{name}.yaml") as f:
        return yaml.safe_load(f)


def symptom_desc(spec: dict) -> str:
    """Description inserted into the prompt: empty for the 12 symptoms marked omit: True."""
    return "" if spec["omit"] else spec["description"]


def scale_string(scale: list[dict]) -> str:
    """
    Rubric text: one line "- <value>: <anchor>" per entry of the battery's `scale` list.

    Example: [{0: "Symptom is entirely absent"}, {100: "Symptom is present every day"}] gives
    "- 0: Symptom is entirely absent\\n- 100: Symptom is present every day".
    """
    return "\n".join([f"- {list(x)[0]}: {x[list(x)[0]]}" for x in scale])


def symptom_fields(spec: dict) -> dict:
    """
    Values of the symptom placeholders of question_prompt for one battery entry.

    The title is the title shown to the model (`prompt_title` when present).
    """
    return {
        "symptom_title": spec.get("prompt_title", spec["title"]),
        "symptom_description": symptom_desc(spec),
        "symptom_scale_text": scale_string(spec["scale"]),
    }


def render(template: str, **kw) -> str:
    """
    Fill a template with str.format.

    Only the template's own {placeholders} are parsed; braces inside the inserted values (for
    example in posts) are copied unchanged.
    """
    return template.format(**kw)


def chat_messages(posts_text: str, spec: dict, task: dict | None = None) -> list[dict]:
    """
    System and user message of one request (one window and one symptom).

    Args:
        posts_text: text of the window (slm.windows.posts_text).
        spec: one entry of load_symptoms().
        task: prompt templates (default: load_task("symptom")).

    Returns:
        [{"role": "system", "content": ...}, {"role": "user", "content": ...}], to be passed to the
        model's chat template with a generation prompt appended.
    """
    task = task or load_task("symptom")
    sys_msg = render(task["system_prompt"], posts_text=posts_text)
    usr_msg = render(task["question_prompt"], **symptom_fields(spec))
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}]


def merge_system_into_user(messages: list[dict]) -> list[dict]:
    """
    Merge a leading system message into the first user message.

    For chat templates without a system role: the user message becomes
    system content + blank line + user content.  Messages without a leading system message are
    returned unchanged.
    """
    if messages and messages[0]["role"] == "system":
        sys_c = messages[0]["content"]
        rest = [dict(m) for m in messages[1:]]
        if rest and rest[0]["role"] == "user":
            rest[0]["content"] = sys_c + "\n\n" + rest[0]["content"]
        return rest
    return messages
