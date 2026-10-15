"""Biomedical item loaders.

Every item is converted into a single-best-answer multiple-choice question with
letter labels, so answer extraction is identical across domains:

  literature   -> PubMedQA PQA-L (expert-labelled; yes/no items only, 'maybe' dropped)
  clinical     -> MedQA-USMLE 4-option test split
  pharmacology -> MedMCQA validation split, subject == Pharmacology, single-choice only

The gold label is the dataset's expert/exam answer key; it is the human anchor
against which every judge verdict is scored.
"""
from __future__ import annotations

import json
import os
import random
import re
import urllib.request
from dataclasses import asdict, dataclass, field

PUBMEDQA_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
_BAD_OPTION = re.compile(r"\b(all|none|both|neither) of the (above|following)\b|^both\b|^all\b|^none\b", re.I)


@dataclass
class Item:
    item_id: str
    domain: str
    source: str
    question: str               # full text shown to every model (stem + context + options)
    options: dict               # letter -> option text
    gold: str                   # letter
    distractor: str | None = None
    distractor_source: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Item":
        return Item(**d)


def format_question(stem: str, options: dict, context: str | None = None) -> str:
    parts = []
    if context:
        parts.append("Context (structured abstract, conclusions removed):\n" + context.strip())
    parts.append("Question: " + stem.strip())
    parts.append("Options:\n" + "\n".join(f"{k}. {v}" for k, v in options.items()))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- PubMedQA
def _pubmedqa_raw(path: str | None, cache_dir: str) -> dict:
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, "ori_pqal.json")
    if not os.path.exists(local):
        print(f"[data] downloading PubMedQA PQA-L -> {local}")
        urllib.request.urlretrieve(PUBMEDQA_URL, local)
    with open(local) as f:
        return json.load(f)


def load_pubmedqa(n: int, rng: random.Random, domain: str, path=None, cache_dir="data_cache") -> list[Item]:
    raw = _pubmedqa_raw(path, cache_dir)
    rows = sorted((pmid, v) for pmid, v in raw.items() if v["final_decision"] in ("yes", "no"))
    rows = rng.sample(rows, min(n, len(rows)))
    items = []
    for pmid, v in rows:
        ctx = "\n".join(f"{lab.title()}: {txt}" for lab, txt in zip(v["LABELS"], v["CONTEXTS"]))
        options = {"A": "yes", "B": "no"}
        gold = "A" if v["final_decision"] == "yes" else "B"
        items.append(Item(
            item_id=f"pqa-{pmid}", domain=domain, source="pubmedqa",
            question=format_question(v["QUESTION"], options, ctx),
            options=options, gold=gold, meta={"pmid": pmid, "year": v.get("YEAR")},
        ))
    return items


# --------------------------------------------------------------------------- HF helpers
def _hf(name: str, split: str, config: str | None = None):
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install datasets  (needed for MedQA / MedMCQA)") from e
    return load_dataset(name, config, split=split) if config else load_dataset(name, split=split)


def load_medqa(n: int, rng: random.Random, domain: str, **_) -> list[Item]:
    ds = _hf("GBaker/MedQA-USMLE-4-options", "test")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    items = []
    for i in idx:
        r = ds[i]
        options = dict(sorted(r["options"].items()))
        if any(_BAD_OPTION.search(str(t)) for t in options.values()):
            continue
        gold = r.get("answer_idx")
        if gold not in options:
            continue
        items.append(Item(
            item_id=f"medqa-{i}", domain=domain, source="medqa",
            question=format_question(r["question"], options), options=options, gold=gold,
            meta={"meta_info": r.get("meta_info")},
        ))
        if len(items) >= n:
            break
    return items


def load_medmcqa_pharm(n: int, rng: random.Random, domain: str, **_) -> list[Item]:
    ds = _hf("openlifescienceai/medmcqa", "validation")
    idx = [i for i, s in enumerate(ds["subject_name"]) if s == "Pharmacology"]
    rng.shuffle(idx)
    items = []
    for i in idx:
        r = ds[i]
        if r.get("choice_type") != "single":
            continue
        options = {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]}
        if any((not str(t).strip()) or _BAD_OPTION.search(str(t)) for t in options.values()):
            continue
        gold = "ABCD"[int(r["cop"])]
        items.append(Item(
            item_id=f"mmcqa-{r['id']}", domain=domain, source="medmcqa_pharm",
            question=format_question(r["question"], options), options=options, gold=gold,
            meta={"topic": r.get("topic_name")},
        ))
        if len(items) >= n:
            break
    return items


# --------------------------------------------------------------------------- synthetic (tests)
def load_synthetic(n: int, rng: random.Random, domain: str, **_) -> list[Item]:
    items = []
    for i in range(n):
        options = {L: f"synthetic option {L.lower()}{i}" for L in "ABCD"}
        gold = rng.choice("ABCD")
        items.append(Item(
            item_id=f"syn-{domain}-{i}", domain=domain, source="synthetic",
            question=format_question(f"Synthetic biomedical question {i} ({domain})?", options),
            options=options, gold=gold,
        ))
    return items


LOADERS = {
    "pubmedqa": load_pubmedqa,
    "medqa": load_medqa,
    "medmcqa_pharm": load_medmcqa_pharm,
    "synthetic": load_synthetic,
}


def build_items(cfg: dict) -> list[Item]:
    dcfg = cfg["data"]
    out: list[Item] = []
    for domain, spec in dcfg["domains"].items():
        rng = random.Random(f"{cfg['seed']}-{domain}")
        loader = LOADERS[spec["source"]]
        items = loader(spec["n"], rng, domain, path=dcfg.get("pubmedqa_path"), cache_dir=dcfg["cache_dir"])
        if len(items) < spec["n"]:
            print(f"[data] warning: {domain} yielded {len(items)} < {spec['n']} items")
        out.extend(items)
    return out


def save_items(items: list[Item], path: str) -> None:
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it.to_dict()) + "\n")


def load_items(path: str) -> list[Item]:
    with open(path) as f:
        return [Item.from_dict(json.loads(l)) for l in f if l.strip()]
