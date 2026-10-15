"""Configuration loading. A run is fully specified by one YAML file."""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "run_name": "run",
    "out_dir": "runs",
    "seed": 2026,
    "data": {
        "domains": {
            "literature": {"source": "pubmedqa", "n": 100},
            "clinical": {"source": "medqa", "n": 100},
            "pharmacology": {"source": "medmcqa_pharm", "n": 100},
        },
        "distractor": "hard",  # hard | random
        "pubmedqa_path": None,
        "cache_dir": "data_cache",
    },
    "models": {},
    "roles": {"generators": [], "judges": [], "utility": None},
    "experiment": {
        "protocols": ["base_raw", "debias_raw", "base_norm", "debias_norm"],
        "attacks": ["authority", "citation", "format", "confidence", "injection", "combo"],
        # Controls for verdict instability. attacks_control go on the incorrect rationale like the
        # attacks; attacks_sym put the same single edits on the correct one. Each has its own protocol
        # list so a control costs only the calls its question needs.
        "attacks_control": [],
        "attacks_sym": [],
        "control_protocols": None,
        "sym_protocols": None,
        "attack_pairs_per_item": 4,
        "pad_mode": "llm",  # llm | template
        "pad_ratio": 2.5,
        "select": {"enabled": True, "protocols": ["base_raw", "debias_norm"]},
    },
    "generation": {
        "max_tokens_solo": 200,
        "max_tokens_generate": 400,
        "max_tokens_pad": 900,
        "max_tokens_normalize": 250,
        "max_tokens_judge": 300,
        "max_retries_invalid": 2,
        "retry_temperature": 0.7,
    },
    "runtime": {"concurrency": 16, "request_timeout": 120, "max_api_retries": 6,
                "interleave_models": False},  # True when every model stays resident (multi-GPU host)
    "analysis": {"bootstrap": 1000},
    "mock": {},
}

# base/debias = judge prompt; raw/norm/strip = what the judge is shown (unchanged text, LLM-normalised
# text, or the strip-only ablation of the normaliser).
VALID_PROTOCOLS = {"base_raw", "debias_raw", "base_norm", "debias_norm", "base_strip", "debias_strip"}
VALID_ATTACKS = {"authority", "citation", "format", "confidence", "injection", "combo"}
VALID_CONTROLS = {"placebo", "consensus"}


def _deep_merge(base: dict, over: dict) -> dict:
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict) and k not in ("models", "domains"):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str) -> dict:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = _deep_merge(copy.deepcopy(DEFAULTS), user)
    validate(cfg)
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def validate(cfg: dict) -> None:
    models = cfg["models"]
    roles = cfg["roles"]
    for role in ("generators", "judges"):
        if not roles[role]:
            raise ValueError(f"roles.{role} must list at least one model key")
        for k in roles[role]:
            if k not in models:
                raise ValueError(f"roles.{role} references unknown model '{k}'")
    if roles["utility"] not in models:
        raise ValueError("roles.utility must reference a model key")
    if len(roles["generators"]) < 3:
        raise ValueError("need >= 3 generators so every judge sees neutral, self-right and self-wrong pairs")
    fams = [models[g]["family"] for g in roles["generators"]]
    if len(set(fams)) != len(fams):
        raise ValueError("generators must come from distinct model families")
    util_fam = models[roles["utility"]]["family"]
    if util_fam in {models[j]["family"] for j in roles["judges"]}:
        print(f"[warn] utility model family '{util_fam}' overlaps a judge family; "
              "normalised/padded text may carry that family's style and confound self-preference.")
    for p in cfg["experiment"]["protocols"] + cfg["experiment"]["select"]["protocols"]:
        if p not in VALID_PROTOCOLS:
            raise ValueError(f"unknown protocol {p}; valid: {sorted(VALID_PROTOCOLS)}")
    for a in cfg["experiment"]["attacks"]:
        if a not in VALID_ATTACKS:
            raise ValueError(f"unknown attack {a}; valid: {sorted(VALID_ATTACKS)}")
    for a in cfg["experiment"].get("attacks_control") or []:
        if a not in VALID_CONTROLS:
            raise ValueError(f"unknown control edit {a}; valid: {sorted(VALID_CONTROLS)}")
    for a in cfg["experiment"].get("attacks_sym") or []:
        if a not in VALID_ATTACKS - {"combo"}:
            raise ValueError(f"unknown symmetric edit {a}; valid: {sorted(VALID_ATTACKS - {'combo'})}")
    for key in ("control_protocols", "sym_protocols"):
        for p in cfg["experiment"].get(key) or []:
            if p not in cfg["experiment"]["protocols"]:
                raise ValueError(f"experiment.{key} lists {p}, which is not in experiment.protocols")
    for key, spec in models.items():
        for req in ("provider", "model", "family"):
            if req not in spec:
                raise ValueError(f"models.{key} missing '{req}'")


def run_dir(cfg: dict) -> str:
    d = os.path.join(cfg["out_dir"], cfg["run_name"])
    os.makedirs(d, exist_ok=True)
    return d
