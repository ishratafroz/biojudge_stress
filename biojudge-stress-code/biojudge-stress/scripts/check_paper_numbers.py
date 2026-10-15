"""Audit every number in the manuscript template.

fill_paper.py guarantees that each @@slot@@ comes from metrics.json. This script guards the other
direction: it finds numbers typed straight into the prose, where nothing checks them. Each literal is
either matched against the run configuration (design constants such as the number of items per domain,
the padding ratio or the bootstrap size), matched against the standard-constant list, or reported.

  python scripts/check_paper_numbers.py --template paper/biojudge_stress_template.tex \
      --config configs/ollama_l40s_ctrl.yaml --metrics runs/local/results/metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from biojudge.config import load_config  # noqa: E402

NUM = re.compile(r"(?<![\w.@{\\])(\d+(?:[.,]\d+)*)(?![\w.@}])")
# Numbers that are part of LaTeX machinery rather than claims about the experiment.
SKIP_LINE = re.compile(r"\\(usepackage|setlength|documentclass|includegraphics|cmidrule|multicolumn|"
                       r"tabcolsep|arraystretch|begin\{tabular|hline|label|cite|ref|bibitem|newcommand|"
                       r"providecommand|renewcommand|IEEEauthorblock|definecolor|arXiv)")
SKIP_CONTEXT = re.compile(r"(columnwidth|textwidth|linewidth|pt\}|ex\}|\{b\d|arXiv:|\\cite|\\hw)")
# model names (Llama-3.2-3B, Qwen2.5-3B, Gemma-2-2B, Phi-4-mini, ...) and the usual alpha levels
SKIP_TOKEN = re.compile(r"(?:Llama|Qwen|Gemma|Ministral|Mistral|Phi|GPT)[-\d.]*$|^p[<=]")
ALPHA = {"0.05", "0.01", "0.001"}
# Conventional constants that need no run to justify.
STANDARD = {
    "0": "zero", "1": "unit", "2": "two sides / two options", "3": "three domains or sentences",
    "4": "four judges, four options, best-of-4", "5": "five single edits, five families",
    "6": "six edits", "7": "seven corrections", "8": "eight rationales per item",
    "12": "twelve regression tests", "24": "units per block (derived in the text)",
    "25": "chance level for four options (%), items per domain", "50": "chance level for two options (%)",
    "80": "power (%), normaliser word budget", "95": "confidence level (%)", "100": "percent",
    "9": "nine configurations", "20": "twenty years (edit text)", "48": "units per item/judge/protocol",
    "1960": "year", "2026": "year",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics")
    a = ap.parse_args()
    cfg = load_config(a.config)
    ex, gen, data = cfg["experiment"], cfg["generation"], cfg["data"]
    design = {
        str(cfg["seed"]): "config seed",
        str(cfg["analysis"]["bootstrap"]): "config analysis.bootstrap",
        str(ex["pad_ratio"]): "config experiment.pad_ratio",
        str(gen["max_tokens_judge"]): "config generation.max_tokens_judge",
        str(gen["max_tokens_solo"]): "config generation.max_tokens_solo",
        str(gen["retry_temperature"]): "config generation.retry_temperature",
        str(ex["attack_pairs_per_item"]): "config experiment.attack_pairs_per_item",
        "20,000": "permutation draws (analysis.kendall_w_test default)",
        "2,000": "config analysis.bootstrap",
    }
    for d, spec in data["domains"].items():
        design[str(spec["n"])] = f"config data.domains.{d}.n"
    prov = cfg.get("provenance") or {}
    for v in prov.values():
        for tok in re.findall(r"\d+(?:\.\d+)*", str(v)):
            design.setdefault(tok, "config provenance (host/engine record)")
    macro_vals = set()
    if a.metrics and os.path.exists(a.metrics):
        with open(a.metrics) as f:
            macro_vals = {str(v) for v in json.load(f)["macros"].values()}

    unexplained, explained = [], 0
    for i, line in enumerate(open(a.template), 1):
        if SKIP_LINE.search(line) or line.lstrip().startswith("%"):
            continue
        for mo in NUM.finditer(line):
            tok = mo.group(1)
            ctx = line[max(0, mo.start() - 40):mo.end() + 40].strip()
            if SKIP_CONTEXT.search(ctx):
                continue
            before = line[:mo.start()]
            if SKIP_TOKEN.search(before[-20:]) or tok in ALPHA:
                continue
            if tok in STANDARD or tok in design or tok in macro_vals:
                explained += 1
                continue
            unexplained.append((i, tok, ctx))
    print(f"[check] {explained} literal numbers matched a design constant, the config, or a computed value")
    if unexplained:
        print(f"[check] {len(unexplained)} literal number(s) with no automatic source -- review each:")
        for ln, tok, ctx in unexplained:
            print(f"  line {ln}: {tok}\n      ...{ctx}...")
    else:
        print("[check] no unexplained literals")
    raise SystemExit(1 if unexplained else 0)


if __name__ == "__main__":
    main()
