"""Run the experiment.

  python -m biojudge.run --config configs/openrouter.yaml --estimate
  python -m biojudge.run --config configs/openrouter.yaml                 # all stages
  python -m biojudge.run --config configs/openrouter.yaml --stages judge  # resume a single stage

Every LLM call is cached, so re-running any stage after an interruption costs nothing for work
already done.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from .config import load_config, run_dir
from .data import build_items, load_items, save_items
from .llm import LLMClient
from .pipeline import (choose_distractors, estimate, read_jsonl, run_generate, run_pairwise, run_perturb,
                       run_select, run_solo, valid_item_ids, write_jsonl)

STAGES = ["items", "solo", "generate", "perturb", "judge", "select"]


def _print_estimate(cfg):
    rows = estimate(cfg)
    print(f"{'stage':<16}{'calls':>10}{'in tok (M)':>12}{'out tok (M)':>12}")
    for s, r in rows.items():
        print(f"{s:<16}{r['calls']:>10,}{r['in_tok_M']:>12.1f}{r['out_tok_M']:>12.1f}")
    print("Multiply by your provider's $/M-token prices. Generation retries add a few percent.")


async def main_async(cfg, stages):
    rd = run_dir(cfg)
    paths = {s: os.path.join(rd, f"{s}.jsonl") for s in ["items", "solo", "responses", "variants", "trials", "select"]}
    client = LLMClient(cfg, os.path.join(rd, "cache.sqlite"))
    t0 = time.time()

    if "items" in stages or not os.path.exists(paths["items"]):
        items = build_items(cfg)
        save_items(items, paths["items"])
        print(f"[items] {len(items)} items")
    items = load_items(paths["items"])

    if "solo" in stages:
        solo = await run_solo(cfg, client, items)
        write_jsonl(solo, paths["solo"])
        choose_distractors(cfg, items, solo)
        save_items(items, paths["items"])
        print(f"[solo] done; distractors: "
              f"{sum(i.distractor_source == 'model_error' for i in items)} hard / {len(items)}")
    if any(i.distractor is None for i in items):
        raise SystemExit("items have no distractors yet: run the 'solo' stage first")

    if "generate" in stages:
        resp = await run_generate(cfg, client, items)
        write_jsonl(resp, paths["responses"])
        keep = valid_item_ids(cfg, resp)
        print(f"[generate] valid rationales {sum(r['valid'] for r in resp)}/{len(resp)}; "
              f"items kept {len(keep)}/{len(items)}")
    resp = read_jsonl(paths["responses"]) if os.path.exists(paths["responses"]) else None

    if "perturb" in stages:
        variants = await run_perturb(cfg, client, items, resp)
        write_jsonl([{"key": k, **v} for k, v in variants.items()], paths["variants"])
        print(f"[perturb] {len(variants)} text variants")
    variants = ({r.pop("key"): r for r in read_jsonl(paths["variants"])}
                if os.path.exists(paths["variants"]) else None)

    keep = valid_item_ids(cfg, resp) if resp else set()
    if "judge" in stages:
        trials = await run_pairwise(cfg, client, items, variants, keep)
        write_jsonl(trials, paths["trials"])
        print(f"[judge] {len(trials)} pairwise verdicts; parse failures "
              f"{sum(t['parse_fail'] for t in trials)}")
    if "select" in stages:
        sel = await run_select(cfg, client, items, variants, keep)
        write_jsonl(sel, paths["select"])
        print(f"[select] {len(sel)} best-of-N selections")

    client.close()
    with open(os.path.join(rd, "run_meta.json"), "w") as f:
        json.dump({"config": cfg, "stages": stages, "seconds": time.time() - t0,
                   "llm_stats": dict(client.stats)}, f, indent=2, default=str)
    print(f"[done] {time.time() - t0:.0f}s; outputs in {rd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", default=",".join(STAGES[1:]),
                    help=f"comma-separated subset of {STAGES}")
    ap.add_argument("--estimate", action="store_true", help="print call/token counts and exit")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.estimate:
        _print_estimate(cfg)
        return
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = set(stages) - set(STAGES)
    if bad:
        raise SystemExit(f"unknown stages {bad}")
    asyncio.run(main_async(cfg, stages))


if __name__ == "__main__":
    main()
