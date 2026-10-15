"""Why does a verdict fail to parse: truncation, or a missing tag?

The judge prompt asks for a "[[A]]"-style tag. When it is absent the verdict scores as a tie, so it
matters whether the model ran out of tokens or simply did not comply. This script re-issues a sample of
prompts -- those whose verdict did not parse, and a matched sample that did -- and records the server's
finish_reason and completion length, then writes the summary the manuscript quotes.

  python scripts/audit_parse_failures.py --config configs/ollama_l40s_ctrl.yaml --run runs/local
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biojudge import prompts as P
from biojudge.config import load_config
from biojudge.data import load_items
from biojudge.pipeline import get_text, read_jsonl
from biojudge.textutil import parse_pairwise


def judge_prompt(cfg, items, variants, r):
    it = items[r["item_id"]]
    if r["block"] == "attack_sym":
        cvar, wvar = f"atk:{r['attack']}", "raw"
    elif r["attack"] != "none":
        cvar, wvar = "raw", f"atk:{r['attack']}"
    else:
        cvar = "pad" if r["length_cond"] == "pad_right" else "raw"
        wvar = "pad" if r["length_cond"] == "pad_wrong" else "raw"
    ct = get_text(variants, it.item_id, r["cg"], "+", cvar, r["protocol"])
    wt = get_text(variants, it.item_id, r["wg"], "-", wvar, r["protocol"])
    a, b = (ct, wt) if r["correct_slot"] == "A" else (wt, ct)
    return P.judge_system(r["protocol"]), P.JUDGE_USER.format(question=it.question, a=a, b=b)


async def main_async(cfg, run, n_fail, n_ok):
    from openai import AsyncOpenAI
    items = {it.item_id: it for it in load_items(os.path.join(run, "items.jsonl"))}
    variants = {r.pop("key"): r for r in read_jsonl(os.path.join(run, "variants.jsonl"))}
    tr = pd.read_json(os.path.join(run, "trials.jsonl"), lines=True)
    mt = cfg["generation"]["max_tokens_judge"]
    client = AsyncOpenAI(base_url=cfg["models"][cfg["roles"]["judges"][0]]["base_url"],
                         api_key=os.environ.get("OLLAMA_API_KEY", "EMPTY"), timeout=300, max_retries=0)

    async def one(r, group):
        spec = cfg["models"][r["judge"]]
        system, user = judge_prompt(cfg, items, variants, r)
        msgs = ([{"role": "user", "content": f"{system}\n\n{user}"}] if spec.get("merge_system")
                else [{"role": "system", "content": system}, {"role": "user", "content": user}])
        resp = await client.chat.completions.create(model=spec["model"], messages=msgs, max_tokens=mt,
                                                    temperature=0.0, seed=cfg["seed"])
        ch = resp.choices[0]
        return {"judge": r["judge"], "group": group, "finish_reason": ch.finish_reason,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                "parses_now": int(parse_pairwise(ch.message.content or "") is not None)}

    jobs = []
    for j in cfg["roles"]["judges"]:
        f = tr[(tr.judge == j) & tr.parse_fail]
        o = tr[(tr.judge == j) & ~tr.parse_fail]
        jobs += [(r, "parse_fail") for r in f.sample(min(n_fail, len(f)), random_state=2).to_dict("records")]
        jobs += [(r, "parsed") for r in o.sample(min(n_ok, len(o)), random_state=2).to_dict("records")]
    out = pd.DataFrame(await asyncio.gather(*(one(r, g) for r, g in jobs)))
    fails = out[out.group == "parse_fail"]
    summary = {
        "n_calls": int(len(out)), "n_parse_fail_sampled": int(len(fails)),
        "finish_reasons": out.finish_reason.value_counts().to_dict(),
        "max_completion_tokens": int(out.completion_tokens.max()),
        "max_tokens_allowed": mt,
        "reparse_rate_overall": float(fails.parses_now.mean()),
        "reparse_rate_by_judge": fails.groupby("judge").parses_now.mean().to_dict(),
        "tokens_by_group": out.groupby("group").completion_tokens.mean().round(1).to_dict(),
    }
    path = os.path.join(run, "parse_audit.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-fail", type=int, default=15)
    ap.add_argument("--n-ok", type=int, default=10)
    a = ap.parse_args()
    asyncio.run(main_async(load_config(a.config), a.run, a.n_fail, a.n_ok))


if __name__ == "__main__":
    main()
