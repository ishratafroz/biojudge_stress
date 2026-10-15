"""Repeat-decode reliability of the pairwise verdicts.

Every verdict in the main run is a single greedy decode. Greedy is not the same as reproducible: with
continuous batching the same prompt can be split across batches differently, the floating-point
reduction order changes, and the decoded text diverges. This script asks each clean pair again --
same prompt, same model, same temperature 0, only a different cache nonce -- and reports how often the
second decode gives the same text and the same verdict. That rate is the floor under every "share of
verdicts that changed" number in the paper, including Flip and the attack success rates.

  python scripts/redecode.py --config configs/ollama_l40s_ctrl.yaml --run runs/local --protocol base_raw
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biojudge import prompts as P
from biojudge.config import load_config
from biojudge.data import load_items
from biojudge.llm import LLMClient, gather_progress
from biojudge.pipeline import get_text, read_jsonl, write_jsonl
from biojudge.textutil import parse_pairwise


async def main_async(cfg, run, protocol, nonce):
    items = {it.item_id: it for it in load_items(os.path.join(run, "items.jsonl"))}
    variants = {r.pop("key"): r for r in read_jsonl(os.path.join(run, "variants.jsonl"))}
    tr = pd.read_json(os.path.join(run, "trials.jsonl"), lines=True)
    clean = tr[(tr.protocol == protocol) & (tr.block == "bias") & (tr.length_cond == "none")]
    client = LLMClient(cfg, os.path.join(run, "cache.sqlite"))
    mt = cfg["generation"]["max_tokens_judge"]

    def make(r):
        async def f():
            it = items[r["item_id"]]
            ct = get_text(variants, it.item_id, r["cg"], "+", "raw", protocol)
            wt = get_text(variants, it.item_id, r["wg"], "-", "raw", protocol)
            a, b = (ct, wt) if r["correct_slot"] == "A" else (wt, ct)
            system = P.judge_system(protocol)
            user = P.JUDGE_USER.format(question=it.question, a=a, b=b)
            first = client.cache.get(client.cache_key(r["judge"], system, user, max_tokens=mt))
            txt = await client.chat(r["judge"], system, user, max_tokens=mt, task="judge",
                                    meta={"gold": it.gold}, nonce=nonce)
            v = parse_pairwise(txt)
            score = 0.5 if v in (None, "C") else float(v == r["correct_slot"])
            return {"judge": r["judge"], "protocol": protocol, "item_id": r["item_id"],
                    "domain": r["domain"], "pair_idx": r["pair_idx"], "correct_slot": r["correct_slot"],
                    "verdict": r["verdict"], "verdict_retest": v,
                    "verdict_same": int((r["verdict"] if pd.notna(r["verdict"]) else "N")
                                        == (v if v is not None else "N")),
                    "text_same": int(first is not None and (txt or "").strip() == first.strip()),
                    "had_first_text": int(first is not None),
                    "is_correct": float(r["is_correct"]), "is_correct_retest": score}
        return f

    rows = await gather_progress([make(r) for r in clean.to_dict("records")], "redecode", client)
    client.close()
    d = pd.DataFrame(rows)
    out = os.path.join(run, "retest.jsonl")
    write_jsonl(rows, out)
    print(f"\n[redecode] {len(d)} clean units re-decoded under {protocol} (nonce={nonce})")
    print(f"  same text    : {d.text_same.mean():.3f}  (of {int(d.had_first_text.sum())} with the first decode cached)")
    print(f"  same verdict : {d.verdict_same.mean():.3f}")
    print(d.groupby("judge")[["text_same", "verdict_same"]].mean().round(3).to_string())
    print(f"  accuracy, first decode {d.is_correct.mean():.4f} vs re-decode {d.is_correct_retest.mean():.4f}")
    print(f"  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--protocol", default="base_raw")
    ap.add_argument("--nonce", type=int, default=1)
    a = ap.parse_args()
    asyncio.run(main_async(load_config(a.config), a.run, a.protocol, a.nonce))


if __name__ == "__main__":
    main()
