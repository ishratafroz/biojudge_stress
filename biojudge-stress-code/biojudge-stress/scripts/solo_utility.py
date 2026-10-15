"""Ask the utility model (Phi-4-mini) the items on its own.

The utility model is deliberately kept out of the solo stage: that stage also decides each item's hard
negative, and adding a fifth voter would change the distractors and therefore the whole run. Its own
competence is still needed to read the normaliser's effect (does the rewrite help where the rewriting
model knows the answer?), so it is measured here and written to a separate file that nothing else reads.

  python scripts/solo_utility.py --config configs/ollama_l40s_ctrl.yaml --run runs/local
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biojudge import prompts as P
from biojudge.config import load_config
from biojudge.data import load_items
from biojudge.llm import LLMClient, gather_progress
from biojudge.pipeline import write_jsonl
from biojudge.textutil import extract_final


async def main_async(cfg, run):
    items = load_items(os.path.join(run, "items.jsonl"))
    client = LLMClient(cfg, os.path.join(run, "cache.sqlite"))
    util = cfg["roles"]["utility"]

    def make(it):
        async def f():
            txt = await client.chat(util, P.SOLO_SYSTEM, P.SOLO_USER.format(question=it.question),
                                    max_tokens=cfg["generation"]["max_tokens_solo"], task="solo",
                                    meta={"gold": it.gold, "item_id": it.item_id})
            pred = extract_final(txt)
            return {"item_id": it.item_id, "domain": it.domain, "model": util,
                    "family": client.family(util), "pred": pred, "correct": int(pred == it.gold),
                    "parsed": pred is not None}
        return f

    rows = await gather_progress([make(it) for it in items], "solo_utility", client)
    client.close()
    out = os.path.join(run, "solo_utility.jsonl")
    write_jsonl(rows, out)
    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"[solo_utility] {util}: {acc:.3f} on {len(rows)} items "
          f"({sum(r['parsed'] for r in rows)}/{len(rows)} parsed) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    asyncio.run(main_async(load_config(a.config), a.run))


if __name__ == "__main__":
    main()
