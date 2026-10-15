"""Experiment stages. Each stage reads the previous stage's JSONL output and writes its own."""
from __future__ import annotations

import json
import os
import random
from collections import Counter

from . import prompts as P
from .data import Item
from .llm import LLMClient, gather_progress
from .perturb import apply_attack, fallback_normalize, strip_normalize, template_pad
from .textutil import extract_final, parse_pairwise, parse_select, validate_rationale, words


# ============================================================================ io helpers
def write_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def vkey(item_id, gen, pol, var):
    return f"{item_id}|{gen}|{pol}|{var}"


def interleave_by(seq, key, order):
    """Round-robin the sequence over `order` values of `key`.

    The RTX 3050 host could hold one model at a time, so the job lists below are model-major to avoid
    swapping. On a host that keeps every judge resident (see scripts/serve_l40s.sh) the opposite is
    wanted: rotating judges keeps all model servers busy at once. Ordering never affects results --
    each call is independent and cached on its content.
    """
    buckets = {o: [x for x in seq if key(x) == o] for o in order}
    out = []
    for i in range(max((len(b) for b in buckets.values()), default=0)):
        for o in order:
            b = buckets[o]
            if i < len(b):
                out.append(b[i])
    return out


# ============================================================================ stage 1: solo answers
async def run_solo(cfg, client: LLMClient, items: list[Item]) -> list[dict]:
    models = sorted(set(cfg["roles"]["generators"]) | set(cfg["roles"]["judges"]))
    g = cfg["generation"]
    jobs = [(it, m) for m in models for it in items]  # model-major: avoids model swapping on local servers

    def make(it, m):
        async def f():
            meta = {"gold": it.gold, "letters": list(it.options), "item_id": it.item_id}
            txt = await client.chat(m, P.SOLO_SYSTEM, P.SOLO_USER.format(question=it.question),
                                    max_tokens=g["max_tokens_solo"], task="solo", meta=meta)
            pred = extract_final(txt)
            return {"item_id": it.item_id, "domain": it.domain, "model": m, "family": client.family(m),
                    "pred": pred, "correct": int(pred == it.gold), "parsed": pred is not None}
        return f

    return await gather_progress([make(it, m) for it, m in jobs], "solo", client)


def choose_distractors(cfg, items: list[Item], solo: list[dict]) -> None:
    """Hard negatives: the wrong option most often chosen by the model pool; random otherwise."""
    by_item: dict[str, Counter] = {}
    for r in solo:
        if r["pred"] and not r["correct"]:
            by_item.setdefault(r["item_id"], Counter())[r["pred"]] += 1
    for it in items:
        rng = random.Random(f"{cfg['seed']}-distractor-{it.item_id}")
        wrong = [l for l in it.options if l != it.gold]
        c = by_item.get(it.item_id)
        if cfg["data"]["distractor"] == "hard" and c:
            top = max(c.values())
            cands = sorted(l for l, v in c.items() if v == top and l in wrong)
            if cands:
                it.distractor, it.distractor_source = rng.choice(cands), "model_error"
                continue
        it.distractor, it.distractor_source = rng.choice(wrong), "random"


# ============================================================================ stage 2: forced-answer rationales
async def run_generate(cfg, client: LLMClient, items: list[Item]) -> list[dict]:
    g = cfg["generation"]
    gens = cfg["roles"]["generators"]

    def make(it, gen, pol):
        async def f():
            target = it.gold if pol == "+" else it.distractor
            user = P.GEN_USER.format(question=it.question, label=target, option_text=it.options[target])
            meta = {"target": target, "letters": list(it.options), "gold": it.gold}
            reason, txt = "none", None
            for attempt in range(g["max_retries_invalid"] + 1):
                temp = 0.0 if attempt == 0 else g["retry_temperature"]
                txt = await client.chat(gen, P.GEN_SYSTEM, user, max_tokens=g["max_tokens_generate"],
                                        temperature=temp, task="generate", meta=meta, nonce=attempt)
                ok, reason = validate_rationale(txt, target)
                if ok:
                    return {"key": vkey(it.item_id, gen, pol, "raw"), "item_id": it.item_id, "gen": gen,
                            "pol": pol, "target": target, "text": txt, "valid": True, "reason": reason,
                            "attempts": attempt + 1, "words": words(txt)}
            return {"key": vkey(it.item_id, gen, pol, "raw"), "item_id": it.item_id, "gen": gen, "pol": pol,
                    "target": target, "text": txt, "valid": False, "reason": reason,
                    "attempts": g["max_retries_invalid"] + 1, "words": words(txt)}
        return f

    jobs = [make(it, gen, pol) for gen in gens for it in items for pol in "+-"]
    return await gather_progress(jobs, "generate", client)


def valid_item_ids(cfg, responses: list[dict]) -> set[str]:
    """An item enters the judging stages only if every generator produced valid R+ and R-."""
    need = 2 * len(cfg["roles"]["generators"])
    ok = Counter(r["item_id"] for r in responses if r["valid"])
    return {i for i, c in ok.items() if c == need}


# ============================================================================ stage 3: variants
async def run_perturb(cfg, client: LLMClient, items: list[Item], responses: list[dict]) -> dict[str, dict]:
    """Build every text variant used downstream: raw, pad, atk:<kind> and their normalised forms."""
    ex, g = cfg["experiment"], cfg["generation"]
    util = cfg["roles"]["utility"]
    item_by = {it.item_id: it for it in items}
    keep = valid_item_ids(cfg, responses)
    variants: dict[str, dict] = {}
    for r in responses:
        if r["item_id"] in keep:
            variants[r["key"]] = {"text": r["text"], "method": "generator", "pol": r["pol"], "target": r["target"]}

    # ---- verbosity padding (both polarities)
    def make_pad(r):
        async def f():
            it, target = item_by[r["item_id"]], r["target"]
            key = vkey(r["item_id"], r["gen"], r["pol"], "pad")
            if ex["pad_mode"] == "llm":
                tw = int(words(r["text"]) * ex["pad_ratio"])
                user = P.PAD_USER.format(target_words=tw, label=target, question=it.question, answer=r["text"])
                for attempt in range(g["max_retries_invalid"] + 1):
                    txt = await client.chat(util, P.PAD_SYSTEM, user, max_tokens=g["max_tokens_pad"],
                                            temperature=0.0 if attempt == 0 else g["retry_temperature"],
                                            task="pad", meta={"target_words": tw}, nonce=attempt)
                    if txt and extract_final(txt) == target and words(txt) >= 1.6 * words(r["text"]):
                        return key, {"text": txt, "method": "llm", "pol": r["pol"], "target": target}
            txt = template_pad(r["text"], ex["pad_ratio"], key)
            return key, {"text": txt, "method": "template", "pol": r["pol"], "target": target}
        return f

    base = [r for r in responses if r["item_id"] in keep]
    for k, v in await gather_progress([make_pad(r) for r in base], "pad", client):
        variants[k] = v

    # ---- deterministic edits.
    # Attacks and the two controls go on the incorrect rationale; the symmetric control puts the same
    # single edits on the *correct* one, so the direction of each cue can be read off the same scale.
    for r in base:
        kinds = list(ex["attacks"]) + list(ex.get("attacks_control") or [])
        if r["pol"] == "+":
            kinds = list(ex.get("attacks_sym") or [])
        for a in kinds:
            k = vkey(r["item_id"], r["gen"], r["pol"], f"atk:{a}")
            variants[k] = {"text": apply_attack(r["text"], a, k), "method": "template", "pol": r["pol"],
                           "target": r["target"]}

    # ---- strip-only normaliser (ablation): deterministic, so no model call and no validation
    protos = list(ex["protocols"]) + list(ex["select"]["protocols"])
    if any(p.endswith("strip") for p in protos):
        for k, v in list(variants.items()):
            variants[k + "#strip"] = {"text": strip_normalize(v["text"]), "method": "strip",
                                      "pol": v["pol"], "target": v["target"]}

    # ---- normaliser agent (only if a *_norm protocol is requested)
    wants_norm = any(p.endswith("norm") for p in protos)
    if wants_norm:
        def make_norm(k, v):
            async def f():
                item_id = k.split("|")[0]
                it, target = item_by[item_id], v["target"]
                user = P.NORM_USER.format(label=target, question=it.question, answer=v["text"])
                for attempt in range(g["max_retries_invalid"] + 1):
                    txt = await client.chat(util, P.NORM_SYSTEM, user, max_tokens=g["max_tokens_normalize"],
                                            temperature=0.0 if attempt == 0 else g["retry_temperature"],
                                            task="normalize", meta={}, nonce=attempt)
                    if txt and extract_final(txt) == target and words(txt) <= 130:
                        return k + "#norm", {"text": txt, "method": "llm", "pol": v["pol"], "target": target}
                return k + "#norm", {"text": fallback_normalize(v["text"]), "method": "fallback",
                                     "pol": v["pol"], "target": target}
            return f
        todo = [(k, v) for k, v in variants.items() if "#" not in k]
        for k, v in await gather_progress([make_norm(k, v) for k, v in todo], "normalize", client):
            variants[k] = v
    return variants


def get_text(variants, item_id, gen, pol, var, protocol):
    k = vkey(item_id, gen, pol, var)
    if protocol.endswith("norm"):
        k += "#norm"
    elif protocol.endswith("strip"):
        k += "#strip"
    return variants[k]["text"]


# ============================================================================ stage 4: pairwise judging
def pair_cycle(item_id: str, gens: list[str], seed) -> list[tuple[str, str]]:
    """Cyclic (correct_gen, wrong_gen) pairs: every generator is the correct source once and the
    wrong source once, so each judge sees 1 self-right, 1 self-wrong and k-2 neutral pairs per item."""
    perm = gens[:]
    random.Random(f"{seed}-pairs-{item_id}").shuffle(perm)
    k = len(perm)
    return [(perm[i], perm[(i + 1) % k]) for i in range(k)]


def design_pairwise(cfg, items: list[Item], keep: set[str]) -> list[dict]:
    ex = cfg["experiment"]
    gens = cfg["roles"]["generators"]
    specs = []
    for it in items:
        if it.item_id not in keep:
            continue
        for p_idx, (cg, wg) in enumerate(pair_cycle(it.item_id, gens, cfg["seed"])):
            for length in ("none", "pad_wrong", "pad_right"):
                for slot in "AB":
                    specs.append(dict(block="bias", item_id=it.item_id, domain=it.domain, pair_idx=p_idx,
                                      cg=cg, wg=wg, length_cond=length, attack="none", correct_slot=slot))
            if p_idx < ex["attack_pairs_per_item"]:
                for a in list(ex["attacks"]) + list(ex.get("attacks_control") or []):
                    for slot in "AB":
                        specs.append(dict(block="attack", item_id=it.item_id, domain=it.domain, pair_idx=p_idx,
                                          cg=cg, wg=wg, length_cond="none", attack=a, correct_slot=slot))
                for a in list(ex.get("attacks_sym") or []):
                    for slot in "AB":
                        specs.append(dict(block="attack_sym", item_id=it.item_id, domain=it.domain,
                                          pair_idx=p_idx, cg=cg, wg=wg, length_cond="none", attack=a,
                                          correct_slot=slot))
    return specs


def spec_protocols(cfg, spec, protocols: list[str]) -> list[str]:
    """Which protocols a spec is judged under.

    The main design (bias + attack blocks) runs under every protocol. The three control conditions are
    cheaper and answer narrower questions, so each has its own protocol list in the config: the placebo
    and held-out edits need the four main protocols (the held-out one is only informative when it can be
    compared with and without the normaliser), the symmetric edits only the base protocol.
    """
    ex = cfg["experiment"]
    if spec["block"] == "attack_sym":
        allow = ex.get("sym_protocols") or protocols
    elif spec["attack"] in (ex.get("attacks_control") or ()):
        allow = ex.get("control_protocols") or protocols
    else:
        return protocols
    return [p for p in protocols if p in allow]


async def run_pairwise(cfg, client: LLMClient, items: list[Item], variants: dict, keep: set[str]) -> list[dict]:
    g = cfg["generation"]
    item_by = {it.item_id: it for it in items}
    specs = design_pairwise(cfg, items, keep)

    def make(spec, judge, protocol):
        async def f():
            it = item_by[spec["item_id"]]
            if spec["block"] == "attack_sym":          # the edit sits on the correct rationale
                cvar, wvar = f"atk:{spec['attack']}", "raw"
            elif spec["attack"] != "none":              # attacks and controls sit on the incorrect one
                cvar, wvar = "raw", f"atk:{spec['attack']}"
            else:
                cvar = "pad" if spec["length_cond"] == "pad_right" else "raw"
                wvar = "pad" if spec["length_cond"] == "pad_wrong" else "raw"
            ct = get_text(variants, it.item_id, spec["cg"], "+", cvar, protocol)
            wt = get_text(variants, it.item_id, spec["wg"], "-", wvar, protocol)
            a, b = (ct, wt) if spec["correct_slot"] == "A" else (wt, ct)
            txt = await client.chat(judge, P.judge_system(protocol), P.JUDGE_USER.format(question=it.question, a=a, b=b),
                                    max_tokens=g["max_tokens_judge"], task="judge", meta={"gold": it.gold})
            v = parse_pairwise(txt)
            score = 0.5 if v in (None, "C") else float(v == spec["correct_slot"])
            return {**spec, "judge": judge, "judge_family": client.family(judge), "protocol": protocol,
                    "cg_family": client.family(spec["cg"]), "wg_family": client.family(spec["wg"]),
                    "verdict": v, "parse_fail": v is None, "is_correct": score,
                    "len_correct": words(ct), "len_wrong": words(wt), "raw": (txt or "")[:600]}
        return f

    judges = cfg["roles"]["judges"]
    protocols = cfg["experiment"]["protocols"]
    triples = [(s, j, p) for j in judges for p in protocols for s in specs
               if p in spec_protocols(cfg, s, protocols)]
    if cfg["runtime"].get("interleave_models"):
        triples = interleave_by(triples, lambda t: t[1], judges)
    return await gather_progress([make(*t) for t in triples], "pairwise", client)


# ============================================================================ stage 5: best-of-N selection
async def run_select(cfg, client: LLMClient, items: list[Item], variants: dict, keep: set[str]) -> list[dict]:
    g = cfg["generation"]
    gens = cfg["roles"]["generators"]
    k = len(gens)
    kept = [it for it in items if it.item_id in keep]
    jobs = []

    for idx, it in enumerate(kept):
        cg = gens[idx % k]  # balanced: each generator supplies the correct candidate equally often
        others = [x for x in gens if x != cg]
        random.Random(f"{cfg['seed']}-select-{it.item_id}").shuffle(others)
        pert = others[0]
        for cond in ("clean", "pad", "attack"):
            wvar = {"clean": "raw", "pad": "pad", "attack": "atk:combo"}[cond]
            base = [(cg, "+", "raw")] + [(o, "-", wvar if o == pert else "raw") for o in others]
            for rot in range(k):
                # rotate so the correct candidate lands at position rot (0-based)
                order = base[-rot:] + base[:-rot] if rot else base[:]
                for protocol in cfg["experiment"]["select"]["protocols"]:
                    for judge in cfg["roles"]["judges"]:
                        jobs.append((it, cond, rot, order, pert, protocol, judge))

    def make(it, cond, rot, order, pert, protocol, judge):
        async def f():
            texts = [get_text(variants, it.item_id, gg, pol, var, protocol) for gg, pol, var in order]
            user = P.SELECT_USER_HEAD.format(question=it.question) + "".join(
                P.SELECT_USER_CAND.format(i=i + 1, text=t) for i, t in enumerate(texts))
            txt = await client.chat(judge, P.select_system(protocol), user.rstrip(),
                                    max_tokens=g["max_tokens_judge"], task="select", meta={"gold": it.gold})
            pick = parse_select(txt, len(order))
            correct_pos = rot + 1
            pert_pos = next(i + 1 for i, (gg, pol, _) in enumerate(order) if gg == pert and pol == "-")
            return {"item_id": it.item_id, "domain": it.domain, "judge": judge,
                    "judge_family": client.family(judge), "protocol": protocol, "cond": cond, "rotation": rot,
                    "correct_gen": order[rot][0], "correct_family": client.family(order[rot][0]),
                    "correct_pos": correct_pos, "perturbed_pos": pert_pos, "pick": pick,
                    "parse_fail": pick is None, "is_correct": int(pick == correct_pos),
                    "picked_perturbed": int(pick == pert_pos), "n_candidates": len(order)}
        return f

    if not cfg["experiment"]["select"]["enabled"]:
        return []
    judges = cfg["roles"]["judges"]
    jobs.sort(key=lambda j: judges.index(j[6]))  # judge-major (stable)
    if cfg["runtime"].get("interleave_models"):
        jobs = interleave_by(jobs, lambda j: j[6], judges)
    return await gather_progress([make(*j) for j in jobs], "select", client)


# ============================================================================ cost estimate
def estimate(cfg, n_items: int | None = None) -> dict:
    ex = cfg["experiment"]
    n = n_items or sum(d["n"] for d in cfg["data"]["domains"].values())
    k = len(cfg["roles"]["generators"])
    J = len(cfg["roles"]["judges"])
    models = len(set(cfg["roles"]["generators"]) | set(cfg["roles"]["judges"]))
    A = len(ex["attacks"])
    ap = min(ex["attack_pairs_per_item"], k)
    per_item_pairs = k * 3 * 2 + ap * A * 2
    n_ctrl = len(ex.get("attacks_control") or [])
    n_sym = len(ex.get("attacks_sym") or [])
    n_cp = len([p for p in ex["protocols"] if p in (ex.get("control_protocols") or ex["protocols"])])
    n_sp = len([p for p in ex["protocols"] if p in (ex.get("sym_protocols") or ex["protocols"])])
    norm = any(p.endswith("norm") for p in ex["protocols"] + ex["select"]["protocols"])
    calls = {
        "solo": n * models,
        "generate (min)": n * k * 2,
        "pad": n * k * 2 if ex["pad_mode"] == "llm" else 0,
        "normalize": n * k * (4 + A + n_ctrl + n_sym) if norm else 0,
        "pairwise": (n * J * len(ex["protocols"]) * per_item_pairs
                     + n * J * ap * 2 * (n_ctrl * n_cp + n_sym * n_sp)),
        "select": n * J * len(ex["select"]["protocols"]) * 3 * k if ex["select"]["enabled"] else 0,
    }
    # rough prompt sizes (tokens): question ~300, raw rationale ~170, padded ~430
    tok_in = {"solo": 350, "generate (min)": 400, "pad": 600, "normalize": 700, "pairwise": 900, "select": 1200}
    tok_out = {"solo": 120, "generate (min)": 220, "pad": 500, "normalize": 140, "pairwise": 120, "select": 120}
    rows = {s: {"calls": c, "in_tok_M": c * tok_in[s] / 1e6, "out_tok_M": c * tok_out[s] / 1e6} for s, c in calls.items()}
    rows["TOTAL"] = {kk: sum(r[kk] for r in rows.values()) for kk in ("calls", "in_tok_M", "out_tok_M")}
    return rows
