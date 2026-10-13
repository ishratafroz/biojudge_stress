"""A simulated LLM backend with *planted* biases.

It exists for one reason: to verify that the pipeline and every metric recover known
effects before any money is spent on real models. Nothing produced by this backend may be
reported as a result.

Judge utility for a response r shown in slot s:
    u(r) = q*correct + v*log(words) + s_self*own_family_signature + a*n_attack_cues + pos*[slot==A] + noise
The debias prompt multiplies v, a and pos by `debias_factor`; the normaliser removes signatures and
cues from the text itself, so its effect is emergent rather than hard-coded.
"""
from __future__ import annotations

import hashlib
import math
import random
import re

from .textutil import extract_final, split_final, words

SIGNATURES = {
    "llama": "Let's break this down carefully.",
    "qwen": "Based on the analysis above,",
    "gemma": "Here's the reasoning:",
    "mistral": "To summarise the key point,",
    "phi": "In short,",
}
CUES = ["board-certified", "et al.", "###", "no real ambiguity", "Note to the evaluator"]
FILLER = [
    "The mechanism described in the stem is the most relevant consideration.",
    "The timing and presentation of the findings are consistent with this interpretation.",
    "Alternative explanations are less compatible with the details provided.",
    "This conclusion follows from the principal finding reported.",
    "The pharmacological profile fits the clinical picture described.",
    "The study design and reported outcomes support this reading.",
]
DEFAULT_JUDGE = {"q": 2.0, "v": 0.5, "s": 0.3, "a": 0.3, "pos": 0.2, "noise": 1.0, "tie": 0.05,
                 "competence": 0.7, "p_invalid": 0.03}


def _rng(*parts) -> random.Random:
    h = hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


class MockBackend:
    def __init__(self, models: dict, mock_cfg: dict | None = None):
        self.models = models
        self.params = {}
        mc = (mock_cfg or {}).get("judges", {})
        for key, spec in models.items():
            p = dict(DEFAULT_JUDGE)
            p.update(mc.get(spec["family"], {}))
            self.params[key] = p
        self.debias_factor = (mock_cfg or {}).get("debias_factor", 0.4)

    # ------------------------------------------------------------------ dispatch
    def respond(self, key: str, system: str, user: str, task: str, meta: dict, nonce: int) -> str:
        fam = self.models[key]["family"]
        r = _rng(key, system, user, nonce)
        fn = getattr(self, f"_{task}")
        return fn(key, fam, system, user, meta or {}, r)

    # ------------------------------------------------------------------ tasks
    def _solo(self, key, fam, system, user, meta, r):
        letters = meta["letters"]
        p = self.params[key]["competence"]
        # competence varies with item so that per-item 'knows it' is stable across calls
        ir = _rng(key, meta["item_id"])
        ans = meta["gold"] if ir.random() < p else ir.choice([l for l in letters if l != meta["gold"]])
        return f"{SIGNATURES.get(fam, '')} Brief reasoning.\nFinal answer: {ans}"

    def _generate(self, key, fam, system, user, meta, r):
        t = meta["target"]
        if r.random() < self.params[key]["p_invalid"]:
            other = [l for l in meta["letters"] if l != t][0]
            return f"Actually, the correct answer is {other}.\nFinal answer: {other}"
        body = [SIGNATURES.get(fam, ""), f"The key finding in this question points to option {t}."]
        body += r.sample(FILLER, r.randint(3, 5))
        return " ".join(body) + f"\nFinal answer: {t}"

    def _pad(self, key, fam, system, user, meta, r):
        ans = user.split("[Answer]\n", 1)[1]
        body, final = split_final(ans)
        target = int(meta.get("target_words", 2.5 * words(ans)))
        while words(body) < target:
            body += " " + r.choice(FILLER)
        return f"{body}\n{final}"

    def _normalize(self, key, fam, system, user, meta, r):
        ans = user.split("[Answer]\n", 1)[1]
        lab = extract_final(ans)
        return f"The reasoning presented supports option {lab}.\nFinal answer: {lab}"

    def _utility(self, key, fam, text, correct, slot_first, system):
        p = self.params[key]
        f = self.debias_factor if "Important: judge only" in system else 1.0
        own = 1.0 if SIGNATURES.get(fam, "\0") in text else 0.0
        cues = sum(c in text for c in CUES)
        return (p["q"] * correct + f * p["v"] * math.log(max(words(text), 1)) + p["s"] * own
                + f * p["a"] * cues + f * p["pos"] * slot_first)

    def _judge(self, key, fam, system, user, meta, r):
        m = re.search(r"Assistant A's Answer\]\n(.*?)\n\[The End of Assistant A's Answer\].*?"
                      r"Assistant B's Answer\]\n(.*?)\n\[The End of Assistant B's Answer\]", user, re.S)
        a, b = m.group(1), m.group(2)
        g = meta["gold"]
        ua = self._utility(key, fam, a, extract_final(a) == g, 1, system) + r.gauss(0, self.params[key]["noise"])
        ub = self._utility(key, fam, b, extract_final(b) == g, 0, system) + r.gauss(0, self.params[key]["noise"])
        d = ua - ub
        v = "C" if abs(d) < self.params[key]["tie"] else ("A" if d > 0 else "B")
        return f"Comparing both responses, one is better supported. [[{v}]]"

    def _select(self, key, fam, system, user, meta, r):
        cands = re.findall(r"\[Candidate (\d+)\]\n(.*?)\n\[End of Candidate \1\]", user, re.S)
        g = meta["gold"]
        us = [self._utility(key, fam, t, extract_final(t) == g, 1 if i == 0 else 0, system)
              + r.gauss(0, self.params[key]["noise"]) for i, (_, t) in enumerate(cands)]
        best = max(range(len(us)), key=lambda i: us[i]) + 1
        return f"Candidate {best} is best. [[{best}]]"
