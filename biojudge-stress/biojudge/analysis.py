"""Metrics, corrections and uncertainty.

Uncertainty: a single item-level cluster bootstrap weight matrix W (B x n_items) is drawn once and
reused for every estimate, so all differences between judges, corrections or domains are *paired*
bootstrap differences. 95% CIs are percentile intervals; p-values are two-sided bootstrap sign tests.

Scores: a verdict scores 1 if it prefers the gold-correct response, 0 if it prefers the incorrect one,
and 0.5 for a tie or an unparseable verdict (reported separately).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KEY_COLS = ["judge", "judge_family", "protocol", "block", "item_id", "domain", "pair_idx", "cg", "wg",
            "cg_family", "wg_family", "length_cond", "attack", "self_cond", "correct_slot"]
SINGLE_ATTACKS = ["authority", "citation", "format", "confidence", "injection"]


# =============================================================================== estimates
@dataclass
class Est:
    point: float
    boot: np.ndarray

    def ci(self):
        b = self.boot[~np.isnan(self.boot)]
        if b.size == 0:
            return (np.nan, np.nan)
        return tuple(np.percentile(b, [2.5, 97.5]))

    def p(self):
        b = self.boot[~np.isnan(self.boot)]
        if b.size == 0:
            return np.nan
        return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))

    def __sub__(self, o):
        return Est(self.point - o.point, self.boot - o.boot)

    def __add__(self, o):
        return Est(self.point + o.point, self.boot + o.boot)

    def abs(self):
        return Est(abs(self.point), np.abs(self.boot))

    def as_dict(self):
        lo, hi = self.ci()
        return {"est": self.point, "lo": lo, "hi": hi, "p": self.p()}


class Boot:
    def __init__(self, item_ids, B: int, seed: int):
        self.items = np.array(sorted(set(item_ids)))
        n = len(self.items)
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)

    def means(self, df: pd.DataFrame, value: str, cells: list[str] | None = None) -> dict:
        """Cell means with paired bootstrap replicates. Returns {cell_key: Est}."""
        if df is None or df.empty:
            return {}
        d = df
        if not cells:
            d = df.assign(_all="all")
            cells = ["_all"]
        g = d.groupby(["item_id"] + cells, observed=True)[value].agg(["sum", "count"])
        S = g["sum"].unstack(cells).reindex(self.items).fillna(0.0)
        C = g["count"].unstack(cells).reindex(self.items).fillna(0.0)
        tot_c = C.values.sum(0)
        point = np.where(tot_c > 0, S.values.sum(0) / np.maximum(tot_c, 1e-12), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (self.W @ S.values) / (self.W @ C.values)
        out = {}
        for j, col in enumerate(S.columns):
            out[col] = Est(float(point[j]), boot[:, j])
        return out

    def mean(self, df, value) -> Est | None:
        m = self.means(df, value)
        return m.get("all")

    def kappa(self, df) -> Est | None:
        """Cohen's kappa between gold slot and verdict (A/B/C, unparseable -> C)."""
        if df.empty:
            return None
        pred = df["verdict"].fillna("C")
        cats = [("A", "A"), ("A", "B"), ("A", "C"), ("B", "A"), ("B", "B"), ("B", "C")]
        cols = {f"{g}{p}": ((df["correct_slot"] == g) & (pred == p)).astype(float) for g, p in cats}
        M = pd.DataFrame(cols).assign(item_id=df["item_id"].values).groupby("item_id").sum()
        M = M.reindex(self.items).fillna(0.0)

        def k(arr):
            aa, ab, ac, ba, bb, bc = [arr[..., i] for i in range(6)]
            n = aa + ab + ac + ba + bb + bc
            po = (aa + bb) / n
            ga, gb = (aa + ab + ac) / n, (ba + bb + bc) / n
            pa, pb = (aa + ba) / n, (ab + bb) / n
            pe = ga * pa + gb * pb
            return (po - pe) / (1 - pe)

        with np.errstate(invalid="ignore", divide="ignore"):
            point = float(k(M.values.sum(0)))
            boot = k(self.W @ M.values)
        return Est(point, boot)


# =============================================================================== loading / views
def load_trials(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df["self_cond"] = np.where(df.cg_family == df.judge_family, "self_right",
                               np.where(df.wg_family == df.judge_family, "self_wrong", "neutral"))
    df["wrong_in_A"] = (df.correct_slot == "B").astype(int)
    return df


def _combine_swap(g):
    mn, mx = g.min(), g.max()
    return mn if mn == mx else 0.5


def view_swap(df: pd.DataFrame) -> pd.DataFrame:
    """Conservative swap aggregation: a preference counts only if it survives order reversal."""
    keys = [c for c in KEY_COLS if c in df.columns and c != "correct_slot"]
    g = df.groupby(keys, observed=True)["is_correct"]
    out = g.agg(["min", "max", "count"]).reset_index()
    out = out[out["count"] == 2]
    out["is_correct"] = np.where(out["min"] == out["max"], out["min"], 0.5)
    out["correct_slot"] = "both"
    return out.drop(columns=["min", "max", "count"])


def view_jury(df: pd.DataFrame, recusal: bool) -> pd.DataFrame:
    """Majority vote over judges; recusal drops judges from either response's model family."""
    d = df
    if recusal:
        d = df[(df.judge_family != df.cg_family) & (df.judge_family != df.wg_family)]
    keys = ["protocol", "block", "item_id", "domain", "pair_idx", "cg", "wg", "cg_family", "wg_family",
            "length_cond", "attack", "correct_slot"]
    m = d.groupby(keys, observed=True)["is_correct"].mean().reset_index()
    m["is_correct"] = np.where(m.is_correct > 0.5, 1.0, np.where(m.is_correct < 0.5, 0.0, 0.5))
    m["judge"] = "recusal" if recusal else "jury"
    m["judge_family"] = "panel"
    other = m.correct_slot.map({"A": "B", "B": "A"})
    m["verdict"] = np.where(m.is_correct == 1.0, m.correct_slot, np.where(m.is_correct == 0.0, other, "C"))
    return m


def build_view(df: pd.DataFrame, protocol: str, agg: str) -> pd.DataFrame:
    d = df[df.protocol == protocol]
    if d.empty:
        return d
    if agg == "single":
        return d
    if agg == "swap":
        return view_swap(d)
    if agg == "jury":
        return view_jury(d, recusal=False)
    if agg == "recusal":
        return view_jury(d, recusal=True)
    if agg == "recusal+swap":
        return view_swap(view_jury(d, recusal=True))
    raise ValueError(agg)


# =============================================================================== metrics
def _flip_units(bias_clean: pd.DataFrame) -> pd.DataFrame:
    lab = bias_clean.assign(lab=bias_clean.is_correct.map({1.0: "c", 0.0: "w", 0.5: "t"}))
    keys = ["judge", "protocol", "item_id", "pair_idx", "length_cond", "attack"]
    p = lab.pivot_table(index=keys, columns="correct_slot", values="lab", aggfunc="first").dropna()
    p = p.reset_index()
    p["flip"] = (p["A"] != p["B"]).astype(float)
    return p


def family_favoritism(bt: Boot, clean: pd.DataFrame) -> dict:
    """fav_f = acc(family f wrote the correct answer) - acc(family f wrote the incorrect answer).
    For a single judge and f = its own family this is exactly SPB. Note that the *signed* mean of fav_f
    over families is zero by construction in the balanced cyclic design, so panels are summarised by the
    mean absolute favouritism instead (see mean_abs)."""
    fams = sorted(set(clean.cg_family) & set(clean.wg_family))
    parts = []
    for f in fams:
        parts.append(clean[clean.cg_family == f].assign(cell=f"cg:{f}"))
        parts.append(clean[clean.wg_family == f].assign(cell=f"wg:{f}"))
    if not parts:
        return {}
    m = bt.means(pd.concat(parts), "is_correct", ["cell"])
    return {f: m[f"cg:{f}"] - m[f"wg:{f}"] for f in fams if f"cg:{f}" in m and f"wg:{f}" in m}


def mean_abs(ests: list) -> Est | None:
    ests = [e for e in ests if e is not None]
    if not ests:
        return None
    return Est(float(np.mean([abs(e.point) for e in ests])), np.mean([np.abs(e.boot) for e in ests], axis=0))


def asr(bt: Boot, v: pd.DataFrame) -> dict:
    """Attack success rate: among units the judge got right on the clean pair, the share it gets wrong
    after a superficial edit to the incorrect response (ties count as half a success)."""
    keys = [c for c in ["judge", "protocol", "item_id", "pair_idx", "correct_slot"] if c in v.columns]
    base = v[(v.block == "bias") & (v.length_cond == "none")][keys + ["is_correct"]].rename(columns={"is_correct": "base"})
    att = v[v.block == "attack"][keys + ["attack", "is_correct"]]
    if att.empty or base.empty:
        return {}
    j = att.merge(base, on=keys)
    j = j[j.base == 1.0].assign(succ=lambda x: 1.0 - x.is_correct)
    out = {f"asr_{a}": e for a, e in bt.means(j, "succ", ["attack"]).items()}
    singles = j[j.attack.isin(SINGLE_ATTACKS)]
    if not singles.empty:
        anyv = singles.groupby(keys, observed=True)["succ"].max().reset_index()
        out["asr_any"] = bt.mean(anyv, "succ")
    return out


def metrics(bt: Boot, v: pd.DataFrame, per_judge_self: bool) -> dict:
    """All bias metrics for one view (one judge or one aggregated panel)."""
    out: dict = {}
    if v.empty:
        return out
    bias = v[v.block == "bias"]
    clean = bias[bias.length_cond == "none"]
    out["acc"] = bt.mean(clean, "is_correct")
    out["acc_all"] = bt.mean(bias, "is_correct")
    has_slots = "verdict" in v.columns and not (v.correct_slot == "both").any()
    if has_slots:
        out["kappa"] = bt.kappa(clean)
        pp = clean.assign(pp=clean.verdict.map({"A": 1.0, "B": -1.0}).fillna(0.0))
        out["ppi"] = bt.mean(pp, "pp")
        out["flip"] = bt.mean(_flip_units(clean), "flip")
        out["tie_rate"] = bt.mean(clean.assign(t=(clean.is_correct == 0.5).astype(float)), "t")
    L = bt.means(bias, "is_correct", ["length_cond"])
    if {"none", "pad_wrong"} <= set(L):
        out["vb"] = L["none"] - L["pad_wrong"]
    if {"pad_right", "pad_wrong"} <= set(L):
        out["vb_sym"] = L["pad_right"] - L["pad_wrong"]
    if per_judge_self and "self_cond" in v.columns:
        S = bt.means(clean, "is_correct", ["self_cond"])
        if {"self_right", "self_wrong"} <= set(S):
            out["spb"] = S["self_right"] - S["self_wrong"]
        per_j = []
        for _, cj in clean.groupby("judge"):
            Sj = bt.means(cj, "is_correct", ["self_cond"])
            if {"self_right", "self_wrong"} <= set(Sj):
                per_j.append(Sj["self_right"] - Sj["self_wrong"])
        out["spb_abs"] = mean_abs(per_j)
        SL = bt.means(bias, "is_correct", ["self_cond", "length_cond"])
        need = [("self_wrong", "none"), ("neutral", "pad_wrong"), ("neutral", "none"), ("self_wrong", "pad_wrong")]
        if all(k in SL for k in need):
            out["ic_self_len"] = (SL[("self_wrong", "none")] + SL[("neutral", "pad_wrong")]
                                  - SL[("neutral", "none")] - SL[("self_wrong", "pad_wrong")])
    else:
        out["spb_abs"] = mean_abs(list(family_favoritism(bt, clean).values()))
    out.update(asr(bt, v))
    return {k: e for k, e in out.items() if e is not None}


def competence_split(bt: Boot, v: pd.DataFrame, solo: pd.DataFrame) -> dict:
    """Bias metrics split by whether the judge answers the item correctly on its own."""
    s = solo.rename(columns={"model": "judge", "correct": "knows"})[["judge", "item_id", "knows"]]
    d = v.merge(s, on=["judge", "item_id"], how="left")
    out = {}
    for k, name in [(1, "known"), (0, "unknown")]:
        sub = d[d.knows == k]
        m = metrics(bt, sub, per_judge_self=True)
        out[name] = {kk: m[kk] for kk in ("acc", "vb", "spb", "asr_any", "flip") if kk in m}
        out[name]["n_items"] = sub.item_id.nunique()
    return out


# =============================================================================== stability
def kendall_w(rank_matrix: np.ndarray) -> float:
    """rank_matrix: m raters (domains) x n objects (judges)."""
    m, n = rank_matrix.shape
    R = rank_matrix.sum(0)
    S = ((R - R.mean()) ** 2).sum()
    return float(12 * S / (m ** 2 * (n ** 3 - n)))


def kendall_w_test(values: np.ndarray, n_perm: int = 20000, seed: int = 0) -> tuple[float, float]:
    """values: domains x judges. Ranks within each domain; permutation p-value for W."""
    from scipy.stats import rankdata
    R = np.vstack([rankdata(r) for r in values])
    w = kendall_w(R)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        P = np.vstack([rng.permutation(r) for r in R])
        cnt += kendall_w(P) >= w - 1e-12
    return w, (cnt + 1) / (n_perm + 1)


# =============================================================================== regression (RQ1)
def interaction_regression(df_single_base: pd.DataFrame) -> pd.DataFrame:
    """Linear probability model with judge fixed effects and item-clustered SEs (bias block).
    Interactions on the probability scale are the natural test of (super-)additivity."""
    import statsmodels.formula.api as smf
    d = df_single_base[df_single_base.block == "bias"].copy()
    d["SW"] = (d.self_cond == "self_wrong").astype(int)
    d["SR"] = (d.self_cond == "self_right").astype(int)
    d["PW"] = (d.length_cond == "pad_wrong").astype(int)
    d["PR"] = (d.length_cond == "pad_right").astype(int)
    d["WA"] = d.wrong_in_A
    formula = "is_correct ~ WA + SW + SR + PW + PR + SW:PW + WA:PW + WA:SW + C(judge)"
    groups = pd.factorize(d.item_id)[0]
    fit = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": groups})
    rows = []
    for term in ["Intercept", "WA", "SW", "SR", "PW", "PR", "SW:PW", "WA:PW", "WA:SW"]:
        lo, hi = fit.conf_int().loc[term]
        rows.append({"term": term, "coef": fit.params[term], "lo": lo, "hi": hi, "p": fit.pvalues[term]})
    return pd.DataFrame(rows)


# =============================================================================== best-of-N selection
def select_metrics(bt: Boot, sel: pd.DataFrame) -> dict:
    out = {}
    if sel is None or sel.empty:
        return out
    sel = sel.assign(pos1=(sel.pick == 1).astype(float), is_correct=sel.is_correct.astype(float),
                     picked_perturbed=sel.picked_perturbed.astype(float))
    for (proto, judge), d in list(sel.groupby(["protocol", "judge"])) + [((p, "pooled"), d) for p, d in sel.groupby("protocol")]:
        acc = bt.means(d, "is_correct", ["cond"])
        cap = bt.means(d, "picked_perturbed", ["cond"])
        clean = d[d.cond == "clean"]
        n = int(d.n_candidates.iloc[0])
        r = {f"acc_{c}": e for c, e in acc.items()}
        r.update({f"capture_{c}": e for c, e in cap.items()})
        e = bt.mean(clean, "pos1")
        if e is not None:
            r["pos1_excess"] = Est(e.point - 1.0 / n, e.boot - 1.0 / n)
        out[(proto, judge)] = r
    return out
