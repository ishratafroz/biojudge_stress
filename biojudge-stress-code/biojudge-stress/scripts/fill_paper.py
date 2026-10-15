"""Fill the manuscript template from a finished run's metrics.json.

Nothing numeric is typed into the manuscript by hand: the template carries named slots and this script
replaces each one with the value biojudge.report computed, then refuses to write the output if a slot is
left over or a name does not resolve. Table bodies are generated here too, from the same file.

  python scripts/fill_paper.py --metrics runs/local/results/metrics.json \
      --template paper/biojudge_stress_template.tex --out ../../biojudge_stress.tex
  python scripts/fill_paper.py ... --check      # compare against the existing output, write nothing

Slot forms
  @@Name@@                      a macro from metrics.json["macros"]
  @@PJ:judge:metric@@           per-judge point estimate (PJCI / PJP for interval and p)
  @@CORR:config:metric@@        pooled estimate for one configuration (CORRCI / CORRP)
  @@DOM:domain:judge:metric@@   per-domain per-judge estimate (DOMCI)
  @@SEL:protocol:key@@          best-of-N estimate (SELCI)
  @@VG:judge@@                  verification gain (VGCI / VGP), @@SOLO:judge@@ its solo accuracy
  @@IMP:config:metric@@         signed improvement over the baseline (IMPCI for its interval)
  @@EQ:field@@                  equivalence-test field (se, mde, bound, p_tost)
  @@WORDS:display@@             mean rationale length of one author family
  @@FB:kind@@                   normaliser fallback share for one variant kind
  @@TABLE:name@@                a generated table body
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

SLOT = re.compile(r"@@([A-Za-z0-9_:.+\-]+)@@")
JUDGE_ORDER = ["llama", "qwen", "gemma", "mistral"]
DISPLAY = {"llama": "Llama-3.2-3B", "qwen": "Qwen2.5-3B", "gemma": "Gemma-2-2B", "mistral": "Ministral-3-3B"}
DOMAINS = ["literature", "clinical", "pharmacology"]
CONFIG_ROWS = [  # (macro suffix in metrics.json, row label, indent group)
    ("Base", "Single judge (baseline)", "base"),
    ("Swap", "\\quad + Swap aggregation", "single"),
    ("Debias", "\\quad + Debiasing prompt", "single"),
    ("Norm", "\\quad + Normalizer agent", "single"),
    ("DebNorm", "\\quad + Debias + Normalizer", "single"),
    ("Strip", "\\qquad \\textit{strip-only ablation}", "single"),
    ("Jury", "\\quad Cross-family jury (4)", "panel"),
    ("Recusal", "\\quad Recusal jury", "panel"),
    ("Full", "\\quad Full pipeline (D+N+R+S)", "panel"),
]
EDIT_ROWS = [("authority", "Authority (fake credentials)"), ("citation", "Fabricated citations"),
             ("format", "Markdown formatting"), ("confidence", "Asserted confidence"),
             ("injection", "\\textbf{Note to evaluator}"), ("combo", "All combined"),
             ("any", "Any single edit"), ("placebo", "\\textit{Control: placebo sentence}"),
             ("consensus", "\\textit{Control: held-out edit}")]
REG_LABELS = [("WA", "\\textbf{Wrong answer in slot A}"), ("SW", "\\textbf{Self-wrong}"),
              ("SR", "Self-right"), ("PW", "\\textbf{Pad wrong}"), ("PR", "\\textbf{Pad right}"),
              ("SW:PW", "Self-wrong $\\times$ Pad wrong"), ("WA:PW", "\\textbf{Slot A $\\times$ Pad wrong}"),
              ("WA:SW", "Slot A $\\times$ Self-wrong")]


# --------------------------------------------------------------------------- small formatters
def pct(x, digits=1):
    return "--" if x is None else f"{100 * x:.{digits}f}"


def minus(s: str) -> str:
    """LaTeX minus that does not break a table column."""
    return s.replace("-", "\\textminus{}") if s.startswith("-") else s


def hw(e, digits=1, scale=100):
    return f"\\hw{{{scale * (e['hi'] - e['lo']) / 2:.{digits}f}}}"


def sig(e) -> bool:
    return e is not None and (e["lo"] > 0 or e["hi"] < 0)


def cellv(e, digits=1, scale=100, bold_if_sig=False, bold=False):
    if e is None:
        return "-- & "
    v = f"{scale * e['est']:.{digits}f}"
    if bold or (bold_if_sig and sig(e)):
        v = f"\\textbf{{{minus(v)}}}"
    else:
        v = minus(v)
    return f"{v} & {hw(e, digits, scale)}"


def ci(e, scale=100, digits=1):
    if e is None:
        return "--"
    return f"[{scale * e['lo']:.{digits}f}, {scale * e['hi']:.{digits}f}]".replace("-", "$-$")


def ci_math(e, scale=100, digits=1):
    """Interval for use inside math mode, where a bare hyphen is not a minus sign."""
    if e is None:
        return "--"
    return f"[{scale * e['lo']:.{digits}f}, {scale * e['hi']:.{digits}f}]".replace("-", "{-}")


def pv(e):
    p = e["p"] if isinstance(e, dict) else e
    if p is None:
        return "--"
    return "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"


# --------------------------------------------------------------------------- resolvers
class Filler:
    def __init__(self, m: dict):
        self.m = m
        self.macros = m["macros"]
        self.used, self.missing = set(), []

    def get(self, name: str) -> str | None:
        parts = name.split(":")
        head = parts[0]
        try:
            if head == "WORD" and len(parts) > 1:
                v = self.get(":".join(parts[1:]))
                if v is None:
                    return None
                return {"0": "none", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
                        "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"}.get(v.strip(), v)
            if head in ("ABS", "NEG") and len(parts) > 1:
                v = self.get(":".join(parts[1:]))
                if v is None:
                    return None
                if head == "ABS":
                    return v.lstrip("-").replace("$-$", "").replace("\\textminus{}", "")
                return "$-$" + v[1:] if v.startswith("-") else v
            if len(parts) == 1:
                return self.macros.get(head)
            if head in ("PJ", "PJCI", "PJP"):
                e = self.m["per_judge"][parts[1]][parts[2]]
                return {"PJ": pct(e["est"]), "PJCI": ci(e), "PJP": pv(e)}[head]
            if head in ("CORR", "CORRCI", "CORRP"):
                e = self.m["corrections"][parts[1]][parts[2]]
                return {"CORR": pct(e["est"]), "CORRCI": ci(e), "CORRP": pv(e)}[head]
            if head in ("DOM", "DOMCI"):
                e = self.m["domain"][parts[1]][parts[2]][parts[3]]
                return {"DOM": pct(e["est"]), "DOMCI": ci(e)}[head]
            if head in ("SEL", "SELCI"):
                e = self.m["select"][f"{parts[1]}|pooled"][parts[2]]
                return {"SEL": pct(e["est"]), "SELCI": ci(e)}[head]
            if head in ("VG", "VGCI", "VGP", "SOLO"):
                d = self.m["verification_gain"][parts[1]]
                return {"VG": pct(d["vg"]["est"]), "VGCI": ci(d["vg"]), "VGP": pv(d["vg"]),
                        "SOLO": pct(d["solo"]["est"])}[head]
            if head in ("IMP", "IMPCI"):
                e = self.m["improvements"][parts[1]][parts[2]]
                return {"IMP": pct(e["est"]), "IMPCI": ci(e)}[head]
            if head == "EQ":
                v = self.m["ic_equivalence"][parts[1]]
                return pv(v) if parts[1] == "p_tost" else pct(v)
            if head == "WORDS":
                return f"{self.m['words_by_author'][parts[1]]:.0f}"
            if head == "FB":
                return pct(self.m["normalizer_fallback_by_kind"][parts[1]])
            if head == "NSPLIT":
                e = self.m["normalizer_split"][parts[1]][parts[2]]
                return ci(e) if len(parts) > 3 else pct(e["est"])
            if head == "TABLE":
                return TABLES[parts[1]](self.m)
        except (KeyError, IndexError, TypeError):
            return None
        return None

    def fill(self, text: str) -> str:
        def sub(mo):
            name = mo.group(1)
            v = self.get(name)
            if v is None:
                self.missing.append(name)
                return mo.group(0)
            self.used.add(name)
            return v
        return SLOT.sub(sub, text)


# --------------------------------------------------------------------------- table bodies
def t_data(m):
    names = {"literature": "Literature & PubMedQA", "clinical": "Clinical & MedQA-USMLE",
             "pharmacology": "Pharmacology & MedMCQA"}
    rows, drawn, kept, hard = [], 0, 0, []
    for d in DOMAINS:
        r = m["data_table"][d]
        rows.append(f"{names[d]} & {r['drawn']} & {r['kept']} & {100 * r['hard']:.1f} & "
                    f"{r['words']:.0f} & {r['pad_ratio']:.2f} \\\\")
        drawn += r["drawn"]
        kept += r["kept"]
        hard.append(100 * r["hard"] * r["drawn"])
    rows.append("\\midrule")
    rows.append(f"\\textbf{{All}} & & \\textbf{{{drawn}}} & \\textbf{{{kept}}} & "
                f"\\textbf{{{sum(hard) / drawn:.1f}}} & -- & -- \\\\")
    return "\n".join(rows)


def t_bias(m):
    rows = []
    for j in JUDGE_ORDER:
        d = m["per_judge"][j]
        cells = [cellv(d.get("acc")), cellv(d.get("kappa"), digits=2, scale=1),
                 cellv(d.get("flip")), cellv(d.get("ppi"), bold_if_sig=True),
                 cellv(d.get("vb"), bold_if_sig=True), cellv(d.get("spb"), bold_if_sig=True),
                 cellv(d.get("ic_self_len"), bold_if_sig=True), cellv(d.get("asr_combo")),
                 cellv(d.get("asr_any"))]
        rows.append(f"{DISPLAY[j]} & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows)


def t_corrections(m):
    cols = ["acc", "flip", "vb", "spb_abs", "asr_combo", "asr_any"]
    C = m["corrections"]
    singles = [s for s, _, g in CONFIG_ROWS if g in ("base", "single") and s in C]
    best = {}
    for c in cols:
        pool = [s for s in C if c in C[s]]
        if not pool:
            continue
        if c == "acc":
            best[c] = max(pool, key=lambda s: C[s][c]["est"])
        elif c in ("vb", "spb_abs"):
            cand = [s for s in (singles if c == "spb_abs" else pool) if c in C[s]]
            best[c] = min(cand, key=lambda s: abs(C[s][c]["est"]))
        else:
            best[c] = min(pool, key=lambda s: C[s][c]["est"])
    rows = []
    for suf, label, group in CONFIG_ROWS:
        if suf not in C:
            continue
        d = C[suf]
        cells = []
        for c in cols:
            e = d.get(c)
            if e is None:
                cells.append("\\multicolumn{2}{c}{--}")
            else:
                cells.append(cellv(e, bold=(best.get(c) == suf)))
        if suf == "Base":
            cells.append("\\multicolumn{2}{c}{--}")
        else:
            dacc = m["corrections"][suf].get("_dacc")
            if not dacc:
                cells.append("\\multicolumn{2}{c}{--}")
            else:
                best_d = max((s2 for s2, _, _ in CONFIG_ROWS if "_dacc" in C.get(s2, {})),
                             key=lambda s2: C[s2]["_dacc"]["est"], default=None)
                v = minus(f"{100 * dacc['est']:.1f}")
                if sig(dacc):
                    v += "$^{*}$"
                if best_d == suf:
                    v = f"\\textbf{{{v}}}"
                cells.append(f"{v} & {hw(dacc)}")
        pre = ""
        if group == "single" and suf == "Swap":
            pre = "\\multicolumn{15}{@{}l}{\\textit{Single-judge corrections}} \\\\\n"
        if group == "panel" and suf == "Jury":
            pre = "\\midrule\n\\multicolumn{15}{@{}l}{\\textit{Panel corrections}} \\\\\n"
        if suf == "Swap":
            pre = "\\midrule\n" + pre
        rows.append(pre + f"{label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows)


def t_domain(m):
    rows = []
    best = {}
    for c, how in (("acc", max), ("vb", "abs"), ("spb", "abs"), ("asr_any", min)):
        for d in DOMAINS:
            vals = {j: m["domain"][d][j][c]["est"] for j in JUDGE_ORDER if c in m["domain"][d][j]}
            if not vals:
                continue
            best[(c, d)] = (min(vals, key=lambda j: abs(vals[j])) if how == "abs"
                            else how(vals, key=lambda j: vals[j]))
    for j in JUDGE_ORDER:
        cells = []
        for c in ("acc", "vb", "spb", "asr_any"):
            for d in DOMAINS:
                e = m["domain"][d][j].get(c)
                v = minus(pct(e["est"])) if e else "--"
                if best.get((c, d)) == j:
                    v = f"\\textbf{{{v}}}"
                cells.append(v)
        rows.append(f"{DISPLAY[j]} & " + " & ".join(cells) + " \\\\")
    kw = []
    for c in ("Acc", "VB", "SPB", "ASR"):
        w, p = m["macros"].get(f"KW{c}"), m["macros"].get(f"KW{c}P", "")
        txt = f"{w} ({p.replace('p', '$p$')})" if w else "--"
        if c == "Acc":
            txt = f"\\textbf{{{txt}}}"
        kw.append(f"\\multicolumn{{3}}{{c}}{{{txt}}}")
    rows.append("\\midrule\nKendall's $W$ ($p$) & " + " & ".join(kw) + " \\\\")
    return "\n".join(rows)


def t_regression(m):
    r = m["regression"]
    r2 = m.get("regression_length", {})
    rows = []
    for t, lab in REG_LABELS:
        d = r[t]
        p = "{\\boldmath$<0.001$}" if d["p"] < 0.001 else (f"\\textbf{{{d['p']:.3f}}}" if d["p"] < 0.05
                                                           else f"${d['p']:.3f}$")
        coef = f"{100 * d['coef']:.1f}"
        coef = f"\\textbf{{{minus(coef)}}}" if d["p"] < 0.05 else minus(coef)
        adj = f"{100 * r2[t]['coef']:.1f}" if t in r2 else "--"
        rows.append(f"{lab} & {coef} & ${ci_math(d)}$ & {p} & {minus(adj)} \\\\")
        if t == "PR":
            rows.append("\\midrule")
    if "LR" in r2:
        d = r2["LR"]
        rows.append("\\midrule\n\\textit{log length ratio (model 2)} & -- & -- & -- & "
                    f"\\textbf{{{minus(f'{100 * d[chr(99)+chr(111)+chr(101)+chr(102)]:.1f}')}}} \\\\")
    return "\n".join(rows)


def t_attacks(m):
    C = m["corrections"]
    cfgs = ["Base", "Debias", "Norm", "DebNorm", "Jury", "Full"]
    rows = []
    for e, lab in EDIT_ROWS:
        key = f"asr_{e}"
        vals = {c: C.get(c, {}).get(key) for c in cfgs}
        avail = {c: v for c, v in vals.items() if v}
        bestc = min(avail, key=lambda c: avail[c]["est"]) if avail else None
        cells = [cellv(vals[c], digits=0, bold=(c == bestc)) if vals[c] else "\\multicolumn{2}{c}{--}"
                 for c in cfgs]
        for pre in ("rr", "net", "dacc", "sym"):
            v = C["Base"].get(f"{pre}_{e}")
            cells.append(minus(pct(v["est"], 1)) if v else "--")
        pre = "\\midrule\n" if e in ("combo", "any", "placebo") else ""
        rows.append(pre + f"{lab} & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows)


TABLES = {"data": t_data, "bias": t_bias, "corrections": t_corrections, "domain": t_domain,
          "regression": t_regression, "attacks": t_attacks}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="compare with --out instead of writing it")
    ap.add_argument("--report", help="write the slot -> value list here (provenance for every number)")
    a = ap.parse_args()

    with open(a.metrics) as f:
        m = json.load(f)
    # the paired delta-accuracy of each configuration, needed by the corrections table
    base = m["corrections"].get("Base", {}).get("acc")
    for suf, d in m["corrections"].items():
        if suf != "Base" and "acc" in d and base:
            dm = m["macros"].get(f"DeltaAcc{suf}")
            lo, hi = (m["macros"].get(f"DeltaAcc{suf}CI") or "[,]").strip("[]").split(",")
            try:
                d["_dacc"] = {"est": float(dm) / 100, "lo": float(lo) / 100, "hi": float(hi) / 100, "p": None}
            except ValueError:
                pass
    with open(a.template) as f:
        tpl = f.read()
    fl = Filler(m)
    out = fl.fill(tpl)
    left = sorted(set(fl.missing))
    if left:
        print(f"[fill_paper] {len(left)} unresolved slot(s):", file=sys.stderr)
        for n in left:
            print(f"    @@{n}@@", file=sys.stderr)
        raise SystemExit(1)
    if a.report:
        with open(a.report, "w") as f:
            for n in sorted(fl.used):
                if not n.startswith("TABLE"):
                    f.write(f"{n}\t{fl.get(n)}\n")
    if a.check:
        cur = open(a.out).read() if os.path.exists(a.out) else ""
        same = cur == out
        print(f"[fill_paper] {a.out} is {'up to date' if same else 'OUT OF DATE'} "
              f"({len(fl.used)} slots resolved)")
        raise SystemExit(0 if same else 2)
    with open(a.out, "w") as f:
        f.write(out)
    print(f"[fill_paper] wrote {a.out}: {len(fl.used)} slots resolved, {len(SLOT.findall(out))} left")


if __name__ == "__main__":
    main()
