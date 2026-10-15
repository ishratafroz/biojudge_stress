"""Turn a finished run into everything the paper needs.

  python -m biojudge.report --run runs/main --config configs/openrouter.yaml --paper paper/
  python -m biojudge.report --placeholders --paper paper/     # compile the paper before results exist

Writes paper/results_macros.tex (every number quoted in the text), paper/tables/*.tex,
paper/figs/*.pdf, and <run>/results/{metrics.json,*.csv}.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

from .analysis import (CONTROL_EDITS, SINGLE_ATTACKS, Boot, Est, build_view, competence_split,
                       equivalence, holm, interaction_regression, kendall_w_test, load_trials, metrics,
                       select_metrics, tie_rule_acc, verification_gain)
from .config import load_config
from .pipeline import read_jsonl

CONFIGS = [  # (macro suffix, table label, protocol, aggregation)
    ("Base", "Single judge (baseline)", "base_raw", "single"),
    ("Swap", "+ Swap aggregation", "base_raw", "swap"),
    ("Debias", "+ Debiasing prompt", "debias_raw", "single"),
    ("Norm", "+ Normalizer agent", "base_norm", "single"),
    ("DebNorm", "+ Debias prompt + Normalizer", "debias_norm", "single"),
    ("Strip", "\\quad \\textit{strip-only ablation}", "base_strip", "single"),
    ("Jury", "Cross-family jury (4)", "base_raw", "jury"),
    ("Recusal", "Recusal jury", "base_raw", "recusal"),
    ("Full", "Full NRS pipeline", "debias_norm", "recusal+swap"),
]
CORR_METRICS = ["acc", "flip", "vb", "spb_abs", "asr_combo", "asr_any"]
MNAME = {"acc": "Acc", "flip": "Flip", "vb": "VB", "spb_abs": "SPB", "asr_combo": "ASRcombo", "asr_any": "ASRany"}
ATTACK_MACRO = {"authority": "ASRauthBase", "citation": "ASRcitBase", "format": "ASRfmtBase",
                "confidence": "ASRconfBase", "injection": "ASRinjBase"}
DOMAIN_ORDER = ["literature", "clinical", "pharmacology"]
DOMAIN_SHORT = {"literature": "Lit.", "clinical": "Clin.", "pharmacology": "Pharm."}


# =============================================================================== formatting
def pct(x):
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:.1f}"


def ci_str(e: Est | None):
    if e is None:
        return "--"
    lo, hi = e.ci()
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]".replace("-", "$-$")


def p_str(p):
    if p is None or not np.isfinite(p):
        return "--"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def cell(e: Est | None, scale=100, digits=1, sig=True):
    if e is None or not np.isfinite(e.point):
        return "--"
    lo, hi = e.ci()
    hw = (hi - lo) / 2 * scale
    s = f"{e.point * scale:.{digits}f}"
    if sig and np.isfinite(lo) and (lo > 0 or hi < 0) and e is not None:
        s = f"\\textbf{{{s}}}"
    return f"{s}{{\\scriptsize$\\pm${hw:.{digits}f}}}"


def tex_escape(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


# =============================================================================== macro registry
def macro_names(judges_n: int = 4) -> list[str]:
    names = ["NItemsSampled", "NItemsKept", "NItemsLit", "NItemsClin", "NItemsPharm", "NPairwise", "NSelect",
             "ValidRate", "HardNegRate", "PadLLMRate", "PadRatio", "NormLLMRate", "ParseFailRate", "TieRate",
             "ParseFailMin", "ParseFailMax", "ParseFailJudgeMin", "ParseFailJudgeMax",
             "RunHours", "SecPerKCall", "CallsPerSec", "NCallsMade", "NCallsTotal", "OllamaVersion", "JudgeCallsPerSec", "JudgeSecPerKCall",
             "NMacrosMain", "PhasePadHours", "PhaseJudgeHours", "PhaseCtrlHours", "RetestFlipPct", "RegWA", "RegSW", "RegSR", "RegPW", "RegPR",
             "AccBaseCI", "KappaBase", "VBBaseCI", "SPBBaseCI", "ICBase", "ICBaseCI", "ICBaseP", "VBsymBase",
             "ASRauthBase", "ASRcitBase", "ASRfmtBase", "ASRconfBase", "ASRinjBase",
             "FlipMin", "FlipMax", "VBMin", "VBMax", "SPBMin", "SPBMax", "ASRanyMin", "ASRanyMax",
             "JudgeMaxVB", "JudgeMaxSPB", "JudgeMaxASR", "JudgeBestAcc", "AccBest", "TopAttack", "TopAttackASR",
             "RegSWPW", "RegSWPWCI", "RegSWPWP", "RegWAPW", "RegWAPWP", "RegWASW", "RegWASWP",
             "KWAcc", "KWAccP", "KWVB", "KWVBP", "KWSPB", "KWSPBP", "KWASR", "KWASRP",
             "AccKnown", "AccUnknown", "VBKnown", "VBUnknown", "SPBKnown", "SPBUnknown", "ASRKnown", "ASRUnknown",
             "FlipKnown", "FlipUnknown",
             # tie rules (A2)
             "AccStrictBase", "AccStrictBaseCI", "AccDropBase", "AccDropBaseCI", "DecidedShareBase",
             # solo competence and verification gain (A3)
             "SoloPhi", "SoloMin", "SoloMax", "VGMin", "VGMax", "VGJudgeBest", "VGBest", "VGBestCI",
             "VGPooled", "VGPooledCI", "VGPooledTwoOpt", "VGPooledTwoOptCI", "VGPooledFourOpt",
             "VGPooledFourOptCI",
             # rationale length by author family and the length covariate (A4)
             "WordsMin", "WordsMax", "WordsSpread", "RegLR", "RegLRP", "RegSWadj", "RegPWadj",
             # normaliser mechanism (A5)
             "DAccNormPhiKnown", "DAccNormPhiKnownCI", "DAccNormPhiUnknown", "DAccNormPhiUnknownCI",
             "NPhiKnownItems", "NPhiUnknownItems",
             "AccNormRewrite", "AccNormFallback", "NormFallbackPad", "NormFallbackRaw", "NormFallbackCombo",
             # equivalence and multiplicity (A8, A10)
             "ICSE", "ICMDE", "ICBound", "ICTOSTP", "HolmIC", "HolmDAccFull", "HolmKWAcc", "HolmASRany",
             # controls (B1-B4)
             "ASRplaceboBase", "ASRplaceboBaseCI", "ASRconsBase", "ASRconsBaseCI", "ASRconsDebNorm",
             "ASRconsDebNormCI", "NEditsAbovePlacebo", "AdjInjectionBase", "AdjInjectionBaseCI",
             "SymAnyBase", "SymAnyBaseCI", "SymInjectionBase", "RRAnyBase", "RRAnyBaseCI",
             "NetAnyBase", "NetAnyBaseCI", "RRInjectionBase", "NetInjectionBase",
             "NetInjectionBaseCI", "DAccInjectionBase", "DAccInjectionBaseCI", "NCtrlVerdicts",
             # best-of-N, scoring rule made explicit
             "SelParseRate", "SelChanceScored", "SelParseRateDN", "SelChanceScoredDN",
             "SelAccCleanCI", "SelAccAttackCI", "SelCapAttackCI", "SelAccAttackParsed", "SelPickPosMax",
             "SelPickPosMaxShare",
             # repeat-decode reliability
             "RetestVerdict", "RetestText", "RetestAcc", "RetestN", "RetestVerdictMin",
             "RetestVerdictMax", "RetestJudgeMin", "RetestJudgeMax",
             "ParseAuditN", "ParseAuditFails", "ParseAuditStop", "ParseAuditMaxTok", "ParseAuditCap",
             "ParseAuditReparse"]
    for suf, *_ in CONFIGS:
        for m in CORR_METRICS:
            names.append(f"{MNAME[m]}{suf}")
            if suf != "Base":
                names += [f"Delta{MNAME[m]}{suf}", f"Delta{MNAME[m]}{suf}CI"]
    for tag in ("", "DN"):
        for c in ("Clean", "Pad", "Attack"):
            names += [f"SelAcc{c}{tag}", f"SelCap{c}{tag}"]
        names.append(f"SelPosOne{tag}")
    return names


def write_macros(values: dict, path: str, placeholder: bool = False):
    lines = ["% Auto-generated by biojudge.report -- do not edit by hand.",
             "\\providecommand{\\PH}{\\textcolor{red}{\\textbf{??}}}"]
    for n in macro_names():
        v = "\\PH{}" if placeholder else values.get(n, "\\PH{}")
        lines.append(f"\\newcommand{{\\{n}}}{{{v}}}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================== placeholder tables
def placeholder_tables(tdir: str, judge_names: list[str]):
    os.makedirs(tdir, exist_ok=True)
    ph = "\\PH"
    rows = "\n".join(f"{j} & " + " & ".join([ph] * 9) + " \\\\" for j in judge_names)
    _write(tdir, "tab_bias_profile.tex", _bias_profile_tex(rows))
    rows = "\n".join(f"{lab} & " + " & ".join([ph] * 7) + " \\\\" for _, lab, *_ in CONFIGS)
    _write(tdir, "tab_corrections.tex", _corrections_tex(rows))
    rows = "\n".join(f"{j} & " + " & ".join([ph] * 12) + " \\\\" for j in judge_names)
    rows += "\n\\midrule\nKendall's $W$ & " + " & ".join([f"\\multicolumn{{3}}{{c}}{{{ph}}}"] * 4) + " \\\\"
    _write(tdir, "tab_domain.tex", _domain_tex(rows))
    rows = "\n".join(f"{d} & {ph} & {ph} & {ph} & {ph} & {ph} \\\\" for d in ["Literature (PubMedQA)", "Clinical (MedQA)", "Pharmacology (MedMCQA)"])
    _write(tdir, "tab_data.tex", _data_tex(rows))
    rows = "\n".join(f"{t} & {ph} & {ph} & {ph} \\\\" for t in REG_LABELS.values())
    _write(tdir, "tab_regression.tex", _reg_tex(rows))
    rows = "\n".join(f"{lab} & " + " & ".join([ph] * len(ATTACK_COLS)) + " \\\\" for _, lab in ATTACK_ROWS)
    rows = rows.replace("All combined", "\\midrule\nAll combined", 1)
    _write(tdir, "tab_attacks.tex", _attacks_tex(rows))


def _write(d, name, s):
    with open(os.path.join(d, name), "w") as f:
        f.write(s)


def _bias_profile_tex(rows):
    return ("\\begin{table*}[t]\n\\centering\n\\caption{Bias profile of each judge (base prompt, raw responses), "
            "pooled over the three domains. Acc = agreement with the expert answer key on clean pairs (\\%); "
            "$\\kappa$ = Cohen's kappa vs.\\ gold; Flip = verdict changes under order reversal (\\%); "
            "PPI = position-preference index ($+$ favours slot A, pp); VB = accuracy lost when the incorrect "
            "response is padded (pp); SPB = accuracy when the judge's own family wrote the correct vs.\\ the "
            "incorrect answer (pp); IC = self$\\times$verbosity interaction contrast ($>0$ super-additive, pp); "
            "ASR = attack success rate (\\%). Subscripts are 95\\% cluster-bootstrap half-widths; bold marks "
            "CIs excluding 0.}\n\\label{tab:bias}\n\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{lccccccccc}\n\\toprule\n"
            "Judge & Acc & $\\kappa$ & Flip & PPI & VB & SPB & IC$_{S\\times V}$ & ASR$_\\text{combo}$ & ASR$_\\text{any}$ \\\\\n"
            "\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def _corrections_tex(rows):
    return ("\\begin{table*}[t]\n\\centering\n\\caption{Debiasing corrections applied individually and jointly, "
            "pooled over judges and domains. Single-judge rows average the four judges; jury rows aggregate them. "
            "$|$SPB$|$ is the mean absolute self-preference over judges (single-judge rows) or the mean absolute "
            "family favouritism (jury rows; Sec.~\\ref{sec:metrics}). $\\Delta$Acc is the paired "
            "change from the baseline. Flip is undefined after swap aggregation.}\n\\label{tab:corr}\n"
            "\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular}{lccccccc}\n\\toprule\n"
            "Configuration & Acc & Flip & VB & $|$SPB$|$ & ASR$_\\text{combo}$ & ASR$_\\text{any}$ & $\\Delta$Acc \\\\\n"
            "\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def _domain_tex(rows):
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{m}}}" for m in ["Acc (\\%)", "VB (pp)", "SPB (pp)", "ASR$_\\text{any}$ (\\%)"])
    sub = " & ".join(["L & C & P"] * 4)
    return ("\\begin{table*}[t]\n\\centering\n\\caption{Cross-domain stability (base protocol). L = literature "
            "(PubMedQA), C = clinical (MedQA), P = pharmacology (MedMCQA). Kendall's $W$ measures agreement of "
            "the judge ranking across the three domains (1 = identical rankings); permutation $p$ in parentheses.}\n"
            "\\label{tab:domain}\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{l" + "ccc" * 4 + "}\n\\toprule\n"
            f"Judge & {head} \\\\\n\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-7}}\\cmidrule(lr){{8-10}}\\cmidrule(lr){{11-13}}\n"
            f" & {sub} \\\\\n\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def _data_tex(rows):
    return ("\\begin{table}[t]\n\\centering\n\\caption{Evaluation items. Kept = items for which all four "
            "generators produced valid correct and incorrect rationales. Hard = distractor chosen from pooled "
            "model errors. Words = mean length of raw rationales; Pad = padded/raw length ratio.}\n"
            "\\label{tab:data}\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{lccccc}\n\\toprule\n"
            "Domain & Sampled & Kept & Hard (\\%) & Words & Pad \\\\\n\\midrule\n" + rows +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


ATTACK_ROWS = [("authority", "Authority (fake credentials)"), ("citation", "Fabricated citations"),
               ("format", "Markdown formatting"), ("confidence", "Asserted confidence"),
               ("injection", "Note to evaluator"), ("combo", "All combined"), ("any", "Any single edit"),
               ("placebo", "\\textit{Control: placebo sentence}"),
               ("consensus", "\\textit{Control: held-out edit}")]
ATTACK_COLS = [("Base", "Base"), ("Debias", "+Debias"), ("Norm", "+Norm"), ("DebNorm", "+Deb+Norm"), ("Jury", "Jury"),
               ("Full", "Full")]


def _attacks_tex(rows):
    head = " & ".join(lab for _, lab in ATTACK_COLS)
    return ("\\begin{table}[t]\n\\centering\n\\caption{Attack success rate (\\%) by content-free edit and "
            "configuration: share of verdicts that were correct on the clean pair and become incorrect once the "
            "edit is applied to the incorrect response (ties count one half). Any = broken by at least one of the "
            "five single edits. Subscripts: 95\\% cluster-bootstrap half-widths.}\n\\label{tab:attacks}\n"
            "\\setlength{\\tabcolsep}{2.2pt}\n\\footnotesize\n\\begin{tabular}{l" + "c" * len(ATTACK_COLS) + "}\n\\toprule\n"
            f"Edit & {head} \\\\\n\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


REG_LABELS = {"WA": "Wrong answer in slot A", "SW": "Self-wrong", "SR": "Self-right", "PW": "Pad wrong",
              "PR": "Pad right", "SW:PW": "Self-wrong $\\times$ Pad wrong", "WA:PW": "Slot A $\\times$ Pad wrong",
              "WA:SW": "Slot A $\\times$ Self-wrong"}


def _reg_tex(rows):
    return ("\\begin{table}[t]\n\\centering\n\\caption{Linear probability model of a correct verdict (bias "
            "block, base protocol) with judge fixed effects; SEs clustered by item. Coefficients in pp. A negative "
            "interaction means the joint manipulation costs more accuracy than the sum of its parts.}\n"
            "\\label{tab:reg}\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lccc}\n\\toprule\n"
            "Term & Coef. & 95\\% CI & $p$ \\\\\n\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def vkey_of(r: dict) -> str:
    """The variant key of the incorrect response shown in one trial (used to look up how it was built)."""
    if r["block"] == "attack_sym":
        var = "raw"
    elif r["attack"] != "none":
        var = f"atk:{r['attack']}"
    else:
        var = "pad" if r["length_cond"] == "pad_wrong" else "raw"
    return f"{r['item_id']}|{r['wg']}|-|{var}"


# =============================================================================== real report
def build_report(run: str, cfg: dict, paper: str):
    res_dir = os.path.join(run, "results")
    os.makedirs(res_dir, exist_ok=True)
    tdir, fdir = os.path.join(paper, "tables"), os.path.join(paper, "figs")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    disp = {k: v.get("display", k) for k, v in cfg["models"].items()}
    judges = cfg["roles"]["judges"]

    trials = load_trials(os.path.join(run, "trials.jsonl"))
    items = read_jsonl(os.path.join(run, "items.jsonl"))
    resp = read_jsonl(os.path.join(run, "responses.jsonl"))
    variants = read_jsonl(os.path.join(run, "variants.jsonl"))
    solo = pd.DataFrame(read_jsonl(os.path.join(run, "solo.jsonl")))
    sel_path = os.path.join(run, "select.jsonl")
    sel = pd.DataFrame(read_jsonl(sel_path)) if os.path.exists(sel_path) else pd.DataFrame()

    bt = Boot(trials.item_id.unique(), cfg["analysis"]["bootstrap"], cfg["seed"])
    M: dict[str, str] = {}
    store: dict = {}

    # ---------------------------------------------------------------- data description
    kept = set(trials.item_id)
    idf = pd.DataFrame(items)
    rdf = pd.DataFrame(resp)
    vdf = pd.DataFrame(variants)
    M["NItemsSampled"] = str(len(idf))
    M["NItemsKept"] = str(len(kept))
    for d, n in [("literature", "NItemsLit"), ("clinical", "NItemsClin"), ("pharmacology", "NItemsPharm")]:
        M[n] = str(int(idf[(idf.domain == d) & idf.item_id.isin(kept)].shape[0]))
    M["NPairwise"] = f"{len(trials):,}".replace(",", "{,}")
    M["NSelect"] = f"{len(sel):,}".replace(",", "{,}")
    M["ValidRate"] = pct(rdf.valid.mean())
    M["HardNegRate"] = pct((idf.distractor_source == "model_error").mean())
    pads = vdf[vdf.key.str.endswith("|pad")]
    M["PadLLMRate"] = pct((pads.method == "llm").mean()) if len(pads) else "--"
    norms = vdf[vdf.key.str.endswith("#norm")]
    M["NormLLMRate"] = pct((norms.method == "llm").mean()) if len(norms) else "--"
    M["ParseFailRate"] = pct(trials.parse_fail.mean())
    pf = trials.groupby("judge").parse_fail.mean()
    M["ParseFailMin"], M["ParseFailMax"] = pct(pf.min()), pct(pf.max())
    M["ParseFailJudgeMin"] = tex_escape(disp[pf.idxmin()])
    M["ParseFailJudgeMax"] = tex_escape(disp[pf.idxmax()])
    store["parse_fail_by_judge"] = {disp.get(j, j): float(v) for j, v in pf.items()}
    store["parse_fail_by_protocol"] = {k: float(v) for k, v in trials.groupby("protocol").parse_fail.mean().items()}
    prov = cfg.get("provenance") or {}
    if prov.get("ollama_version"):
        M["OllamaVersion"] = str(prov["ollama_version"])
    if prov.get("macros_reproduced"):
        M["NMacrosMain"] = str(prov["macros_reproduced"])
    store["provenance"] = prov
    # wall-clock per phase. run_meta.json is overwritten by each run, so the archived copies in
    # runs/<run>/backup_*/ and run_meta_perturb.json are read as well.
    tot_s = tot_c = 0.0
    for name, path in sorted((os.path.basename(os.path.dirname(q)) + "/" + os.path.basename(q), q)
                             for q in glob.glob(os.path.join(run, "run_meta*.json"))
                             + glob.glob(os.path.join(run, "backup_*", "run_meta*.json"))):
        try:
            with open(path) as f:
                mt = json.load(f)
        except (OSError, ValueError):
            continue
        made = sum(v for k, v in (mt.get("llm_stats") or {}).items() if k.startswith("ok:"))
        secs = float(mt.get("seconds", 0) or 0)
        store.setdefault("phases", {})[name] = {"stages": mt.get("stages"), "seconds": secs,
                                                "calls_made": made,
                                                "cache_hits": (mt.get("llm_stats") or {}).get("cache_hit", 0)}
        tot_s += secs
        tot_c += made
        stages = ",".join(mt.get("stages") or [])
        if "perturb" in stages and "judge" not in stages:
            M["PhasePadHours"] = f"{secs / 3600:.1f}"
        elif stages == "judge,select":
            M["PhaseJudgeHours"] = f"{secs / 3600:.1f}"
            if secs and made:
                M["JudgeCallsPerSec"], M["JudgeSecPerKCall"] = f"{made / secs:.1f}", f"{1000 * secs / made:.0f}"
        elif "judge" in stages and "perturb" in stages:
            M["PhaseCtrlHours"] = f"{secs / 3600:.1f}"
    if tot_s and tot_c:
        M["CallsPerSec"], M["SecPerKCall"] = f"{tot_c / tot_s:.1f}", f"{1000 * tot_s / tot_c:.0f}"
        M["NCallsTotal"] = f"{tot_c:,.0f}".replace(",", "{,}")
    mp = os.path.join(run, "run_meta.json")
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
        secs = float(meta.get("seconds", 0) or 0)
        made = sum(v for k, v in (meta.get("llm_stats") or {}).items() if k.startswith("ok:"))
        store["run_timing"] = {"seconds": secs, "calls_made": made, "cache_hits":
                               (meta.get("llm_stats") or {}).get("cache_hit", 0)}
        if secs and made:
            M["RunHours"] = f"{secs / 3600:.1f}"
            M["SecPerKCall"] = f"{1000 * secs / made:.0f}"
            M["CallsPerSec"] = f"{made / secs:.1f}"
            M["NCallsMade"] = f"{made:,}".replace(",", "{,}")
    M["TieRate"] = pct((trials.verdict == "C").mean())
    raw_w = rdf.set_index("key")["words"]
    pad_w = pads.assign(w=pads.text.str.split().str.len(), base=pads.key.str.replace("|pad", "|raw"))
    pad_w["ratio"] = pad_w.w / pad_w.base.map(raw_w)
    M["PadRatio"] = f"{pad_w.ratio.mean():.2f}" if len(pad_w) else "--"
    gen_valid = rdf.groupby("gen").valid.mean().to_dict()
    store["generator_validity"] = gen_valid

    drows = []
    names = {"literature": "Literature (PubMedQA)", "clinical": "Clinical (MedQA)", "pharmacology": "Pharmacology (MedMCQA)"}
    for d in DOMAIN_ORDER:
        sub = idf[idf.domain == d]
        if sub.empty:
            continue
        ids = set(sub.item_id)
        rw = rdf[rdf.item_id.isin(ids & kept)]
        pw = pad_w[pad_w.key.str.split("|").str[0].isin(ids & kept)]
        drows.append(f"{names[d]} & {len(sub)} & {len(ids & kept)} & {pct((sub.distractor_source == 'model_error').mean())}"
                     f" & {rw.words.mean():.0f} & {pw.ratio.mean():.2f}\\\\")
        store.setdefault("data_table", {})[d] = {
            "drawn": int(len(sub)), "kept": int(len(ids & kept)),
            "hard": float((sub.distractor_source == "model_error").mean()),
            "words": float(rw.words.mean()), "pad_ratio": float(pw.ratio.mean()),
            "n_options": int(np.median([len(o) for o in sub.options])) if "options" in sub else None}
    _write(tdir, "tab_data.tex", _data_tex("\n".join(drows)))

    # ---------------------------------------------------------------- per-judge bias profile (base_raw)
    base = trials[trials.protocol == "base_raw"]
    per_judge = {}
    brows = []
    for j in judges:
        m = metrics(bt, base[base.judge == j], per_judge_self=True)
        per_judge[j] = m
        vals = [cell(m.get("acc"), sig=False), cell(m.get("kappa"), scale=1, digits=2, sig=False), cell(m.get("flip"), sig=False),
                cell(m.get("ppi")), cell(m.get("vb")), cell(m.get("spb")), cell(m.get("ic_self_len")),
                cell(m.get("asr_combo"), sig=False), cell(m.get("asr_any"), sig=False)]
        brows.append(f"{tex_escape(disp[j])} & " + " & ".join(vals) + " \\\\")
    _write(tdir, "tab_bias_profile.tex", _bias_profile_tex("\n".join(brows)))
    store["per_judge"] = {j: {k: e.as_dict() for k, e in m.items()} for j, m in per_judge.items()}

    def rng_macro(key, lo_name, hi_name, arg_name=None, which="max"):
        vals = {j: per_judge[j][key].point for j in judges if key in per_judge[j]}
        if not vals:
            return
        M[lo_name] = pct(min(vals.values()))
        M[hi_name] = pct(max(vals.values()))
        if arg_name:
            M[arg_name] = tex_escape(disp[max(vals, key=vals.get)])

    rng_macro("flip", "FlipMin", "FlipMax")
    rng_macro("vb", "VBMin", "VBMax", "JudgeMaxVB")
    rng_macro("spb", "SPBMin", "SPBMax", "JudgeMaxSPB")
    rng_macro("asr_any", "ASRanyMin", "ASRanyMax", "JudgeMaxASR")
    accs = {j: per_judge[j]["acc"].point for j in judges if "acc" in per_judge[j]}
    if accs:
        bj = max(accs, key=accs.get)
        M["JudgeBestAcc"], M["AccBest"] = tex_escape(disp[bj]), pct(accs[bj])

    # ---------------------------------------------------------------- corrections (pooled)
    conf_metrics = {}
    for suf, label, proto, agg in CONFIGS:
        v = build_view(trials, proto, agg)
        if v.empty:
            continue
        conf_metrics[suf] = metrics(bt, v, per_judge_self=(agg in ("single", "swap")))
    b0 = conf_metrics.get("Base", {})
    crow = []
    delta_tab = {}
    for suf, label, proto, agg in CONFIGS:
        if suf not in conf_metrics:
            continue
        m = conf_metrics[suf]
        for k in CORR_METRICS:
            if k in m:
                M[f"{MNAME[k]}{suf}"] = pct(m[k].point)
                if suf != "Base" and k in b0:
                    d = m[k] - b0[k]
                    M[f"Delta{MNAME[k]}{suf}"] = pct(d.point)
                    M[f"Delta{MNAME[k]}{suf}CI"] = ci_str(d)
        dacc = (m["acc"] - b0["acc"]) if suf != "Base" and "acc" in m and "acc" in b0 else None
        vals = [cell(m.get("acc"), sig=False), cell(m.get("flip"), sig=False), cell(m.get("vb")), cell(m.get("spb_abs"), sig=False),
                cell(m.get("asr_combo"), sig=False), cell(m.get("asr_any"), sig=False), cell(dacc) if dacc else "--"]
        crow.append(f"{label} & " + " & ".join(vals) + " \\\\")
        delta_tab[suf] = {}
        for k in CORR_METRICS:
            if suf != "Base" and k in m and k in b0:
                imp = (m[k] - b0[k]) if k == "acc" else (b0[k].abs() - m[k].abs())
                delta_tab[suf][k] = imp
    _write(tdir, "tab_corrections.tex", _corrections_tex("\n".join(crow)))
    # signed improvements over the baseline: higher Acc, or smaller |VB|, |SPB|, Flip, ASR
    store["improvements"] = {s2: {k: e.as_dict() for k, e in d.items()} for s2, d in delta_tab.items()}
    arows = []
    for a, lab in ATTACK_ROWS:
        key = f"asr_{a}"
        vals = [cell(conf_metrics.get(suf, {}).get(key), digits=0, sig=False) for suf, _ in ATTACK_COLS]
        arows.append(("\\midrule\n" if a == "combo" else "") + f"{lab} & " + " & ".join(vals) + " \\\\")
    _write(tdir, "tab_attacks.tex", _attacks_tex("\n".join(arows)))
    store["corrections"] = {s: {k: e.as_dict() for k, e in m.items()} for s, m in conf_metrics.items()}

    if b0:
        M["AccBaseCI"] = ci_str(b0.get("acc"))
        M["KappaBase"] = f"{np.nanmean([per_judge[j]['kappa'].point for j in judges if 'kappa' in per_judge[j]]):.2f}"
        M["VBBaseCI"] = ci_str(b0.get("vb"))
        M["SPBBaseCI"] = ci_str(b0.get("spb_abs"))
        M["VBsymBase"] = pct(b0["vb_sym"].point) if "vb_sym" in b0 else "--"
        if "ic_self_len" in b0:
            M["ICBase"], M["ICBaseCI"], M["ICBaseP"] = pct(b0["ic_self_len"].point), ci_str(b0["ic_self_len"]), p_str(b0["ic_self_len"].p())
        for a, name in ATTACK_MACRO.items():
            if f"asr_{a}" in b0:
                M[name] = pct(b0[f"asr_{a}"].point)
        singles = {a: b0[f"asr_{a}"].point for a in ATTACK_MACRO if f"asr_{a}" in b0}
        if singles:
            ta = max(singles, key=singles.get)
            M["TopAttack"], M["TopAttackASR"] = ta, pct(singles[ta])

    # ---------------------------------------------------------------- regression (RQ1)
    try:
        reg = interaction_regression(base)
        reg.to_csv(os.path.join(res_dir, "regression.csv"), index=False)
        r = reg.set_index("term")
        store["regression"] = r.to_dict("index")
        def _pv(v):
            return "$<$0.001" if v < 0.001 else "%.3f" % v

        rrows = [f"{REG_LABELS[t]} & {100 * r.loc[t, 'coef']:.1f} & [{100 * r.loc[t, 'lo']:.1f}, "
                 f"{100 * r.loc[t, 'hi']:.1f}] & {_pv(r.loc[t, 'p'])} \\\\" for t in REG_LABELS]
        _write(tdir, "tab_regression.tex", _reg_tex("\n".join(rrows)))
        M["RegSWPW"], M["RegSWPWCI"], M["RegSWPWP"] = (f"{100 * r.loc['SW:PW', 'coef']:.1f}",
                                                        f"[{100 * r.loc['SW:PW', 'lo']:.1f}, {100 * r.loc['SW:PW', 'hi']:.1f}]".replace("-", "$-$"),
                                                        p_str(r.loc['SW:PW', 'p']))
        M["RegWAPW"], M["RegWAPWP"] = f"{100 * r.loc['WA:PW', 'coef']:.1f}", p_str(r.loc["WA:PW", "p"])
        for t in ("WA", "SW", "SR", "PW", "PR"):
            M[f"Reg{t}"] = f"{abs(100 * r.loc[t, 'coef']):.1f}"   # sign is stated in the prose
            M[f"Reg{t}CI"] = f"[{100 * r.loc[t, 'lo']:.1f}, {100 * r.loc[t, 'hi']:.1f}]".replace("-", "$-$")
            M[f"Reg{t}P"] = p_str(r.loc[t, "p"])
        M["RegWASW"], M["RegWASWP"] = f"{100 * r.loc['WA:SW', 'coef']:.1f}", p_str(r.loc["WA:SW", "p"])
    except Exception as e:  # noqa: BLE001
        print(f"[report] regression skipped: {e}")

    # ---------------------------------------------------------------- domain stability (RQ3)
    dom = {}
    for d in DOMAIN_ORDER:
        sub = base[base.domain == d]
        if sub.empty:
            continue
        dom[d] = {j: metrics(bt, sub[sub.judge == j], per_judge_self=True) for j in judges}
    doms = [d for d in DOMAIN_ORDER if d in dom]
    drows = []
    for j in judges:
        vals = []
        for k in ["acc", "vb", "spb", "asr_any"]:
            for d in doms:
                e = dom[d][j].get(k)
                vals.append(pct(e.point) if e is not None else "--")
        drows.append(f"{tex_escape(disp[j])} & " + " & ".join(vals) + " \\\\")
    kw_cells = []
    for k, name in [("acc", "KWAcc"), ("vb", "KWVB"), ("spb", "KWSPB"), ("asr_any", "KWASR")]:
        try:
            mat = np.array([[dom[d][j][k].point for j in judges] for d in doms])
            w, p = kendall_w_test(mat, seed=cfg["seed"])
            M[name], M[name + "P"] = f"{w:.2f}", p_str(p)
            if k == "acc":
                store["_kw_acc_p"] = float(p)
            kw_cells.append(f"\\multicolumn{{{len(doms)}}}{{c}}{{{w:.2f} ({p_str(p).replace('p', '$p$')})}}")
        except (KeyError, ValueError):
            kw_cells.append(f"\\multicolumn{{{len(doms)}}}{{c}}{{--}}")
    drows.append("\\midrule\nKendall's $W$ & " + " & ".join(kw_cells) + " \\\\")
    _write(tdir, "tab_domain.tex", _domain_tex("\n".join(drows)))
    store["domain"] = {d: {j: {k: e.as_dict() for k, e in m.items()} for j, m in dj.items()} for d, dj in dom.items()}

    # ---------------------------------------------------------------- competence split
    comp = competence_split(bt, base, solo)
    for grp, suffix in [("known", "Known"), ("unknown", "Unknown")]:
        for k, name in [("acc", "Acc"), ("vb", "VB"), ("spb", "SPB"), ("asr_any", "ASR"), ("flip", "Flip")]:
            if k in comp[grp]:
                M[f"{name}{suffix}"] = pct(comp[grp][k].point)
    store["competence"] = {g: {k: (e.as_dict() if isinstance(e, Est) else e) for k, e in m.items()} for g, m in comp.items()}

    # ---------------------------------------------------------------- best-of-N selection (agentic)
    sm = select_metrics(bt, sel)
    for proto, tag in [("base_raw", ""), ("debias_norm", "DN")]:
        r = sm.get((proto, "pooled"))
        if not r:
            continue
        for c, C in [("clean", "Clean"), ("pad", "Pad"), ("attack", "Attack")]:
            if f"acc_{c}" in r:
                M[f"SelAcc{C}{tag}"] = pct(r[f"acc_{c}"].point)
                M[f"SelCap{C}{tag}"] = pct(r[f"capture_{c}"].point)
        if "pos1_excess" in r:
            M[f"SelPosOne{tag}"] = pct(r["pos1_excess"].point)
    store["select"] = {f"{p}|{j}": {k: e.as_dict() for k, e in r.items()} for (p, j), r in sm.items()}

    # ---------------------------------------------------------------- scoring-rule sensitivity (A2)
    tr = tie_rule_acc(bt, base)
    if "acc_strict" in tr:
        M["AccStrictBase"], M["AccStrictBaseCI"] = pct(tr["acc_strict"].point), ci_str(tr["acc_strict"])
        M["AccDropBase"], M["AccDropBaseCI"] = pct(tr["acc_drop"].point), ci_str(tr["acc_drop"])
        M["DecidedShareBase"] = pct(tr["decided_share"].point)
    store["tie_rules"] = {k: e.as_dict() for k, e in tr.items()}

    # ---------------------------------------------------------------- verification gain (A3)
    vg = verification_gain(bt, base, solo, judges)
    store["verification_gain"] = {j: {k: e.as_dict() for k, e in d.items()} for j, d in vg.items()}
    if vg:
        M["SoloMin"] = pct(min(d["solo"].point for d in vg.values()))
        M["SoloMax"] = pct(max(d["solo"].point for d in vg.values()))
        M["VGMin"] = pct(min(d["vg"].point for d in vg.values()))
        M["VGMax"] = pct(max(d["vg"].point for d in vg.values()))
        bj = max(vg, key=lambda j: vg[j]["vg"].point)
        M["VGJudgeBest"], M["VGBest"], M["VGBestCI"] = tex_escape(disp[bj]), pct(vg[bj]["vg"].point), ci_str(vg[bj]["vg"])
    # pooled, and split by how many options the solo task offered: a two-option item gives the solo
    # model the same 50% chance floor as the verifier, so that split is the fair comparison.
    bias_clean = base[(base.block == "bias") & (base.length_cond == "none")]
    s_keep = solo[solo.item_id.isin(kept)]
    two_opt = {d for d, r in (store.get("data_table") or {}).items() if r.get("n_options") == 2}
    def _vg(vsub, ssub):
        a = bt.mean(vsub, "is_correct")
        b2 = bt.mean(ssub.assign(c=ssub.correct.astype(float)), "c")
        return (a - b2) if (a is not None and b2 is not None) else None
    for tag, vmask, smask in (("", slice(None), slice(None)),
                              ("TwoOpt", bias_clean.domain.isin(two_opt), s_keep.domain.isin(two_opt)),
                              ("FourOpt", ~bias_clean.domain.isin(two_opt), ~s_keep.domain.isin(two_opt))):
        e = _vg(bias_clean[vmask] if tag else bias_clean, s_keep[smask] if tag else s_keep)
        if e is not None:
            M[f"VGPooled{tag}"], M[f"VGPooled{tag}CI"] = pct(e.point), ci_str(e)
            store.setdefault("verification_gain_pooled", {})[tag or "all"] = e.as_dict()
    store["verification_gain_by_domain"] = {
        d: {j: (_vg(bias_clean[(bias_clean.domain == d) & (bias_clean.judge == j)],
                    s_keep[(s_keep.domain == d) & (s_keep.model == j)]) or Est(float("nan"), np.array([]))).as_dict()
            for j in judges} for d in DOMAIN_ORDER}

    sp = os.path.join(run, "solo_utility.jsonl")
    if os.path.exists(sp):
        su = pd.DataFrame(read_jsonl(sp))
        su_keep = su[su.item_id.isin(kept)]
        M["SoloPhi"] = pct(su_keep.correct.mean())
        store["solo_utility"] = {"acc_kept": float(su_keep.correct.mean()), "n": int(len(su_keep)),
                                 "acc_all": float(su.correct.mean()),
                                 "by_domain": su_keep.groupby("domain").correct.mean().to_dict()}

    # ---------------------------------------------------------------- author length and covariate (A4)
    fam_words = rdf[rdf.item_id.isin(kept)].groupby("gen").words.mean().to_dict()
    store["words_by_author"] = {disp.get(g, g): float(w) for g, w in fam_words.items()}
    if fam_words:
        M["WordsMin"], M["WordsMax"] = f"{min(fam_words.values()):.0f}", f"{max(fam_words.values()):.0f}"
        M["WordsSpread"] = f"{max(fam_words.values()) - min(fam_words.values()):.0f}"
    try:
        reg2 = interaction_regression(base, length_covariate=True).set_index("term")
        store["regression_length"] = reg2.to_dict("index")
        M["RegLR"], M["RegLRP"] = f"{100 * reg2.loc['LR', 'coef']:.1f}", p_str(reg2.loc["LR", "p"])
        M["RegSWadj"], M["RegPWadj"] = f"{100 * reg2.loc['SW', 'coef']:.1f}", f"{100 * reg2.loc['PW', 'coef']:.1f}"
        reg2.reset_index().to_csv(os.path.join(res_dir, "regression_length.csv"), index=False)
    except Exception as e:  # noqa: BLE001
        print(f"[report] length-covariate regression skipped: {e}")

    # ---------------------------------------------------------------- what the normaliser does (A5)
    norm_split = {}
    if "solo_utility" in store:
        su = pd.DataFrame(read_jsonl(sp))
        know = dict(zip(su.item_id, su.correct))
        for tag, want in (("phi_known", 1), ("phi_unknown", 0)):
            ids = {i for i, c in know.items() if int(c) == want}
            b = base[base.item_id.isin(ids)]
            nv = trials[(trials.protocol == "base_norm") & trials.item_id.isin(ids)]
            mb, mn = metrics(bt, b, per_judge_self=False), metrics(bt, nv, per_judge_self=False)
            if "acc" in mb and "acc" in mn:
                norm_split[tag] = {"base": mb["acc"], "norm": mn["acc"], "delta": mn["acc"] - mb["acc"],
                                   "n_items": len(ids & kept)}
        if "phi_known" in norm_split:
            M["DAccNormPhiKnown"] = pct(norm_split["phi_known"]["delta"].point)
            M["DAccNormPhiKnownCI"] = ci_str(norm_split["phi_known"]["delta"])
            M["NPhiKnownItems"] = str(norm_split["phi_known"]["n_items"])
        if "phi_unknown" in norm_split:
            M["DAccNormPhiUnknown"] = pct(norm_split["phi_unknown"]["delta"].point)
            M["DAccNormPhiUnknownCI"] = ci_str(norm_split["phi_unknown"]["delta"])
            M["NPhiUnknownItems"] = str(norm_split["phi_unknown"]["n_items"])
    store["normalizer_split"] = {t: {k: (e.as_dict() if isinstance(e, Est) else e) for k, e in d.items()}
                                 for t, d in norm_split.items()}
    # accuracy of the normalised protocol split by whether the rewrite passed validation
    nmethod = vdf[vdf.key.str.endswith("#norm")].assign(
        base_key=lambda d: d.key.str.replace("#norm", "", regex=False))[["base_key", "method"]]
    mm = dict(zip(nmethod.base_key, nmethod.method))
    nv = trials[trials.protocol == "base_norm"].copy()
    nv["wkey"] = [vkey_of(r) for r in nv.to_dict("records")]
    nv["wmethod"] = nv.wkey.map(mm)
    for tag, want in (("AccNormRewrite", "llm"), ("AccNormFallback", "fallback")):
        sub = nv[(nv.block == "bias") & (nv.length_cond == "none") & (nv.wmethod == want)]
        e = bt.mean(sub, "is_correct")
        if e is not None:
            M[tag] = pct(e.point)
            store.setdefault("normalizer_method_acc", {})[want] = {**e.as_dict(), "n_units": int(len(sub))}
    nm = vdf[vdf.key.str.endswith("#norm")].assign(kind=lambda d: d.key.str.split("|").str[-1].str.replace("#norm", "", regex=False))
    fb = nm.groupby("kind").method.apply(lambda x: (x == "fallback").mean())
    store["normalizer_fallback_by_kind"] = {k: float(v) for k, v in fb.items()}
    for key, name in (("pad", "NormFallbackPad"), ("raw", "NormFallbackRaw"), ("atk:combo", "NormFallbackCombo")):
        if key in fb:
            M[name] = pct(fb[key])

    # ---------------------------------------------------------------- equivalence + multiplicity (A8, A10)
    prim = {}
    if "ic_self_len" in b0:
        eq = equivalence(b0["ic_self_len"], bound=0.05)
        if eq:
            M["ICSE"], M["ICMDE"] = f"{100 * eq['se']:.1f}", f"{100 * eq['mde']:.1f}"
            M["ICBound"], M["ICTOSTP"] = f"{100 * eq['bound']:.0f}", p_str(eq["p_tost"])
            store["ic_equivalence"] = eq
        prim["IC"] = b0["ic_self_len"].p()
    if "Full" in conf_metrics and "acc" in conf_metrics["Full"] and "acc" in b0:
        prim["DAccFull"] = (conf_metrics["Full"]["acc"] - b0["acc"]).p()
    if "asr_any" in conf_metrics.get("DebNorm", {}) and "asr_any" in b0:
        prim["ASRany"] = (conf_metrics["DebNorm"]["asr_any"] - b0["asr_any"]).p()
    if "KWAccP" in M:
        prim["KWAcc"] = store.get("_kw_acc_p")
    adj = holm({k: v for k, v in prim.items() if v is not None})
    store["primary_tests"] = {"raw": prim, "holm": adj}
    for k, name in (("IC", "HolmIC"), ("DAccFull", "HolmDAccFull"), ("KWAcc", "HolmKWAcc"), ("ASRany", "HolmASRany")):
        if k in adj:
            M[name] = p_str(adj[k])

    # ---------------------------------------------------------------- controls B1-B4 and RR/net/sym
    for a, name in (("placebo", "ASRplaceboBase"), ("consensus", "ASRconsBase")):
        if f"asr_{a}" in b0:
            M[name], M[name + "CI"] = pct(b0[f"asr_{a}"].point), ci_str(b0[f"asr_{a}"])
    dn = conf_metrics.get("DebNorm", {})
    if "asr_consensus" in dn:
        M["ASRconsDebNorm"], M["ASRconsDebNormCI"] = pct(dn["asr_consensus"].point), ci_str(dn["asr_consensus"])
    for a, tag in [(a, a.capitalize()) for a in SINGLE_ATTACKS] + [("any", "Any"), ("combo", "Combo")]:
        for pre, mac in (("rr", "RR"), ("net", "Net"), ("adj", "Adj"), ("sym", "Sym"),
                         ("dacc", "DAcc"), ("symdacc", "SymDAcc")):
            if f"{pre}_{a}" in b0:
                M[f"{mac}{tag}Base"] = pct(b0[f"{pre}_{a}"].point)
                M[f"{mac}{tag}BaseCI"] = ci_str(b0[f"{pre}_{a}"])
    above = [a for a in SINGLE_ATTACKS if f"adj_{a}" in b0 and b0[f"adj_{a}"].ci()[0] > 0]
    M["NEditsAbovePlacebo"] = str(len(above))
    store["edits_above_placebo"] = above
    nctrl = int(trials[trials.attack.isin(CONTROL_EDITS)].shape[0] + trials[trials.block == "attack_sym"].shape[0]
                + trials[trials.protocol.str.endswith("strip")].shape[0])
    M["NCtrlVerdicts"] = f"{nctrl:,}".replace(",", "{,}")

    # ---------------------------------------------------------------- best-of-N scoring rule
    for proto, tag in [("base_raw", ""), ("debias_norm", "DN")]:
        r = sm.get((proto, "pooled"))
        if not r:
            continue
        if "parse_rate" in r:
            M[f"SelParseRate{tag}"] = pct(r["parse_rate"].point)
            M[f"SelChanceScored{tag}"] = pct(r["chance_scored"].point)
        if not tag:
            for c, C in (("clean", "Clean"), ("attack", "Attack")):
                if f"acc_{c}" in r:
                    M[f"SelAcc{C}CI"] = ci_str(r[f"acc_{c}"])
                if f"capture_{c}" in r:
                    M[f"SelCap{C}CI"] = ci_str(r[f"capture_{c}"])
            if "acc_attack_parsed" in r:
                M["SelAccAttackParsed"] = pct(r["acc_attack_parsed"].point)
            picks = {i: r[f"pick_pos{i}"].point for i in range(1, 5) if f"pick_pos{i}" in r}
            if picks:
                mp = max(picks, key=picks.get)
                M["SelPickPosMax"], M["SelPickPosMaxShare"] = str(mp), pct(picks[mp])

    # ---------------------------------------------------------------- why verdicts fail to parse
    pa = os.path.join(run, "parse_audit.json")
    if os.path.exists(pa):
        with open(pa) as f:
            au = json.load(f)
        store["parse_audit"] = au
        M["ParseAuditN"] = str(au["n_calls"])
        M["ParseAuditFails"] = str(au["n_parse_fail_sampled"])
        M["ParseAuditStop"] = str(au["finish_reasons"].get("stop", 0))
        M["ParseAuditMaxTok"] = str(au["max_completion_tokens"])
        M["ParseAuditCap"] = str(au["max_tokens_allowed"])
        M["ParseAuditReparse"] = pct(au["reparse_rate_overall"])

    # ---------------------------------------------------------------- repeat-decode reliability
    rt = os.path.join(run, "retest.jsonl")
    if os.path.exists(rt):
        rd = pd.DataFrame(read_jsonl(rt))
        M["RetestN"] = f"{len(rd):,}".replace(",", "{,}")
        M["RetestVerdict"] = pct(rd.verdict_same.mean())
        M["RetestFlipPct"] = pct(1 - rd.verdict_same.mean())
        M["RetestText"] = pct(rd.text_same.mean())
        e = bt.mean(rd.assign(c=rd.is_correct_retest.astype(float)), "c")
        if e is not None:
            M["RetestAcc"] = pct(e.point)
        bj = rd.groupby("judge").verdict_same.mean().sort_values()
        M["RetestVerdictMin"], M["RetestVerdictMax"] = pct(bj.iloc[0]), pct(bj.iloc[-1])
        M["RetestJudgeMin"] = tex_escape(disp.get(bj.index[0], bj.index[0]))
        M["RetestJudgeMax"] = tex_escape(disp.get(bj.index[-1], bj.index[-1]))
        store["retest"] = {"n": int(len(rd)), "verdict_agreement": float(rd.verdict_same.mean()),
                           "text_agreement": float(rd.text_same.mean()),
                           "acc_retest": (e.point if e is not None else None),
                           "by_judge": rd.groupby("judge").verdict_same.mean().to_dict()}

    # ---------------------------------------------------------------- figures
    _fig_delta(delta_tab, os.path.join(fdir, "fig_delta.pdf"))
    _fig_select(sm, os.path.join(fdir, "fig_select.pdf"), n_cand=int(sel.n_candidates.iloc[0]) if len(sel) else 4)
    # the manuscript includes these two under their descriptive names
    for src, dst in (("fig_delta", "fig_corrections"), ("fig_select", "fig_bestofn")):
        for ext in (".pdf", ".png"):
            try:
                shutil.copyfile(os.path.join(fdir, src + ext), os.path.join(fdir, dst + ext))
            except OSError:
                pass

    write_macros(M, os.path.join(paper, "results_macros.tex"))
    with open(os.path.join(res_dir, "metrics.json"), "w") as f:
        json.dump({"macros": M, **store}, f, indent=2, default=float)
    pd.DataFrame([{"judge": j, "metric": k, **e.as_dict()} for j, m in per_judge.items() for k, e in m.items()]) \
        .to_csv(os.path.join(res_dir, "per_judge.csv"), index=False)
    pd.DataFrame([{"config": s, "metric": k, **e.as_dict()} for s, m in conf_metrics.items() for k, e in m.items()]) \
        .to_csv(os.path.join(res_dir, "corrections.csv"), index=False)
    missing = [n for n in macro_names() if n not in M]
    print(f"[report] wrote {len(M)} macros ({len(missing)} unfilled), tables -> {tdir}, figures -> {fdir}")
    return M, store


# =============================================================================== figures
def _fig_delta(delta_tab, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["acc", "flip", "vb", "spb_abs", "asr_combo", "asr_any"]
    labels = {"acc": "Acc $\\uparrow$", "flip": "Flip $\\downarrow$", "vb": "|VB| $\\downarrow$",
              "spb_abs": "|SPB| $\\downarrow$", "asr_combo": "ASR$_{combo}$ $\\downarrow$", "asr_any": "ASR$_{any}$ $\\downarrow$"}
    rows = [s for s, *_ in CONFIGS if s in delta_tab and s != "Base"]
    rlab = {s: l.replace("+ ", "+") for s, l, *_ in CONFIGS}
    Z = np.full((len(rows), len(cols)), np.nan)
    S = np.zeros_like(Z, dtype=bool)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            e = delta_tab[r].get(c)
            if e is not None:
                Z[i, j] = 100 * e.point
                lo, hi = e.ci()
                S[i, j] = lo > 0 or hi < 0
    fig, ax = plt.subplots(figsize=(3.45, 0.32 * len(rows) + 1.0))
    lim = np.nanmax(np.abs(Z)) if np.isfinite(Z).any() else 1
    im = ax.imshow(Z, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            t = "n/a" if np.isnan(Z[i, j]) else f"{Z[i, j]:+.1f}{'*' if S[i, j] else ''}"
            ax.text(j, i, t, ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(cols)), [labels[c] for c in cols], fontsize=6, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), [rlab[r] for r in rows], fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=5)
    cb.set_label("improvement over baseline (pp)", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)  # for the Word version
    plt.close(fig)


def _fig_select(sm, path, n_cand):
    """Best-of-N: accuracy and capture per pool, with both chance levels drawn.

    A pick that cannot be parsed counts as a miss, so the rate a random selector reaches is the parse
    rate over N, not 1/N. Both lines are shown, because the attacked pool sits between them.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    conds = ["clean", "pad", "attack"]
    protos = [p for p in ("base_raw", "debias_norm") if (p, "pooled") in sm]
    w = 0.38
    names = {"base_raw": "Single judge", "debias_norm": "Debias + Norm."}
    hatches = {"base_raw": "", "debias_norm": "///"}
    for k, p in enumerate(protos):
        r = sm[(p, "pooled")]
        acc = [r.get(f"acc_{c}") for c in conds]
        cap = [r.get(f"capture_{c}") for c in conds]
        x = np.arange(len(conds)) + (k - (len(protos) - 1) / 2) * w
        ys = [100 * e.point if e else np.nan for e in acc]
        err = np.array([[100 * (e.point - e.ci()[0]), 100 * (e.ci()[1] - e.point)] if e else [0, 0]
                        for e in acc]).T
        ax.bar(x, ys, w, yerr=err, capsize=2, color="#4a76b8", alpha=0.9 - 0.25 * k,
               hatch=hatches[p], edgecolor="white", label=f"{names[p]}: correct")
        cerr = np.array([[100 * (e.point - e.ci()[0]), 100 * (e.ci()[1] - e.point)] if e else [0, 0]
                         for e in cap]).T
        ax.bar(x, [100 * e.point if e else np.nan for e in cap], w * 0.5, yerr=cerr, capsize=1.5,
               color="#c0392b", alpha=0.85 - 0.25 * k, hatch=hatches[p], edgecolor="white",
               label=f"{names[p]}: picked perturbed")
        ch = r.get("chance_scored")
        if ch is not None:
            ax.hlines(100 * ch.point, x[0] - w / 2, x[-1] + w / 2, ls=":", lw=0.9, color="#777777")
    ax.axhline(100 / n_cand, ls="--", lw=0.8, color="gray")
    ax.set_xticks(range(len(conds)), ["clean pool", "one wrong\npadded", "one wrong\nattacked"], fontsize=7)
    ax.set_ylabel("% of selections", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(fontsize=5.2, loc="upper center", ncol=2, frameon=False)
    ax.set_xlim(-0.5, len(conds) - 0.5)
    ax.set_ylim(0, 62)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--config")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--placeholders", action="store_true")
    ap.add_argument("--judges", default="Llama-3.1-8B,Qwen2.5-7B,Gemma-2-9B,Mistral-7B")
    a = ap.parse_args()
    if a.placeholders:
        write_macros({}, os.path.join(a.paper, "results_macros.tex"), placeholder=True)
        placeholder_tables(os.path.join(a.paper, "tables"), a.judges.split(","))
        print("[report] placeholder macros and tables written")
        return
    cfg = load_config(a.config)
    build_report(a.run, cfg, a.paper)


if __name__ == "__main__":
    main()
