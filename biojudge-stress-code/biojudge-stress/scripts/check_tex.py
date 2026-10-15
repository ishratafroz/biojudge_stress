"""Structural checks on the filled manuscript, for hosts with no LaTeX installed.

Catches what a failed compile would catch late: a table row with the wrong number of cells, an
unbalanced brace, a reference with no label, a figure file that is not next to the .tex, and any
placeholder left in the text.

  python scripts/check_tex.py ../../biojudge_stress.tex --figdir paper/figs
"""
from __future__ import annotations

import argparse
import os
import re
import sys

COLSPEC = re.compile(r"\\begin\{tabular\}(?:\[[^\]]*\])?\{(.*)\}")
MULTICOL = re.compile(r"\\multicolumn\{(\d+)\}")


def n_columns(spec: str) -> int:
    """Count column slots in a tabular preamble, ignoring @{...}, >{...} and p{...} arguments."""
    spec = re.sub(r"[@>!<]\{(?:[^{}]|\{[^{}]*\})*\}", "", spec)
    spec = re.sub(r"\*\{(\d+)\}\{([^{}]*)\}", lambda m: m.group(2) * int(m.group(1)), spec)
    spec = re.sub(r"[pmb]\{(?:[^{}]|\{[^{}]*\})*\}", "c", spec)
    return sum(1 for ch in spec if ch in "lcrpmbX")


def split_cells(row: str) -> int:
    cells = re.split(r"(?<!\\)&", row)
    n = 0
    for c in cells:
        m = MULTICOL.search(c)
        n += int(m.group(1)) if m else 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("--figdir", default=None, help="directory holding the figure files")
    a = ap.parse_args()
    src = open(a.tex).read()
    problems = []

    # 1. leftovers
    for pat, what in ((r"@@[A-Za-z0-9_:.+\-]+@@", "unfilled slot"),
                      (r"\\pend\{", "pending-result macro"), (r"\\pc\b", "pending table cell"),
                      (r"\\achk\{", "author-check note"), (r"\[\[P\]\]", "pending marker")):
        for mo in re.finditer(pat, src):
            line = src[:mo.start()].count("\n") + 1
            problems.append(f"line {line}: {what} {mo.group(0)[:40]}")

    # 2. tables: every row must fill the column spec. The preamble can nest braces several deep
    # (@{\hspace{1.5pt}}), so it is scanned by brace matching rather than by a regex.
    for mo in re.finditer(r"\\begin\{tabular\}(\[[^\]]*\])?\{", src):
        i, depth = mo.end(), 1
        while i < len(src) and depth:
            if src[i] == "{" and src[i - 1] != "\\":
                depth += 1
            elif src[i] == "}" and src[i - 1] != "\\":
                depth -= 1
            i += 1
        spec, rest = src[mo.end():i - 1], src[i:]
        end = rest.find("\\end{tabular}")
        if end < 0:
            problems.append(f"line {src[:mo.start()].count(chr(10)) + 1}: tabular with no \\end")
            continue
        ncol = n_columns(spec)
        body = rest[:end]
        start_line = src[:i].count("\n") + 1
        for raw in body.split("\\\\"):
            row = re.sub(r"\\(toprule|midrule|bottomrule|cmidrule(\(.*?\))?(\{.*?\})?|hline)", "", raw)
            row = re.sub(r"(?<!\\)%.*", "", row).strip()   # a comment, not an escaped percent
            if not row or row.startswith("\\end"):
                continue
            got = split_cells(row)
            if got != ncol:
                ln = start_line + body[:body.index(raw)].count("\n")
                problems.append(f"line ~{ln}: table row has {got} cells, spec has {ncol}: "
                                f"{row[:60].strip()}...")

    # 3. refs and labels
    labels = set(re.findall(r"\\label\{([^}]*)\}", src))
    refs = set(re.findall(r"\\ref\{([^}]*)\}", src))
    for r in sorted(refs - labels):
        problems.append(f"\\ref{{{r}}} has no \\label")
    for l in sorted(labels - refs):
        problems.append(f"\\label{{{l}}} is never referenced")

    # 4. citations
    cited = {c.strip() for grp in re.findall(r"\\cite\{([^}]*)\}", src) for c in grp.split(",")}
    defined = set(re.findall(r"\\bibitem\{([^}]*)\}", src))
    for c in sorted(cited - defined):
        problems.append(f"\\cite{{{c}}} has no \\bibitem")
    for d in sorted(defined - cited):
        problems.append(f"\\bibitem{{{d}}} is never cited")

    # 5. braces and environments
    depth = 0
    for i, ch in enumerate(src):
        if ch == "{" and (i == 0 or src[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or src[i - 1] != "\\"):
            depth -= 1
            if depth < 0:
                problems.append(f"unbalanced '}}' at line {src[:i].count(chr(10)) + 1}")
                depth = 0
    if depth:
        problems.append(f"{depth} unclosed '{{'")
    for env in set(re.findall(r"\\begin\{([a-zA-Z*]+)\}", src)):
        nb, ne = len(re.findall(r"\\begin\{%s\}" % re.escape(env), src)), len(re.findall(r"\\end\{%s\}" % re.escape(env), src))
        if nb != ne:
            problems.append(f"environment {env}: {nb} begin vs {ne} end")

    # 6. figures present
    if a.figdir:
        for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", src):
            cands = [os.path.join(a.figdir, f), os.path.join(a.figdir, f + ".pdf"),
                     os.path.join(a.figdir, f + ".png")]
            if not any(os.path.exists(c) for c in cands):
                problems.append(f"figure not found in {a.figdir}: {f}")

    words = len(re.sub(r"\\[a-zA-Z]+|[{}&$\\]", " ", src).split())
    print(f"[check_tex] {a.tex}: {src.count(chr(10)) + 1} lines, ~{words} words, "
          f"{len(labels)} labels, {len(defined)} bibitems")
    if problems:
        print(f"[check_tex] {len(problems)} problem(s):")
        for p in problems:
            print("  " + p)
    else:
        print("[check_tex] no structural problems found")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
