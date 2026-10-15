"""Structural check of the built Word document.

No Word or LibreOffice is available on the Linux host, so the .docx cannot be rendered to PDF for a
visual check. This inspects the file the way a reader would instead: headings in order, table shapes,
embedded images, and any leftover fill-in token or placeholder.

  python scripts/check_docx.py ../../BioJudge-Stress_paper_final.docx
"""
from __future__ import annotations

import re
import sys
import zipfile

from docx import Document
from docx.shared import Pt

EXPECTED_TABLE_COLS = {"tab_data": 6, "tab_bias_profile": 10, "tab_corrections": 8,
                       "tab_regression": 4, "tab_attacks": 7, "tab_domain": 13}
LEFTOVER = re.compile(r"@@[A-Z_]+@@|\[(?:[A-Z][A-Za-z]+)\]|\\[a-zA-Z]+\{|\[—\]|«|»|\bPH\b|\?\?")
ALLOWED_BRACKET = re.compile(r"\[\d+\](?:[–,-]\s*\[\d+\])?"       # citation markers like [12], [8]-[10]
                             r"|\[(?:Question|Answer)\]")             # literal prompt text in the Appendix
                                                                       # (same exemption as make_docx.NON_MACROS)


def main(path: str) -> int:
    doc = Document(path)
    problems: list[str] = []

    heads = [(p.style.name, p.text.strip()) for p in doc.paragraphs
             if p.style.name.startswith("Heading") and p.text.strip()]
    paras = [p for p in doc.paragraphs if p.text.strip()]
    words = sum(len(p.text.split()) for p in paras)

    print(f"file        {path}")
    print(f"paragraphs  {len(paras)} non-empty, ~{words:,} words")
    print(f"tables      {len(doc.tables)}  (cols: {[len(t.columns) for t in doc.tables]})")
    with zipfile.ZipFile(path) as z:
        media = [n for n in z.namelist() if n.startswith("word/media/")]
    print(f"images      {len(media)}  {[n.split('/')[-1] for n in media]}")
    print("\nheadings:")
    for st, t in heads:
        print(f"  {'  ' if st == 'Heading 2' else ''}{t}")

    # leftover tokens anywhere, including inside tables
    def scan(text, where):
        t = ALLOWED_BRACKET.sub("", text)
        for m in LEFTOVER.finditer(t):
            problems.append(f"leftover {m.group(0)!r} in {where}: {text[:90]}")

    for i, p in enumerate(paras):
        scan(p.text, f"paragraph {i}")
    for ti, t in enumerate(doc.tables):
        for ri, row in enumerate(t.rows):
            for ci, c in enumerate(row.cells):
                scan(c.text, f"table {ti} r{ri}c{ci}")

    # every table should have >= 2 rows and no empty body row
    for ti, t in enumerate(doc.tables):
        if len(t.rows) < 2:
            problems.append(f"table {ti} has {len(t.rows)} row(s)")
        for ri, row in enumerate(t.rows[1:], 1):
            if not any(c.text.strip() for c in row.cells):
                problems.append(f"table {ti} row {ri} is empty")

    # fonts: everything should be Times New Roman
    bad_fonts = {r.font.name for p in doc.paragraphs for r in p.runs
                 if r.font.name and r.font.name != "Times New Roman"}
    if bad_fonts:
        problems.append(f"unexpected fonts: {bad_fonts}")

    print()
    if problems:
        print(f"PROBLEMS ({len(problems)}):")
        for p in problems[:40]:
            print("  -", p)
        return 1
    print("OK: no leftover tokens, no empty table rows, fonts consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
