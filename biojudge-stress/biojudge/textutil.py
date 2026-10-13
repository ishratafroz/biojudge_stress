"""Parsing and validation helpers shared by every stage."""
from __future__ import annotations

import re

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FINAL = re.compile(r"final\s+answer\s*[:：]?\s*\**\s*(?:option\s*)?\(?\s*([A-E])\b", re.I)
_CLAIM = re.compile(r"\b(?:correct|right|best)\s+(?:answer|option|choice)\s+(?:is|would be|should be)\s*"
                    r"(?:option\s*)?\(?([A-E])\b", re.I)
_REFUSAL = re.compile(r"\b(?:I\s+(?:cannot|can't|won't|will not)|as an ai\b|I must (?:point out|note) that)", re.I)
_PAIR = re.compile(r"\[\[\s*([ABC])\s*\]\]")
_PAIR_FALLBACK = re.compile(r"(?:final\s+)?verdict\s*[:：]?\s*(?:assistant\s*)?\[?\s*([AB])\b", re.I)
_SELECT = re.compile(r"\[\[\s*(\d+)\s*\]\]")


def strip_think(text: str | None) -> str:
    return _THINK.sub("", text or "").strip()


def words(text: str | None) -> int:
    return len((text or "").split())


def extract_final(text: str | None) -> str | None:
    m = _FINAL.findall(strip_think(text))
    return m[-1].upper() if m else None


def split_final(text: str) -> tuple[str, str]:
    """Split a response into (body, final_line). final_line is '' if absent."""
    t = strip_think(text).rstrip()
    lines = t.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if _FINAL.search(lines[i]):
            return "\n".join(lines[:i]).rstrip(), lines[i].strip()
    return t, ""


def validate_rationale(text: str | None, target: str, min_words: int = 30) -> tuple[bool, str]:
    """A forced-answer rationale is valid if it ends on the target and never endorses another option."""
    if not text:
        return False, "empty"
    t = strip_think(text)
    fin = extract_final(t)
    if fin != target:
        return False, f"final={fin}"
    if any(c.upper() != target for c in _CLAIM.findall(t)):
        return False, "contradiction"
    if _REFUSAL.search(t):
        return False, "refusal"
    if words(t) < min_words:
        return False, "too_short"
    return True, "ok"


def parse_pairwise(text: str | None) -> str | None:
    t = strip_think(text)
    m = _PAIR.findall(t)
    if m:
        return m[-1]
    m = _PAIR_FALLBACK.findall(t)
    return m[-1].upper() if m else None


def parse_select(text: str | None, n: int) -> int | None:
    m = _SELECT.findall(strip_think(text))
    for v in reversed(m):
        if 1 <= int(v) <= n:
            return int(v)
    return None
