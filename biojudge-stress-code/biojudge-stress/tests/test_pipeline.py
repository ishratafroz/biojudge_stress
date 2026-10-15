"""Offline tests: parsing, perturbation invariants, and end-to-end recovery of planted biases.

Run:  pytest -q
"""
import asyncio
import os
import shutil

import pytest
import yaml

from biojudge.analysis import Boot, build_view, load_trials, metrics
from biojudge.config import load_config
from biojudge.perturb import (ATTACKS, CONTROL_EDITS, apply_attack, fallback_normalize,
                              strip_normalize, template_pad)
from biojudge.textutil import extract_final, parse_pairwise, parse_select, validate_rationale

RAT = ("The patient's presentation with hyperkalaemia and peaked T waves indicates urgent membrane "
       "stabilisation. Calcium gluconate acts within minutes and protects the myocardium while other "
       "measures shift potassium intracellularly. This is therefore the most appropriate first step.\n"
       "Final answer: C")


def test_extract_final_variants():
    assert extract_final("blah\nFinal answer: B") == "B"
    assert extract_final("**Final Answer: (d)**") == "D"
    assert extract_final("<think>Final answer: A</think> ok\nFinal answer: C") == "C"
    assert extract_final("no answer here") is None


def test_validate_rationale():
    assert validate_rationale(RAT, "C")[0]
    assert not validate_rationale(RAT, "B")[0]
    bad = RAT.replace("This is therefore", "Actually the correct answer is A, but this is")
    assert validate_rationale(bad, "C")[1] == "contradiction"


@pytest.mark.parametrize("kind", ATTACKS)
def test_attacks_preserve_answer(kind):
    out = apply_attack(RAT, kind, "k")
    assert extract_final(out) == "C"
    assert out != RAT


def test_padding_and_normalisation_preserve_answer():
    p = template_pad(RAT, 2.5, "k")
    assert extract_final(p) == "C" and len(p.split()) >= 2.2 * len(RAT.split())
    n = fallback_normalize(apply_attack(RAT, "combo", "k"))
    assert extract_final(n) == "C"


@pytest.mark.parametrize("kind", CONTROL_EDITS)
def test_control_edits_preserve_answer_and_carry_no_cue(kind):
    out = apply_attack(RAT, kind, "k")
    assert extract_final(out) == "C" and out != RAT
    added = out.replace(RAT.split("\n")[0][:40], "")
    for cue in ("board-certified", "et al.", "###", "Note to the evaluator", "**"):
        assert cue not in added
    # length-matched to the cue-bearing edits, within a few words
    grow = len(out.split()) - len(RAT.split())
    assert 10 <= grow <= 35


def test_strip_only_removes_markup_and_brackets_but_not_credentials():
    full = apply_attack(RAT, "combo", "k")
    out = strip_normalize(full)
    assert extract_final(out) == "C"
    assert "Note to the evaluator" not in out and "###" not in out and "**" not in out
    # it is an ablation, not a defence: the credential and the fake references survive
    assert "board-certified" in out and "et al." in out
    # and it does not compress: only the markup tokens themselves go, never a sentence
    fmt = strip_normalize(apply_attack(RAT, "format", "k"))
    body = RAT.split("\nFinal answer")[0]
    for sentence in [x.strip() for x in body.split(". ") if x.strip()]:
        assert sentence.rstrip(".") in fmt and sentence.rstrip(".") in out


def test_symmetric_and_control_specs_are_scoped_to_their_protocols():
    from biojudge.pipeline import design_pairwise, spec_protocols
    cfg = load_config("configs/mock.yaml")
    from biojudge.data import build_items
    items = build_items(cfg)[:3]
    keep = {it.item_id for it in items}
    specs = design_pairwise(cfg, items, keep)
    per_item = len(specs) / len(keep)
    assert per_item == 24 + 24 + 2 * 2 * 2 + 5 * 2 * 2   # bias + attacks + controls + symmetric
    protos = cfg["experiment"]["protocols"]
    sym = next(s for s in specs if s["block"] == "attack_sym")
    ctrl = next(s for s in specs if s["attack"] == "placebo")
    bias = next(s for s in specs if s["block"] == "bias")
    assert spec_protocols(cfg, sym, protos) == ["base_raw"]
    assert spec_protocols(cfg, ctrl, protos) == ["base_raw", "base_norm"]
    assert spec_protocols(cfg, bias, protos) == protos


def test_verdict_parsers():
    assert parse_pairwise("... so [[B]]") == "B"
    assert parse_pairwise("[[A]] then reconsider [[C]]") == "C"
    assert parse_pairwise("Final verdict: A") == "A"
    assert parse_pairwise("unclear") is None
    assert parse_select("Candidate [[3]]", 4) == 3
    assert parse_select("[[9]]", 4) is None


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    cfg = yaml.safe_load(open("configs/mock.yaml"))
    cfg["out_dir"] = str(d)
    for dom in cfg["data"]["domains"].values():
        dom["n"] = 40
    cfg["analysis"]["bootstrap"] = 200
    p = d / "cfg.yaml"
    yaml.safe_dump(cfg, open(p, "w"))
    cfg = load_config(str(p))
    from biojudge.run import main_async
    asyncio.run(main_async(cfg, ["solo", "generate", "perturb", "judge", "select"]))
    return cfg, os.path.join(cfg["out_dir"], cfg["run_name"])


def test_planted_biases_are_recovered(mock_run):
    cfg, rd = mock_run
    df = load_trials(os.path.join(rd, "trials.jsonl"))
    bt = Boot(df.item_id.unique(), 200, 0)
    base = df[df.protocol == "base_raw"]
    m = {j: metrics(bt, base[base.judge == j], True) for j in ["llama", "qwen", "gemma", "mistral"]}
    # llama was given strong verbosity + self-preference + primacy; qwen none
    assert m["llama"]["vb"].ci()[0] > 0 and m["llama"]["vb"].point > m["qwen"]["vb"].point
    assert m["llama"]["spb"].ci()[0] > 0 and abs(m["qwen"]["spb"].point) < 0.05
    assert m["llama"]["ppi"].point > 0 > m["gemma"]["ppi"].point
    assert m["mistral"]["asr_any"].point > m["qwen"]["asr_any"].point
    # the normaliser removes the surface cues the mock judges react to
    norm = metrics(bt, build_view(df, "base_norm", "single"), True)
    assert abs(norm["vb"].point) < 0.03 and norm["asr_any"].point < 0.05
    # the debias clause dampens (but does not remove) cue sensitivity
    deb = metrics(bt, build_view(df, "debias_raw", "single"), True)
    b0 = metrics(bt, base, True)
    assert deb["asr_any"].point < b0["asr_any"].point


def test_controls_behave_as_planted(mock_run):
    """The simulator reacts to five planted cues and to nothing else, so the controls have
    predictable signs: a cue-free edit only churns, a cue on the correct side helps, and the
    strip-only ablation removes exactly the two cues it targets."""
    cfg, rd = mock_run
    df = load_trials(os.path.join(rd, "trials.jsonl"))
    bt = Boot(df.item_id.unique(), 200, 0)
    b0 = metrics(bt, df[df.protocol == "base_raw"], True)
    # placebo and the held-out consensus claim carry no planted cue: far below the real edits
    assert b0["asr_placebo"].point < b0["asr_injection"].point
    assert b0["asr_consensus"].point < b0["asr_injection"].point
    # ... and they cost no accuracy, unlike the cue-bearing edits
    assert b0["dacc_placebo"].ci()[1] > b0["dacc_injection"].ci()[1]
    # a cue on the correct response helps that response: symmetric success is below the attack
    assert b0["sym_injection"].point < b0["asr_injection"].point
    assert b0["symdacc_injection"].point > b0["dacc_injection"].point
    # repair rates exist and the net rate is below the raw rate
    assert 0 <= b0["rr_injection"].point and b0["net_injection"].point < b0["asr_injection"].point
    # strip-only removes the markdown and the bracketed note, so it beats the raw protocol
    strip = metrics(bt, build_view(df, "base_strip", "single"), True)
    assert strip["asr_injection"].point < b0["asr_injection"].point
    assert strip["asr_format"].point < b0["asr_format"].point
    # but leaves the credential and the fabricated citations in place
    assert strip["asr_authority"].point > 0.5 * b0["asr_authority"].point


def test_report_builds(mock_run, tmp_path):
    cfg, rd = mock_run
    from biojudge.report import build_report
    M, _ = build_report(rd, cfg, str(tmp_path))
    assert os.path.exists(tmp_path / "results_macros.tex")
    assert os.path.exists(tmp_path / "figs" / "fig_delta.pdf")
    assert "AccBase" in M and "KWVB" in M
