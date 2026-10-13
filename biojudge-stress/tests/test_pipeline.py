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
from biojudge.perturb import ATTACKS, apply_attack, fallback_normalize, template_pad
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


def test_report_builds(mock_run, tmp_path):
    cfg, rd = mock_run
    from biojudge.report import build_report
    M, _ = build_report(rd, cfg, str(tmp_path))
    assert os.path.exists(tmp_path / "results_macros.tex")
    assert os.path.exists(tmp_path / "figs" / "fig_delta.pdf")
    assert "AccBase" in M and "KWVB" in M
