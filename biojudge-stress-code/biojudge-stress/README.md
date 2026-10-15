# BioJudge-Stress

Stress-testing LLM judges as verifier agents in biomedical multi-agent pipelines: position, verbosity and
self-preference bias, their interactions, content-free attacks, corrections, and best-of-N selection, all scored
against expert answer keys (PubMedQA, MedQA-USMLE, MedMCQA).

## Install

```bash
pip install -r requirements.txt          # or: pip install numpy pandas scipy statsmodels pyyaml matplotlib openai anthropic
```

## Run

```bash
export OPENROUTER_API_KEY=...            # any OpenAI-compatible gateway; see configs/openrouter.yaml

python -m biojudge.run --config configs/openrouter.yaml --estimate   # token/call budget, no API calls
python -m biojudge.run --config configs/pilot.yaml                   # 10 items/domain smoke test — do this first
python -m biojudge.run --config configs/openrouter.yaml              # main run (resumable; everything is cached)

python -m biojudge.report --run runs/main --config configs/openrouter.yaml --paper paper/
python scripts/export_prompts.py --out paper/prompts_appendix.tex
cd paper && latexmk -pdf main.tex
```

`configs/vllm_local.yaml` runs the same design against five local vLLM servers.
`configs/mock.yaml` runs the whole pipeline offline against a simulator with planted biases (no API keys, ~30 s) —
used to validate the estimators; its numbers are never results.

Stages can be run separately: `--stages items,solo,generate,perturb,judge,select`.

## Controls and ablations

The attack conditions mean nothing on their own, because a near-chance judge changes its mind under any
edit. Four conditions separate persuasion from churn; each is switched on in the config and judged only
under the protocols its question needs, so none of them costs a full crossing:

```yaml
experiment:
  protocols:         [base_raw, debias_raw, base_norm, debias_norm, base_strip]
  attacks_control:   [placebo, consensus]      # on R-: no-cue floor, and a cue the normaliser never saw
  control_protocols: [base_raw, debias_raw, base_norm, debias_norm]
  attacks_sym:       [authority, citation, format, confidence, injection]   # the same edits on R+
  sym_protocols:     [base_raw]
```

`base_strip` shows the judge a deterministic strip of markup and bracketed text instead of the LLM
rewrite, which isolates what the rewrite itself contributes. `biojudge.analysis` reports, for every
edit, the attack success rate, the repair rate (clean-wrong verdicts the edit makes right), the net and
placebo-adjusted rates, the accuracy the edit actually costs, and the symmetric rate.

## Reproducibility checks

```bash
python scripts/redecode.py --config configs/ollama_l40s_ctrl.yaml --run runs/local    # temp-0 is not bit-reproducible
python scripts/audit_parse_failures.py --config configs/ollama_l40s_ctrl.yaml --run runs/local
python scripts/solo_utility.py --config configs/ollama_l40s_ctrl.yaml --run runs/local
```

`redecode.py` asks every clean pair a second time and reports how often the text and the verdict repeat;
`audit_parse_failures.py` re-issues unparseable verdicts and records the server's `finish_reason`;
`solo_utility.py` measures the utility model's own competence without touching the distractor choice.

## Building the manuscript

The manuscript is a template whose every number is a named slot, filled from the run:

```bash
python scripts/fill_paper.py --metrics runs/local/results/metrics.json \
    --template paper/biojudge_stress_template.tex --out ../../biojudge_stress.tex \
    --report runs/local/results/paper_values.tsv
python scripts/check_paper_numbers.py --template paper/biojudge_stress_template.tex \
    --config configs/ollama_l40s_ctrl.yaml --metrics runs/local/results/metrics.json
```

`fill_paper.py` refuses to write if a slot does not resolve, and `check_paper_numbers.py` flags any
number typed into the prose that no config value or computed result accounts for.

## Layout

```
biojudge/config.py      YAML config + validation (families must differ; utility model outside judge families)
biojudge/data.py        PubMedQA / MedQA / MedMCQA loaders -> lettered single-best-answer items with a gold key
biojudge/prompts.py     every prompt template, in one file so the paper appendix quotes them verbatim
biojudge/textutil.py    answer extraction, rationale validation, verdict parsing
biojudge/perturb.py     the six deterministic content-free edits, template padding, fallback normalizer
biojudge/llm.py         async client with SQLite cache, retries, concurrency; openai / anthropic / mock providers
biojudge/pipeline.py    stages: solo -> hard negatives -> forced-answer rationales -> perturb -> judge -> select
biojudge/analysis.py    metrics (Acc, kappa, Flip, PPI, VB, SPB, IC, ASR), views (swap, jury, recusal),
                        item-level cluster bootstrap, LPM with clustered SEs, Kendall's W
biojudge/report.py      writes paper/results_macros.tex, paper/tables/*.tex, paper/figs/*.pdf, runs/<run>/results/*
biojudge/mock.py        deterministic simulator with planted biases (validation only)
tests/test_pipeline.py  12 tests incl. an end-to-end mock run that must recover the planted effects
paper/                  IEEE two-column LaTeX (main.tex), refs.bib, auto-generated macros/tables/figures
paper/paper_text.md     the same paper as plain text with fill-in tokens
```

## Run outputs

`runs/<name>/` holds `items.jsonl`, `solo.jsonl`, `responses.jsonl`, `variants.jsonl`, `trials.jsonl`,
`select.jsonl`, `run_meta.json` and `cache.sqlite`; `runs/<name>/results/` holds `metrics.json`, `per_judge.csv`,
`corrections.csv` and `regression.csv`.

## Before the main run

1. Confirm the model identifiers in `configs/openrouter.yaml` — provider IDs change; record the exact IDs and the
   run date in the paper.
2. Run the pilot and check three things: refusal rate on forced-incorrect rationales, verdict parse rate, and
   whether the padding agent is adding content it should not.
3. `--estimate` prints the call and token budget (about 392k calls, 357M input / 49M output tokens for 300 items).
