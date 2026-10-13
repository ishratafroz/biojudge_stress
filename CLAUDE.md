# BioJudge-Stress: run the experiment and finish the paper

## Goal (from the user)
Run the code in `biojudge-stress-code/biojudge-stress/`, complete the analysis, and fill every incomplete
part of the paper (`BioJudge-Stress_paper.docx`, same content as `paper/paper_text.md`). Output a **Word
document**; LaTeX does not need to be maintained. Never put simulator (mock) numbers in the paper.

## Layout
- `BioJudge-Stress_paper.docx` / `_preview.pdf`: the user's original draft. Every number is a `[Token]`,
  and sentences in «guillemets» are expected directions that must be checked against real results.
- `biojudge-stress-code/biojudge-stress/`: the code. The README there explains the pipeline.
  - `.venv/`: the original **Windows** venv (uv, CPython 3.12.7). Unusable on the current Linux host.
  - `.venv_linux/`: the venv to use now (uv, CPython 3.12.13). **Always use `.venv_linux/bin/python`.**
    Created 2026-09-26 with `~/.local/bin/uv venv --python 3.12 .venv_linux` +
    `uv pip install -r requirements.txt python-docx pymupdf` (`python3 -m venv` fails: no ensurepip).
  - `.openrouter_key`: an OpenRouter key the user pasted in chat. The account has **$0 credit**
    (HTTP 402), so it is unusable. The user was told to revoke it.

## Environment change on 2026-09-26 (IMPORTANT — read before anything else)
The project moved from the Windows/RTX-3050 laptop to a **Linux host with 4x NVIDIA L40S (46 GB each),
64 CPUs, 503 GB RAM**. Everything in this file that describes Windows paths, `D:\`, `.venv/Scripts`,
PowerShell or the 6 GB GPU is history, not instructions.
- The old driver (`scripts/run_local.sh`, phases A/B/C) is a **Windows** script and is not used here.
  Phase A (perturb) had already finished on the laptop and is in the cache, so only judging remained.
- Ports on this host: **11434** = the user's own Ollama (models `llama4:scout`, `llama3.2:3b`) — do not
  touch. **11435** = the user's *other* service, a `serve_adapter` for `aaditya/Llama3-OpenBioLLM-8B`
  — do not touch. Our server therefore runs on **11436**.
- `scripts/serve_l40s.sh <slots>`: our Ollama server. GPUs 0,2,3, `OLLAMA_MAX_LOADED_MODELS=4` so all
  four judges stay resident, 8192 ctx/slot, KV q8_0, flash attention, and `LLAMA_ARG_CACHE_RAM=0`
  (same host-prompt-cache problem as on the laptop: it was saving 50-70 MiB of slot state per idle slot).
- `configs/ollama_l40s.yaml`: same run dir, run_name and cache as `ollama_local.yaml`; only the endpoint
  (11436), `concurrency: 48` and `interleave_models: true` differ. **The cache key does not include the
  endpoint**, so all work done on the laptop was reused with no repeat calls.
- `biojudge/pipeline.py`: new `interleave_by()` + `runtime.interleave_models` (default False). The
  model-major ordering was a workaround for a 6 GB GPU holding one model at a time; with every judge
  resident, rotating judges keeps all four servers busy. Ordering never changes results. 12 tests pass.
- Models re-pulled here: `qwen2.5:3b`, `gemma2:2b`, `ministral-3:3b` (llama3.2:3b was already present).
  Digests/quantizations are in `runs/local/model_tags.json` and as comments in `configs/ollama_l40s.yaml`:
  llama/qwen/ministral Q4_K_M, gemma Q4_0. `phi4-mini` was NOT re-pulled — phase A is fully cached.
- Throughput: 3.1 calls/s at first, **10-11 calls/s** after disabling the host prompt cache and
  interleaving judges. Judge+select (63k calls) therefore takes ~1.7 h here instead of ~15 h.
- **No Word, LibreOffice or pandoc on this host**, so the .docx cannot be rendered to PDF for a visual
  check. Validate it structurally (python-docx) and have the user open it in Word.

## Decisions made
1. The OpenRouter full run (~$25–30) was not possible because the user has no money. We switched to
   **local Ollama on the RTX 3050 (6 GB)**.
2. Models (small, one per family, Q4 GGUF):
   - judges and authors: `llama3.2:3b`, `qwen2.5:3b`, `gemma2:2b` (Q4_0), `ministral-3:3b`
   - utility agent (padding and normalizer): `phi4-mini`
   (`configs/openrouter.yaml` and `pilot.yaml` were also updated: Gemma-2-9B became Gemma-3-12B and
   Mistral-7B became Mistral-Nemo, because the old IDs were removed from OpenRouter. These configs are not
   used now.)
3. Scale (`configs/ollama_local.yaml`, run_name `local`): 25 items per domain (75 sampled), all 4
   protocols, `attack_pairs_per_item: 2`, select enabled, bootstrap 2000, seed 2026. The user chose the
   **full version** (all protocols) over the faster cut.
4. The paper is written in `paper/paper_final.md` (custom markdown) and converted to .docx by
   `scripts/make_docx.py`.

## Code changes made
- `biojudge/pipeline.py`: job lists are **model-major** (solo, generate, pairwise are judge-major;
  select is sorted by judge) to avoid Ollama model swapping. Results are unaffected.
- `biojudge/report.py`: `_fig_delta` and `_fig_select` also save `.png` (300 dpi) for Word.
- New `configs/ollama_local.yaml`, and `configs/ollama_local_phaseB.yaml` (judges = llama, qwen, gemma
  only; same run dir and cache).
- New `scripts/run_local.sh`: phased driver. Each phase starts its own Ollama server (port **11435**,
  models in `D:\ollama_models`) with the right slot count, then stops it and kills leftover
  `llama-server` runners:
  - A: `perturb` (phi) at 4 slots
  - B: `judge,select` with phaseB config at 8 slots (warms the cache)
  - C: `judge,select` with the full config at 4 slots (only Ministral calls are new)
  It logs to `runs/local/driver.log` and `runs/local/log_phase{A,B,C}.txt`.
- `D:\ollama_models\serve.sh <parallel>`: server launcher. The env sets `LLAMA_ARG_CACHE_RAM=0`
  (important, see below), KV cache q8_0, flash attention, 4096 ctx per slot, MAX_LOADED_MODELS=1.
- New `scripts/fig_pipeline.py`: draws Fig. 1 to `paper/figs/fig_pipeline.png` (done).
- New `scripts/make_docx.py`: markdown to docx. It fills `[Macro]` tokens from
  `runs/local/results/metrics.json`, converts `paper/tables/*.tex` (written by `biojudge.report`) into
  Word tables, and supports `{{TABLE:name|caption}}`, `{{FIG:path|caption|width}}`,
  `{{CAPTION:...}}`, pipe tables, `REF:` blocks, and `x_{sub}` / `x^{sup}`. It prints any unfilled
  tokens. Tested on mock data in the scratchpad only.

## Lessons / gotchas
- 6 GB VRAM: at 8 slots, phi4-mini (28/33 layers) and ministral-3 (21/27) spill to CPU and slow down.
  Use 4 slots for them and 8 for llama, qwen and gemma.
- Killing `ollama.exe` does **not** kill its `llama-server.exe` runner. An orphan held about 6 GB of RAM.
  Kill by PID with `/T`, and clean up runners whose command line contains `ollama_models`.
  **Never kill `ollama.exe` by image name**, because the user's own Ollama (port 11434, tray app) must
  stay untouched.
- llama-server's default host prompt cache (`--cache-ram`, up to 8 GiB) plus the orphan exhausted RAM
  (0.3 GB free). Claude Code then **reaped the background driver**. Fixed with
  `LLAMA_ARG_CACHE_RAM=0`. After the fix the runner uses about 0.9 GB.
- The SQLite cache commits every 200 calls, so progress is not visible in the cache until a commit.
  Everything is resumable: re-running the driver repeats nothing that was already cached.
- Rendering the docx: Word COM works (`render.ps1` in the scratchpad: docx to PDF), and pymupdf (in the
  venv) turns PDF pages into PNG. No LibreOffice, pandoc or docx-js is installed.
- Measured throughput is about 1.1–1.5 judge calls/s (8 slots) and about 0.7 (Ministral, 4 slots).

## Results so far (real, local run)
- Items: 75 sampled, **73 kept**; 598/600 rationales valid; hard-negative distractors on 47/75 items.
- Solo accuracy: llama 0.71, mistral 0.67, qwen 0.61, gemma 0.57. By domain: literature 0.74,
  clinical 0.58, pharmacology 0.60. Parse rate 99–100%.
- Spot check: R− rationales argue fluently for the distractor, as intended.

## Progress / status (update this section)
- [x] Environment: Linux L40S host, `.venv_linux`, Ollama server on 11436, 4 judge models pulled
- [x] Stage 1: items, solo, generate (`runs/local/{items,solo,responses}.jsonl`)
- [x] Phase A (perturb): finished on the laptop, 5,840 text variants, 8,585 s (`run_meta_perturb.json`)
- [x] Fig. 1 drawn; docx builder written; report->docx chain re-validated end to end on the mock run
      under Linux (12 tests pass; `[report]` leaves only the 6 Flip-under-swap macros unfilled, which is
      by design — do not reference `[FlipSwap]` or `[FlipFull]` in the text)
- [x] Early health check: verdict parse rate per judge. On a kind-balanced 864-verdict subsample
      ministral looked bad (89.4%), but that sample over-weighted attack conditions; **over the full
      56,064 verdicts the rates are qwen 99.5%, gemma 99.1%, llama 98.3%, ministral 97.1%** (1.5%
      overall). Ministral's failures are NOT truncation (`finish_reason=length` never fires, max 232 of
      300 tokens); it just omits the `[[A]]` tag, and raising max_tokens makes it *worse*. Config left
      unchanged; the per-judge rates are reported in Section V. Unparseable verdicts score 0.5.
- [x] Setup section corrected in `paper/paper_final.md`: two hosts instead of "RTX 3050 / 18 hours",
      no unverifiable Ollama version for the laptop, digests recorded. Two markers are left to fill
      with measured values: `@@WALLCLOCK@@` and `@@PARSEBYJUDGE@@`.
- [x] judge + select, one run on the L40S host: 56,064 pairwise + 7,008 select verdicts in 5,966 s
      (~10 calls/s), 831 parse failures (1.5%), zero API failures. `runs/local/{trials,select}.jsonl`.
- [x] Report: `.venv_linux/bin/python -m biojudge.report --run runs/local --config configs/ollama_l40s.yaml --paper paper/`
      207 macros, 6 unfilled (the Flip-under-swap ones, by design). Tables + Figs 2-3 in `paper/`.
- [x] All results-dependent prose written in `paper/paper_final.md`. Every «expected» claim was checked
      against the real numbers; several were **contradicted and rewritten** (see below).
- [x] Build + validate: `BioJudge-Stress_paper_final.docx` (~9.3k words, 7 tables, 3 figures).
      `scripts/check_docx.py` reports no leftover tokens, no empty rows, consistent fonts.
- [x] My Ollama server stopped, GPUs freed; the user's 11434 and 11435 services verified still up.
- [ ] REMAINING FOR THE USER: open the .docx in Word for a visual pass (no renderer on this host);
      revoke the OpenRouter key; decide whether to add Zhenni Liang's email (only
      UmmeAfroz@my.unt.edu is in the author block).

## Headline results (real, 2026-09-26) — several contradict the draft's «expected» claims
- Judges are near chance: Acc 54.7% [51.5, 58.0], mean κ 0.09, Flip 39.2%. Llama is *below* chance
  (47.6%; only 35.8% on literature). This reframes the paper: the manipulations steer an already
  near-arbitrary decision.
- **RQ1 interaction is NULL**: IC = 0.9 pp (p = 0.702), β(SW×PW) = −1.8 pp (p = 0.372). The draft
  expected super-additivity. But β(WA×PW) = −5.6 pp (p = 0.004) — position × length *is* super-additive.
  Main effects large: WA −19.8, PW −7.6, PR +12.4, SW −8.7 pp (all p < 0.001).
- Self-preference is model-specific, not general: ministral +59.2 pp, qwen −12.7, gemma −7.9, llama +5.1.
- Top attack is **injection** (note to evaluator) at 40.7%, not authority/citation. ASR_any 55.8%.
- Only the normalizer helps (ΔAcc +3.1 [0.2, 5.9]; ASR_any −9.1). Debias clause: ΔAcc 0.0, ASR_any +3.4
  (CI includes 0). Juries significantly *worsen* Flip (+6.7, +7.4) and VB (+3.0) while cutting ASR.
- Domain stability: only accuracy ranking is stable (W 0.82, p 0.032); VB/SPB/ASR are not (0.60-0.64,
  ns). The draft's claim that ASR carries across domains is **wrong**.
- Competence split: ASR concentrates on unknown items (61.0 vs 53.2) but VB/SPB/Flip do **not**.
- Best-of-4: clean 28.8%, attacked **19.9% (below chance)**, capture 20.1 -> 40.5%. Selectors
  *under*-select position 1 (−3.9 pp) even though the same models prefer slot A pairwise.

## Resume commands
```bash
cd ~/files/biojudge-stress-code/biojudge-stress
ps -eo pid,cmd | grep "[b]iojudge.run"           # is the run still alive?
tail runs/local/log_judge_l40s.txt               # progress (every 4,000 calls)
curl -s http://127.0.0.1:11436/api/tags          # our server up? if not: bash scripts/serve_l40s.sh 12
# restart judging; everything already cached is skipped:
setsid nohup env OLLAMA_API_KEY=EMPTY .venv_linux/bin/python -u -m biojudge.run \
  --config configs/ollama_l40s.yaml --stages judge,select > runs/local/log_judge_l40s.txt 2>&1 &
```
To stop our Ollama without touching the user's: find the `ollama serve` PID whose children are
`llama-server` on port 11436, kill the children by PID first, then the parent. **Never** kill by name.
