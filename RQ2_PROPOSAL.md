# RQ2 — AI vulnerability-detector evaluation: experiment proposal

**For:** thesis supervisor review and approval
**From:** Eyad Hassan Abdelfatah — VULCAN (B.Sc. thesis)
**Status:** tooling built and dry-run-validated; awaiting approval of model panel + budget

## 1. Why this experiment
VULCAN's central question asks how accurately detectors identify vulnerabilities
in AI-authored code **when scored against execution-confirmed ground truth.** The
static-analysis baselines (Bandit, Semgrep) are done. The remaining — and central
— comparison is **AI models as detectors.** Without it, the thesis has the
execution oracle but not the detector comparison its research question promises.

## 2. What is already in place (no further engineering needed)
- **Frozen corpus:** 1,216 AI-tool-attributed functions from 34 real GitHub repos;
  provenance recorded per finding (commit + tool + source hash); ~90% line-level
  AI-attributed.
- **Execution oracle:** Tier-2 exploit bank; soundness battery passes (0 FP / 0
  miss on its cases); per-CWE confirmation predicates documented.
- **Ground-truth labels:** positive = execution-**confirmed**; negative =
  executed-but-no-trigger; **inconclusive** (triggered-unconfirmed, non-executable)
  **excluded** — not counted as safe.
- **Orchestrator** (`detector_consortium.py`): queries each model blind with a
  fixed neutral prompt, caches every call, scores against the labels, and reports
  a leaderboard + a majority-vote "consortium" detector. Validated with a mock run.

## 3. The ask (three decisions)
1. **Approve this 5-model panel** (credible, versioned, citable; diverse vendors +
   code specialists). All reachable through **one OpenAI-compatible endpoint
   (OpenRouter)** — one key, one base URL, so the run is simple and reproducible:

   | Role | Model (id) |
   |---|---|
   | Frontier, cheap | `openai/gpt-4o-mini` |
   | Frontier, different lab | `anthropic/claude-3.5-haiku` |
   | Frontier, different lab | `google/gemini-2.0-flash` |
   | Open code specialist | `qwen/qwen-2.5-coder-32b-instruct` |
   | Open, strong code | `deepseek/deepseek-chat` |

   Plus a majority-vote "consortium" row (computed, free). **Not** abliterated/
   "dealigned" checkpoints — no detection benefit, degraded reasoning, not citable.
   (Exact ids to be confirmed against the provider's live list before the run.)

2. **API budget: ~US $5–20 total.** At `--balance 2` the evaluation is ~318 calls
   per model (≈106 confirmed-positive + ~212 executed-negative). Across 5 models
   that is ~1,600 calls — a few dollars for the small/open models, a few more for
   the frontier ones. A **10-case pilot** runs first to verify validity and cost
   before the full spend; caching makes any re-run free.

3. **Sign off on the frozen protocol** (section 4) so the result is defensible.

## 4. Frozen protocol
- **Prompt:** one fixed, neutral security-review prompt; identical for every model;
  strict JSON output (`vulnerable`, `cwe`, `confidence`, `reason`).
- **Decoding:** temperature 0; recorded `max_tokens`; model name + version + date
  logged per call.
- **Population:** the frozen label set (confirmed positives vs executed negatives);
  inconclusive excluded and reported separately.
- **Scoring:** precision, recall, F1, FPR, and a **confusion matrix per CWE** and
  aggregate; abstentions / invalid JSON / API errors recorded and scored
  explicitly (not silently dropped).
- **Comparison framing:** VULCAN provides the **reference labels**; detectors
  (static + models) are scored *against* those labels. VULCAN is **not** placed in
  the same precision/recall row as a detector.
- **Reproducibility:** cached per-(model, finding) answers; prompt, model list, and
  scoring script frozen under a tagged commit.

## 5. Deliverables
- A detector leaderboard (models + static baselines + majority-vote consortium).
- Per-CWE confusion matrices (small-N caveats stated).
- The **evidence-policy table**: how detector agreement changes across static alert
  → execution-confirmed → reachable → end-to-end.
- A short discussion of where models beat / match / fail vs execution (expected:
  decent on SSRF/SQL patterns, weak on XXE and traversal; none matching execution).

## 6. Risks / limits (stated up front)
- Small corpus → results describe this sample, not AI code in general.
- Confirmed positives are modest in number → per-CWE cells are small; report counts
  and avoid over-reading percentages.
- Model endpoints drift → the cache + frozen prompt + logged versions are what make
  the run reproducible.

**Requested decision:** approval of the model panel and a small API budget so this
single, central experiment can be completed against the already-frozen corpus.
