# RQ2 pre-registration — AI models as detectors of execution-confirmed weaknesses

## AMENDMENT A1 (2026-10-27): the frozen plan for the paid run. Supersedes §2 and the panel in §5.

Written before ANY model call on this population. Earlier sections are kept as the record.

| | frozen value |
|---|---|
| label set | `validation/rq2/rq2_labelset_v3.json` (`v3-v1v2-srcunique-twins`), items-sha256 `61dace958e3fe874…` |
| corpora | v1 `corpus_census_reproduced.json` + v2 `corpus_census_v2.json` (hashes checked at load) |
| positives | **193** source-unique confirmed: CWE-22 156 (COMPOSED 4 / DIRECT_CALLER_PATH 152, hand-reviewed), CWE-918 30, CWE-78 4, CWE-89 3 |
| negatives | **386** NOT_TRIGGERED, 2 per positive, drawn per generating agent (seed 20261026), no shortfall |
| pairs | **34** execution-verified vulnerable/fixed twins (`validation/twins/`): 30 SSRF (guard v2) + 4 COMPOSED path. Scored ONLY pairwise. Never pooled as negatives: they still TRIGGER, which the main labels treat as inconclusive |
| excluded | 162 TRIGGERED-unconfirmed, 741 COULD_NOT_EXECUTE (never counted safe) |
| panel | `models_panelA.json`: anthropic/claude-haiku-5.5, openai/gpt-6-luna, deepseek/deepseek-v4-pro-0813 via OpenRouter; temperature 0 + seed where supported (the two reasoning models support neither); max_tokens 4000; reasoning effort = provider default |
| conditions | A (primary) `v2-threat`; B (secondary) `v2`. Same items, same settings |
| calls | 1,839 per condition (613 × 3), 3,678 total |
| budget | estimate $1.56–$7.77 for both conditions; hard cap `--max-cost 9.50` (owner budget $10) |

**Metrics (fixed):**
- **Primary:** per model, recall and FPR (Wilson CIs) and MCC on the 579 main items,
  plus precision at benchmark balance and projected to the corpus prevalence of 11.5%.
- **Pairwise:** pair accuracy on the 34 twins (original flagged AND fixed cleared), with
  the both-vuln / both-safe / reversed breakdown.
- **Secondary:** framing effect A vs B (paired McNemar, `analysis/rq2_compare.py`);
  results per generating agent, per CWE, per CWE-22 subtype and per length tercile;
  consortium rules.

**Sensitivity analyses:**
1. **Excluding dipcatcher** (12 of the 37 v2 positives).
2. **Hand adjudication of model-flagged negatives.** Draw a seeded random sample of 30
   negatives that at least 2 of 3 models flag under condition A, hand-check each, and report
   the share that look like real vulnerabilities as an estimate of the oracle's miss rate.
   **Labels are never changed because of model answers.**

**Not done:** no prompt changes, no re-runs to improve scores, no item exclusion after
seeing answers, no extra models before this run completes.

**Run order:**
1. The owner commits the frozen state.
2. Smoke check: `--smoke 2`, 6 calls.
3. Re-estimate the cost from the observed tokens.
4. Run condition A, then condition B, each with `--live --max-cost`.

---

Status: **frozen before any v2 model call** (2026-10-26). No v2 answer exists yet.
Supersedes the population in `RQ2_PROPOSAL.md` (which predates the frozen census).

## 1. Why a new run (and why P5 is kept only as a pilot)

The P5 run (`consortium_confirmed_P5.*`) remains reproducible from its cache with
`--protocol v1`. All 8 confusion matrices were re-derived exactly, with 0 calls
(`analysis/test_rq2_pipeline.py::test_legacy_p5_cache_still_resolves` guards this).
It is **not** the RQ2 result, for three verified reasons:

1. **Wrong population.** P5 was built from `corpus_census_P5_final.json` (1,216 candidates,
   51 positives = 47 distinct functions). The frozen census has 160 confirmed rows
   (156 source-unique), so 109 frozen confirmations were never evaluated.
2. **Possible label leakage, unauditable.** P5 called `opencode run` with the repository as
   its working directory. `opencode run` drives an agent that can use tools, and the repo
   contains the ground-truth labels. The cache stored only the parsed answer, so it cannot
   be shown whether any model read files. NOT VERIFIED either way; treated as a threat.
3. **Construct mismatch.** About 96% of CWE-22 positives (121/126 in this label set) are
   `DIRECT_CALLER_PATH`: the function opens whatever path its caller passes
   (`analysis/cwe22_subtype.py`; the heuristic was hand-checked on 10 random rows, all
   correct). VULCAN's label means *conditional* exploitability, because the oracle drives
   each parameter with attacker input. The v1 prompt asks about "a real, exploitable
   vulnerability … as written", so a model that answers "safe if the caller is trusted"
   is scored as a miss for answering a different question.

## 2. Population (frozen)

`validation/rq2/rq2_labelset_v2.json`, built by `analysis/rq2_labelset.py`. It is never
overwritten; the script refuses to replace it.

| | |
|---|---|
| census | `corpus_census_reproduced.json`, sha256 `b0efdd6a…` (checked at load) |
| unit | source-unique function: (repo, sha256(source)) |
| positives | 156: all confirmed (CWE-22 126 [direct 121 / base-join 5], CWE-918 26, CWE-78 4) |
| negatives | 312: verdict NOT_TRIGGERED, 2 per positive, sampled per generating agent (seed 20261026) |
| excluded | TRIGGERED-unconfirmed 117 rows, COULD_NOT_EXECUTE 565 rows (never counted safe) |
| items hash | `e1038bee91d4646b…` |
| corpus prevalence (projected precision) | 156 / (156 + 1181) = 11.7%, same unit |

Every item's source is re-hashed against the frozen hash before any call.

## 3. Conditions

| Condition | Protocol flag | Prompt | Role |
|---|---|---|---|
| **A (primary)** | `v2-threat` | `v2-untrusted-params-6cwe`: states that every parameter is attacker-controlled | matches the oracle's construct |
| B (secondary) | `v2` | `v1-openended-6cwe` (unchanged P5 wording) | measures the framing effect, pairwise against A |

Both conditions:
- run in a fresh empty working directory;
- store the full raw output, return code and timestamp in the cache;
- use temperature 0 and a fixed seed where the model supports them (two reasoning models support neither, §5);
- give one answer per item, with no sampling and no self-consistency;
- make no CWE hint;
- use strict JSON parsing: an invalid answer is recorded as invalid, never imputed.

## 4. Analysis plan (fixed now)

- **Primary, per model, condition A:** recall and FPR (prevalence-free) with Wilson CIs;
  MCC; precision at the benchmark balance and projected to 11.7%. Computed on the common
  set every model answered; coverage is reported separately.
- **Secondary:**
  - Framing effect: per model, recall(A) − recall(B) on positives and FPR(A) − FPR(B) on
    negatives, using paired McNemar on the same items. Reported separately for
    `DIRECT_CALLER_PATH`, `BASE_JOIN_OVERRIDE`, CWE-918 and CWE-78.
  - By generating agent (claude-code / cursor / devin / github-copilot): recall and FPR
    per cell with n. Descriptive only, because the agent is confounded with the repository.
  - Consortium rules (majority / any / unanimous) against the best single model:
    McNemar plus a paired bootstrap of ΔF1 (as in P5).
- **Exploratory, labelled as such:**
  - Same-vendor detection: an Anthropic detector judging claude-code-generated code, against the other generators.
  - Length confound: negatives are longer than positives (median 878 vs 559 characters); recall and FPR by length tercile.
- **Not done:**
  - No prompt iteration on this label set.
  - No re-run of a model to "fix" its scores.
  - No exclusion of items after seeing answers.
  - Any deviation is reported as a deviation.

## 5. Route, setup, and approval (no call has been made)

**Route: OpenRouter's chat API** (`models_openrouter.json`), not the opencode CLI. This is
a plain chat completion: no agent, no tools, no working directory, so the leakage threat
in §1.2 cannot arise.

Each answer records:
- the model and provider that actually served it;
- token, reasoning-token and cost usage;
- the finish reason, generation id and timestamp;
- the raw text.

Model IDs are pinned (dated where OpenRouter offers it). Pricing and capabilities as of
the freeze are in `openrouter_models_snapshot.json`.

| Detector | OpenRouter id | temperature | seed |
|---|---|---|---|
| claude-haiku-5-5 | anthropic/claude-haiku-5.5 | not supported (reasoning model) | — |
| gpt-6-luna | openai/gpt-6-luna | not supported (reasoning model) | supported, but not set by default |
| deepseek-v4-pro | deepseek/deepseek-v4-pro-0813 | 0 | 20261026 |
| kimi-k3 | moonshotai/kimi-k3 | 0 | 20261026 |
| qwen3.8-max | qwen/qwen3.8-max-0902 | 0 | 20261026 |

`max_tokens` is 4000 because reasoning tokens count against it. If a model spends the
whole cap on reasoning, the call is recorded as an error and retried; it is never scored
as an answer. The generation settings are part of the cache key.

**Owner decisions before spending:**
1. **Reasoning effort.** Default = provider default (`"reasoning_effort": null`). Setting it
   to `"low"` for all five makes cost predictable and uniform. kimi-k3 alone is $2.8–18.8
   per condition at default effort. Decide before the run: changing it later
   starts a new cache namespace.
2. **Provider pinning for the open-weight models** (deepseek, kimi, qwen). OpenRouter may
   route to different hosts with different quantizations. Either pin with `"provider":
   {"order": ["<host>"], "allow_fallbacks": false}`, or accept routing and report
   `served_provider` per answer. It is recorded either way.

**Key handling.** Never paste the key into the chat. Keep it in a file only you can read
(for example `~/.config/openrouter_key`, mode 600) and pass it only through the environment:

    OPENROUTER_API_KEY="$(cat ~/.config/openrouter_key)" python3 analysis/detector_consortium.py ...

The key is never written to the cache, the outputs, or the logs (tested). If it is
missing, nothing is sent.

**Estimated cost** (`--pricing validation/rq2/openrouter_models_snapshot.json`, 300–2000
output tokens per call): **$4.70–$27.76 per condition** (2,340 calls).

**Steps, each needing explicit approval:**
1. Smoke check, **10 calls, ≤ $0.50**:

       OPENROUTER_API_KEY="$(cat ~/.config/openrouter_key)" python3 analysis/detector_consortium.py \
         --models models_openrouter.json --labelset validation/rq2/rq2_labelset_v2.json \
         --protocol v2-threat --smoke 2 --live --max-calls 10 --max-cost 0.5 --out /tmp/rq2_smoke.json

   Check that each model returns parseable JSON, that `finish_reason` is `stop`, and the
   real cost and reasoning tokens per call. Then re-estimate the full run from the observed
   output tokens. Smoke answers are reused by the full run.
2. Condition A, full: 2,340 calls, capped with `--max-calls 2400 --max-cost <budget>`,
   output to `validation/rq2/rq2_A_v2threat.json`.
3. Condition B (optional, recommended): the same with `--protocol v2`, output to
   `rq2_B_v2.json`. Then run `analysis/rq2_compare.py`.

**Archive** `.consortium_cache/` (git-ignored; it holds the raw answers) with the release.

## 6. Known limitations (state them in the paper)

- Positives are sink-selected, and most CWE-22 positives are conditional (§1.3). Results
  describe detection of *execution-confirmed conditional exploitability*, not of attacker-reachable vulnerabilities.
- The corpus repos are public, so the code may be in model training data. Commit dates are
  in `validation/rq1/commit_model_attribution.json` for a post-cutoff sensitivity check.
- Two detectors are reasoning models that reject `temperature`; their answers are not deterministic. There is a single answer per item. A stability re-run on a subset is recommended.
- Negatives are "not triggered under the tested conditions", not proven safe.
