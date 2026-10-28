# VULCAN corpus v2 — application-level extension: selection protocol

Written **before** any v2 discovery, mining or execution (2026-10-26). Every v2 number must
come from artifacts produced under these rules. Deviations are logged in §7 with a reason.

## 1. Purpose and relation to v1

v1 (frozen, 160 confirmed / 156 source-unique) is mostly library or utility code. 125/130
CWE-22 confirmations are functions that open a caller-supplied path, and only 1 confirmed
function is shown end-to-end attacker-reachable. v2 adds **AI-authored Python web
applications**: code with HTTP entry points, where reachability from an attacker can
actually be tested.

v1 is never modified:
- v2 lives in its own root (`calibration/mined_prompts_v2/`) with its own census file.
- Reports show v1, v2 and the union as separate columns.

## 2. Discovery (shortlist only)

- GitHub commit search for the same strong machine signatures as v1
  (`analysis/provenance.py`: Claude Code trailers, Copilot trailers and bot, Devin bot,
  Cursor agent).
- Each signature is combined with one web keyword: `fastapi`, `flask`, `django`,
  `endpoint`, `route`. This only biases the shortlist toward web work. It is not the
  selection criterion (§3).
- Exclusions: forks; repos already in v1; private or unreachable repos.
- Ordering: most recent first. The search result cap (GitHub returns at most 1,000 per
  query) and the dates are recorded.
- Depth, fixed before any lead was screened: 10 pages × 100 commits per query
  (= the 1,000 cap), authenticated GitHub API.

## 3. Application-level screen (decides inclusion; static, recorded for every lead)

The lead is shallow-cloned. It is APP_LEVEL iff all of these hold:
1. **Python-primary:** at least 10 `.py` files outside tests, vendored and virtualenv dirs.
2. **A web application object is constructed in non-test code**, one of:
   - `Flask(`
   - `FastAPI(`
   - `Starlette(`
   - `web.Application(` (aiohttp)
   - `Sanic(`
   - a Django project (`settings.py` with `ROOT_URLCONF` plus a `urls.py` with `urlpatterns`)
3. **At least 3 HTTP route handlers in non-test code**, any of:
   - `@<x>.route(`
   - `@<x>.get/post/put/patch/delete(`
   - `add_url_rule`
   - `add_api_route`
   - Django `path(`/`re_path(` entries inside `urlpatterns`

Every lead's screen record is kept, including rejections and the reason. The discovery →
screen → mined → census funnel is reported in full.

## 3b. Sample size (fixed before discovery results were seen)

- Leads are screened (§3) and mined in one pass, in a **seeded random order over all
  leads** (deviation 2 replaces "every lead is screened").
- APP_LEVEL repos are mined in a **seeded random order** (seed 20261026), with a quota of
  **15 mined repos per agent** (claude-code, cursor, devin, github-copilot): **60 total**.
- A repo counts toward the first of its discovery agents, in alphabetical order, whose
  quota is not yet full.
- A repo that yields 0 AI-attributed on-topic functions does not use quota.
- If an agent cannot fill its quota, the shortfall is reported, not back-filled from another agent.
- Rationale: balanced agents for RQ1, and a bounded cost. Random order avoids choosing
  repos by anything correlated with the outcome.

## 4. Mining (unchanged v1 method)

- `analysis/mine_prompts.py --require-ai-provenance`, which keeps only functions whose
  lines `git blame` to a commit carrying a strong AI signature, using a full clone.
- Sink taxonomy and per-function records are identical to v1.
- Per-repo cap `--limit 80`, chosen before mining so that no single app dominates (v1 had
  one repo with 264 candidates). Truncation by the cap is recorded per repo.
- The repo's pinned commit is recorded (`validation/corpus_v2/repo_pins.json`).

## 5. Execution and labels (unchanged v1 oracle)

- `analysis/corpus_census.py --tier2 --root calibration/mined_prompts_v2 --out corpus_census_v2.json`.
- Same Tier-1 and Tier-2 code at the frozen oracle version. No oracle change is applied to
  v2 unless it is also re-run on v1 and reported as a new oracle version.
- Labels follow `validation/TAXONOMY.md`. CWE-22 subtypes come from `analysis/cwe22_subtype.py`.
- COULD_NOT_EXECUTE is never counted as safe.

## 6. Reachability (the reason v2 exists)

For each confirmed v2 function:
- static entry-point reachability (`analysis/reachability.py`);
- for apps that boot, dynamic probing (`validation/reachability/dyn_prober_v3.py`),
  counted only after a canary-reachability proof.

An app that does not boot is **NOT_TESTED**, never "unreachable". APP_CONFIRMED requires
an observed end-to-end effect.

## 7. Deviations log

1. 2026-10-26: a first unauthenticated run (3 pages × 30 per query) completed 2 of 30
   queries before switching to the authenticated depth above. Its partial results were
   discarded before any lead was screened, so no inclusion decision depended on them.
2. 2026-10-26, before any lead was screened: discovery returned more than 6,600 unique repos,
   more than expected when §3b was written. Screening all of them would mean 10,000+
   clones just to fill 60 mining slots. Changed to: screen in the seeded random order over
   all leads and stop when the quotas are full. The screened set is a random sample of the
   leads, so the APP_LEVEL rate is still an unbiased estimate. Every screened lead is
   reported, including rejections; unscreened leads are counted as such.
3. 2026-10-27, after 72 leads had been screened and before ANY v2 code was executed
   (no v2 oracle verdict existed): added a hard stopping rule. Screening stops when all
   quotas are full **or 1,000 leads have been screened**, whichever comes first. Unfilled
   quotas are reported as shortfall. Reason: the owner judged 15 repos per agent
   non-essential, and the run needs a guaranteed end. The rule depends only on screening
   counts, never on vulnerability outcomes.
4. 2026-10-27, owner decision before ANY v2 code was executed: mining was **stopped early**
   to bound time and RQ2 cost. The repo being mined at the stop was discarded (no partial
   output remained). Final v2 corpus: screened 104 leads ({'APP_LEVEL': 45, 'REJECT': 59}); mined 43 APP_LEVEL repos; 31 yielded AI-attributed candidates; 560 candidates; per agent {'claude-code': 8, 'github-copilot': 11, 'cursor': 9, 'devin': 3}; truncated by cap: 3. Quotas are unfilled; the shortfall is
   reported, not back-filled. Devin is under-represented in v2 (v1 has 29 Devin repos).
   The stopping decision used only mining progress, never vulnerability outcomes.
5. 2026-10-27, execution environment: v2 code ran inside bubblewrap. Your home directory and
   secrets were hidden, there was no external network, loopback was available for the SSRF
   oracle, and the repo was mounted read-only. A first run (kept as
   `corpus_census_v2_rorepo_run.json`) mounted the v2 corpus read-only too, and about 17
   candidates failed only because they wrote into their own vendored tree, which v1 allowed.
   The canonical census (`corpus_census_v2.json`) re-ran with a WRITABLE COPY of the v2 corpus.
   Result: 5 more executable, 0 confirmations lost, 1 gained (`autoclip ai_cover`). Both runs
   are reported. No external network remains a stated difference from v1; about 5 failures
   to execute look network-related. Neither condition can change whether something
   confirms, because the oracle observables are local.

