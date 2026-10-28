# `_repo_src` source-availability audit (Phase E)

Maps frozen confirmed candidates to whether their vendored source is in the clean-clone (tracked) inventory. **Key result:** a clean clone reproduces 160 because those candidates execute from the **embedded source in their tracked candidate JSONs** (self-contained), NOT because every `_repo_src` is tracked. Canonical data: `REPO_SRC_AUDIT.json`.

## V1 (46 confirmation-repos)

- `_repo_src` tracked (needed + committed): **27** repos — includes the 3 whose source is actually consumed by the census (ExposureEclipse, jobbot, otterworks) plus others.
- `_repo_src` present but **UNTRACKED** (supplemental-only; not consumed by the 160's clean-clone reproduction): **19** repos: BerriAI__litellm, Chuqonu__epi-preclean-patent-monitor, TracecatHQ__tracecat, aditya9729__roboeval_challenge, airbytehq__airbyte-python-cdk, atanasvasilevjourney__ORBIS-EQUITY-, c-wenlong__blenderkc, gcnewbief__Log-Grabber, joemccann__radon, oamin__fpl-engine, qndelmar__tbnano, quantam101__profitenginev5, quantam101__tradegate2, royles__AppliedAIModelTraining, tonezzz__chaba, trsdn__macos-notarization-broker, wstomber__devinexecdemo, x-uuid__syforce, z-mio__parse_hub_bot.

Because the committed clean-clone census (`corpus_census_reproduced.json`) returned the 160 confirmed set-equal to pinned using ONLY tracked files, every one of the 160 is reproducible from the tracked inventory by construction — the untracked `_repo_src` dirs above are not consumed by those 160 confirmations.

## V2 (13 confirmation-repos)

All 13 v2 confirmation-repos have **untracked** `_repo_src` in the working tree; the v2 census (`corpus_census_v2.json`) is itself untracked/new and was produced from the working-tree copy. v2 reproducibility therefore depends on committing the v2 corpus + its `_repo_src` (deliverable in the cleanliness plan), OR on each candidate being self-contained (to be verified before any v2 freeze).

## gesh75__netlog-ai (the exception within would-be V1)

`_repo_src` UNTRACKED and the `load_topology` candidate is NOT self-contained (it needed the vendored source to execute), so it is CNE in a clean clone and is correctly **outside** the frozen 160. It appears only as a supplemental finding. See supplemental/SUPPLEMENTAL_FINDINGS.md.

## Caveat

Do not infer 'all required source is tracked' from 'clean clone returned 160'. The correct statement is: the 160 are reproducible from tracked inputs (embedded candidate source + the 3 committed `_repo_src`); the other repos' `_repo_src` is supplemental working-tree material.
