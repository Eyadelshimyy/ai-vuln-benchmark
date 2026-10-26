# VULCAN reproducibility manifest

- generated: 2026-10-26T01:30:09+00:00
- git commit: `8308f5b2528be656c0213a92926dcea0ba52cc1b`  **(working tree dirty)**

## Funnel (this corpus version)

- candidates mined: 2032
- executable: 1501
- triggered (Tier-1): 289
- confirmed (Tier-2): 160
- census tier2 flag: True
- verdict tally: {'TRIGGERED': 289, 'COULD_NOT_EXECUTE': 531, 'NOT_TRIGGERED': 1212}

## Corpus: 103 repos

## Artifact hashes (sha256, first 16)

- `corpus_census.json`  8764876083dbadc8
- `gold_set.json`  1b423a5463660654
- `reachable_worklist.json`  6c8e10be6312af19
- `semgrep_fullrepo.json`  7b825bc91c12fff3
- `semgrep_default.json`  4757b7ce673fc005
- `semgrep_owasp.json`  64e68218521b954e
- `semgrep_union.json`  822f3117c4f09c7f
- `repo_archetypes.json`  e1617b5c04023a89
- `provenance_records.json`  b8ffd1ff78976166
- `finalize.json`  df94b3aaf0e910dc

## Verdict definitions

- **executable**: the function ran under the harness without an import/setup error
- **triggered (Tier-1)**: static taint reached a security-relevant sink
- **confirmed (Tier-2)**: the deterministic exploit bank made the sink fire
- **reachable**: a real entry point (http/cli/api/main) reaches the sink in the static call graph
- **REACHABLE_FROM_ENTRYPOINT**: reachable AND confirmed; a disclosure lead pending trust-boundary review
- **real_net_confirmed**: driven to a live exploit by hand against operator-controlled infra
- **COULD_NOT_EXECUTE**: inconclusive -- excluded from safe/vuln labels, never counted as safe