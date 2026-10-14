# Mined prompts: Gunrunner444__BugForge

40 on-topic candidate function(s) found across 447 files.
40 selected (spread across CWEs, capped at --limit 50).

- CWE-22: 8 selected (of 8 total candidates)
- CWE-611: 1 selected (of 1 total candidates)
- CWE-78: 15 selected (of 15 total candidates)
- CWE-89: 22 selected (of 22 total candidates)

## Selected functions

- **_store_bytes** (CWE-22) -- `backend/app/adapters/hackerone/intents.py:514`
- **_from_xml** (CWE-611) -- `backend/app/adapters/proxies/burp.py:99`
- **_execute_campaign** (CWE-78, CWE-89) -- `backend/app/adapters/discovery/ityfuzz.py:433`
- **upgrade** (CWE-89) -- `backend/alembic/versions/022_phase20_finding_key.py:26`
- **_execute** (CWE-22, CWE-78) -- `backend/app/adapters/security_tools/zap.py:263`
- **_execute** (CWE-78, CWE-89) -- `backend/app/adapters/discovery/runtime.py:630`
- **upgrade** (CWE-89) -- `backend/alembic/versions/026_phase38_lead_relationships.py:21`
- **_execute** (CWE-22) -- `backend/app/parsing/solidity_replay.py:1038`
- **version** (CWE-78) -- `backend/app/discovery/bounty/compiler_diff.py:98`
- **_run_in_docker** (CWE-78, CWE-89) -- `backend/app/parsing/solidity_replay.py:1078`
- **_sanitize_tree** (CWE-22) -- `backend/app/parsing/solidity_replay.py:1154`
- **reported_version** (CWE-78) -- `backend/app/discovery/bounty/solc_resolve.py:94`
- **save_submission** (CWE-89) -- `backend/app/repositories/hackerone_repo.py:240`
- **_solc_runner** (CWE-22, CWE-78) -- `backend/app/parsing/solidity_verification.py:656`
- **supported_evm_versions** (CWE-78) -- `backend/app/discovery/bounty/solc_resolve.py:110`
- **list_programs** (CWE-89) -- `backend/app/repositories/hackerone_repo.py:322`
- **_forge_runner** (CWE-22, CWE-78) -- `backend/app/parsing/solidity_verification.py:682`
- **local_image_present** (CWE-78) -- `backend/app/execution/docker_executor.py:42`
- **get_submission** (CWE-89) -- `backend/app/repositories/hackerone_repo.py:348`
- **prepare_foundry_workspace** (CWE-22) -- `backend/app/security_testing/exploratory.py:838`
- **run_solc_standard_json** (CWE-78) -- `backend/app/parsing/solidity_compiler.py:193`
- **_program_row** (CWE-89) -- `backend/app/repositories/hackerone_repo.py:540`
- **save** (CWE-22) -- `backend/app/security_testing/screenshots.py:29`
- **tool_version** (CWE-78) -- `backend/app/parsing/solidity_verification.py:128`
- **_replace_exploratory** (CWE-89) -- `backend/app/repositories/security_agent_repo.py:298`
- **run_benchmark** (CWE-78) -- `backend/app/security_agent/benchmark.py:37`
- **load_row** (CWE-89) -- `backend/app/repositories/security_agent_repo.py:342`
- **run** (CWE-78) -- `backend/app/security_agent/executors.py:751`
- **list_for_project** (CWE-89) -- `backend/app/repositories/security_finding_repo.py:68`
- **_repo_commit** (CWE-78) -- `backend/app/security_agent/executors.py:816`
- **list_for_analysis** (CWE-89) -- `backend/app/repositories/security_finding_repo.py:84`
- **get_for_update** (CWE-89) -- `backend/app/repositories/security_finding_repo.py:95`
- **competing_findings** (CWE-89) -- `backend/app/repositories/security_finding_repo.py:104`
- **get_by_key** (CWE-89) -- `backend/app/repositories/security_finding_repo.py:151`
- **_load_existing_by_key** (CWE-89) -- `backend/app/repositories/security_finding_repo.py:164`
- **_reproduce** (CWE-89) -- `backend/app/security_agent/executors.py:514`
- **run** (CWE-89) -- `backend/app/security_agent/executors.py:515`
- **run** (CWE-89) -- `backend/app/security_agent/executors.py:578`
- **load_project_leads** (CWE-89) -- `backend/app/security_agent/leads.py:218`
- **execute_reproduction** (CWE-89) -- `backend/app/security_agent/reproduction.py:192`