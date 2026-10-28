# VULCAN — unified reachability worklist (v1 + v2)

Canonical source: `REACH_WORKLIST.json` (this table is generated from it). **Static evidence only** — the AST call-graph screen plus frozen censuses. No application was booted and no HTTP request was sent to build this. Route-reachability here is a *screen*, not a demonstration.

## Counts

- v1: 160 confirmed rows (156 source-unique)
- v2: 37 confirmed rows (37 source-unique)
- repo overlap v1∩v2: none (disjoint by construction)
- combined: 197 rows / 193 source-unique

> v1 pinned census = 160 candidate rows (156 source-unique, 149 name-dedup). v2 = 37 rows (37 source-unique). v1/v2 repo sets are disjoint by construction (v2 discovery excluded in-v1 repos), so combined = 197 rows / 193 source-unique. The '34' elsewhere is RQ2 twin PAIRS (vuln/fixed), a different unit, not vulnerabilities.

Route-reachable (static screen): v1 43, v2 15. Attacker-origin flagged: 3 (v1:nthplusio__skills:skills/scoussens-skills/visualize-spec/evals/fixtures/order-hooks/src/order_hooks/webhooks.py:send_webhook, v1:quantam101__profitenginev5:master_distillation/dataset.py:digest, v1:trsdn__macos-notarization-broker:scripts/broker.py:write_json).

## Route-reachable candidates (adjudication worklist)

Sorted by research priority: SSRF, path-write, path-read, SQL, command. `origin=attacker` first within each.

| id | corpus | cwe | sink | param | origin | call chain | prior finding |
|---|---|---|---|---|---|---|---|
| v1:nthplusio__skills:skills/scoussens-skills/visualize-spec/evals/fixtures/order-hooks/src/order_hooks/webhooks.py:send_webhook | v1 | CWE-918 | ssrf-url | ? | attacker | create_order → send_webhook | APP_CONFIRMED (TAXONOMY.md) / REACHABILITY_SUPPORTED (REACHABILITY_MANUAL_VALIDATION.md) |
| v1:bermuda1023__ExposureEclipse:backend/app/services/nhc_gis.py:_download | v1 | CWE-918 | ssrf-url | ? | unclassified | live_storm_bundle → fetch_live_peak_surge → fetch_peak_surge → _fetch_ | — |
| v1:bermuda1023__ExposureEclipse:backend/app/services/nhc_gtwo.py:_download_kml | v1 | CWE-918 | ssrf-url | ? | unclassified | gtwo_endpoint → fetch_gtwo → _download_kml | — |
| v1:bermuda1023__ExposureEclipse:backend/app/services/jma_typhoon.py:_get_json | v1 | CWE-918 | ssrf-url | ? | unclassified | list_live_storms → fetch_active_typhoons → _get_json | — |
| v1:bermuda1023__ExposureEclipse:backend/scripts/build_event_evidence.py:fetch | v1 | CWE-918 | ssrf-url | ? | constant | live_storm_bundle → fetch_active_alerts → read → fetch | — |
| v1:bermuda1023__ExposureEclipse:backend/scripts/build_onlevel_indices.py:fetch | v1 | CWE-918 | ssrf-url | ? | constant | live_storm_bundle → fetch_active_alerts → read → fetch | — |
| v1:joemccann__radon:cloud/demo-vm/demo_vm_autodeploy.py:http_get | v1 | CWE-918 | ssrf-url | ? | unclassified | _call → read → _demo_read_impl → _safe_read → http_get | — |
| v1:kamoras__civitas:backend/scripts/build_county_district_crosswalk.py:download | v1 | CWE-918 | ssrf-url | ? | constant | score_breakdown → add_findings → build → download | REFUTED (manual full-source + faithful dynamic boot, DYNAMIC_PROBING.md) |
| v1:kamoras__civitas:backend/scripts/fetch_state_leg_crosswalk.py:download | v1 | CWE-918 | ssrf-url | ? | constant | score_breakdown → add_findings → build → download | REFUTED (manual full-source + faithful dynamic boot, DYNAMIC_PROBING.md) |
| v1:quantam101__profitenginev5:runtime/agent_impls/daily_digest.py:_get_json | v1 | CWE-918 | ssrf-url | ? | constant | eaos_health → status → _get_json | — |
| v1:quantam101__profitenginev5:runtime/agent_impls/daily_digest.py:_post_json | v1 | CWE-918 | ssrf-url | ? | unclassified | run_digest → main → send_telegram → _post_json | — |
| v1:saltyscott0521__tracebi:tracebi/_updates.py:_fetch | v1 | CWE-918 | ssrf-url | ? | operator-config | status → latest_release → _fetch | — |
| v1:quantam101__profitenginev5:master_distillation/dataset.py:digest | v1 | CWE-22 | path-write | ? | attacker | telegram_webhook → stable_embedding → digest | — |
| v1:trsdn__macos-notarization-broker:scripts/broker.py:write_json | v1 | CWE-22 | path-write | ? | attacker | validate_archive → command_validate → write_json | — |
| v1:2100preet__stockscanner:odte_scanner/config.py:load_config | v1 | CWE-22 | path-write | ? | operator-cli | snapshot → ensure_challenge_win_table → build_win_rate_table → load_co | — |
| v1:2100preet__stockscanner:odte_scanner/signals/sure_winners.py:save_day_store | v1 | CWE-22 | path-write | ? | unclassified | snapshot → build_sure_winner_board → save_day_store | — |
| v1:BerriAI__litellm:litellm/llms/oci/common_utils.py:load_private_key_from_file | v1 | CWE-22 | path-write | ? | constant | completion → sign_request → sign_oci_request → sign_with_manual_creden | — |
| v1:DeepCivic__womblex:src/womblex/ingest/pdf/_pdfium_doc.py:open_document | v1 | CWE-22 | path-write | ? | constant | get_files → read_parquet_footer → open → open_document | — |
| v1:Gunrunner444__BugForge:backend/app/security_testing/exploratory.py:prepare_foundry_workspace | v1 | CWE-22 | path-write | ? | constant | _execute_once → _execute → prepare_foundry_workspace | — |
| v1:PrjBbrosa__data-analyzer:scripts/probe_switch_smoothness.py:_emit | v1 | CWE-22 | path-write | ? | constant | _fail_fast_unstubbed_modal_exec → start → _forward → _emit | — |
| v1:PrjBbrosa__data-analyzer:mf4_analyzer/batch_manifest.py:sha256_file | v1 | CWE-22 | path-write | ? | constant | evaluate_staging_probe → run_native_reads → verify_package_tree → sha2 | — |
| v1:ProtossBlaster__leapmotor-mate:poller/session_share.py:_write_bytes | v1 | CWE-22 | path-write | ? | unclassified | get_charge_plan → _connect → ensure_account_cert → _write_bytes | REFUTED (stubbed boot + source: SSRF sink uses a hardcoded host) |
| v1:ProtossBlaster__leapmotor-mate:web/session_share.py:_write_bytes | v1 | CWE-22 | path-write | ? | unclassified | get_charge_plan → _connect → ensure_account_cert → _write_bytes | REFUTED (stubbed boot + source: SSRF sink uses a hardcoded host) |
| v1:VibeBB__fpga-agent:src/fpga/gates.py:spdx_ids | v1 | CWE-22 | path-write | ? | constant | test_full_gates_pass → _full → run_gates → check_provenance → spdx_ids | — |
| v1:VibeBB__fpga-agent:src/fpga/render.py:write_png | v1 | CWE-22 | path-write | ? | constant | test_gates_write_pngs_and_vcds → write_outputs → write_png | — |
| v1:bermuda1023__ExposureEclipse:backend/app/services/industry_losses.py:_rows | v1 | CWE-22 | path-write | ? | unclassified | list_industry_losses → load_actuals → _rows | — |
| v1:bramstroker__homeassistant-powercalc:utils/library/common.py:_read_model_json | v1 | CWE-22 | path-write | ? | internal | run → main → build_list → find_model_json_files → _read_model_json | — |
| v1:joemccann__radon:cloud/demo-vm/demo_vm_autodeploy.py:install_tooling | v1 | CWE-22 | path-write | ? | constant | _fetch_ib_closes → run → install_tooling | — |
| v1:kamoras__civitas:backend/scripts/audit_funding_components.py:fetch | v1 | CWE-22 | path-write | ? | constant | get_explore_document_summary → once → fetch | REFUTED (manual full-source + faithful dynamic boot, DYNAMIC_PROBING.md) |
| v1:kamoras__civitas:backend/scripts/fetch_occupation_industry.py:fetch | v1 | CWE-22 | path-write | ? | constant | get_explore_document_summary → once → fetch | REFUTED (manual full-source + faithful dynamic boot, DYNAMIC_PROBING.md) |
| v1:kamoras__civitas:backend/scripts/research_constituent_alignment.py:fetch | v1 | CWE-22 | path-write | ? | constant | get_explore_document_summary → once → fetch | REFUTED (manual full-source + faithful dynamic boot, DYNAMIC_PROBING.md) |
| v1:kamoras__civitas:backend/scripts/research_justice_loyalty.py:fetch | v1 | CWE-22 | path-write | ? | constant | get_explore_document_summary → once → fetch | REFUTED (manual full-source + faithful dynamic boot, DYNAMIC_PROBING.md) |
| v1:kamoras__civitas:backend/scripts/build_county_district_crosswalk.py:load_county_names | v1 | CWE-22 | path-write | ? | constant | score_breakdown → add_findings → build → load_county_names | REFUTED (manual full-source + faithful dynamic boot, DYNAMIC_PROBING.md) |
| v1:kamoras__civitas:backend/app/file_cache.py:read_json | v1 | CWE-22 | path-write | ? | constant | admin_dashboard → _read_system_stats → read → read_json | REFUTED (manual full-source + faithful dynamic boot, DYNAMIC_PROBING.md) |
| v1:kamoras__civitas:backend/app/shared_state.py:read_record | v1 | CWE-22 | path-write | ? | internal | admin_dashboard → _read_system_stats → api_rates → read_record | REFUTED (manual full-source + faithful dynamic boot, DYNAMIC_PROBING.md) |
| v1:kamoras__civitas:backend/app/atomic_write.py:update_json_file | v1 | CWE-22 | path-write | ? | constant | get_explore_document_summary → once → fetch → write → update_json_file | REFUTED (manual full-source + faithful dynamic boot, DYNAMIC_PROBING.md) |
| v1:quantam101__profitenginev5:master_distillation/main.py:atomic_save_text | v1 | CWE-22 | path-write | ? | operator-cli | advisor_ask → run → atomic_save_text | — |
| v1:royles__AppliedAIModelTraining:backend/app/training/dataset_loader.py:_load_jsonl | v1 | CWE-22 | path-write | ? | constant | dataset_preview → load_training_samples → _load_jsonl | — |
| v1:saltyscott0521__tracebi:tracebi/model/model_spec.py:read_model_doc | v1 | CWE-22 | path-write | ? | unclassified | publish_draft → _validate_model_draft → read_model_doc | — |
| v1:saltyscott0521__tracebi:tracebi/pipeline/pipeline_spec.py:read_pipeline_doc | v1 | CWE-22 | path-write | ? | constant | _get → get → _load → load_pipeline_spec → read_pipeline_doc | — |
| v1:tonezzz__chaba:scripts/ada/spend-ledger.py:load_usage | v1 | CWE-22 | path-write | ? | operator-cli | commit → _run → main → load_usage | — |
| v1:trsdn__macos-notarization-broker:scripts/broker.py:sha256_file | v1 | CWE-22 | path-write | ? | operator-cli | validate_archive → command_validate → sha256_file | — |
| v2:Kopanov__EdgeGuard-Knowledge-Graph:scripts/run_cypher_catalog_http.py:_write_outputs | v2 | CWE-22 | path-write | out_dir | unclassified | stix_export → export_cve → _run → main → _write_outputs | — |
| v2:artificial-hedge__dipcatcher:scripts/dipcatcher_install_smoke.py:_emit | v2 | CWE-22 | path-write | report | constant | _finish → _emit | — |
| v2:artificial-hedge__dipcatcher:scripts/dipcatcher_install_smoke.py:_prepare_inputs | v2 | CWE-22 | path-write | work | unclassified | execute → _prepare_inputs | — |
| v2:artificial-hedge__dipcatcher:scripts/fx1_model_smoke.py:_write | v2 | CWE-22 | path-write | path | constant | anthropic_batches_delete → delete → _write | — |
| v2:artificial-hedge__dipcatcher:scripts/dipcatcher_release_smoke.py:write_third_party_pth | v2 | CWE-22 | path-write | site | constant | execute → install_wheel → _try_pth → write_third_party_pth | — |
| v2:ayushdad609-code__autoclip:scripts/packaging_frame_benchmark.py:_write | v2 | CWE-22 | path-write | path | constant | save → _write | — |
| v2:Kopanov__EdgeGuard-Knowledge-Graph:demo/alert_enrichment_demo.py:load_alerts | v2 | CWE-22 | path-read | path | operator-cli | stix_export → export_cve → _run → main → load_alerts | — |
| v2:artificial-hedge__dipcatcher:scripts/dipcatcher_install_smoke.py:_sha256 | v2 | CWE-22 | path-read | path | internal | predict → generate → _sha256 | — |
| v2:artificial-hedge__dipcatcher:scripts/dipcatcher_release_smoke.py:sha256_file | v2 | CWE-22 | path-read | path | constant | _finish → append → sha256_file | — |
| v2:gyrinx-app__gyrinx:playbooks/trace/scripts/analyze_trace_detailed.py:load_and_process | v2 | CWE-22 | path-read | filepath | operator-cli | answer_invitation → deliver → run → main → load_and_process | — |
| v2:zorntyler540-byte__-estate-ez:backend/app/services/ctseller.py:jpeg_bytes_for_ctseller_post | v2 | CWE-22 | path-read | path | constant | facebook_approve_calendar → publish_posts → _photo_jpeg → jpeg_bytes_f | — |
| v2:zorntyler540-byte__-estate-ez:backend/app/services/thumbnail_image.py:thumbnail_jpeg_bytes | v2 | CWE-22 | path-read | src | constant | appraisal_check_photo_thumbnail → thumbnail_jpeg_bytes | — |
| v2:BerniceZimidzi-code__foodies-sop-rag-system:backend/database.py:execute | v2 | CWE-89 | sql | sql | unclassified | activate_account → activate_user → execute | — |
| v2:spectral-butters__hacky-movies:backend/database.py:fetch_all | v2 | CWE-89 | sql | query | unclassified | event_state → fetch_all | — |
| v2:spectral-butters__hacky-movies:backend/database.py:fetch_one | v2 | CWE-89 | sql | query | unclassified | advance_event → fetch_one | — |
| v1:Cognition-Partner-Workshops__otterworks-workshop-2610:security/deps/harness/deps_check.py:run | v1 | CWE-78 | command | ? | operator-cli | change_plan → execute → run | — |

## Not route-reachable from any HTTP route (static screen)

139 confirmed candidates. Recorded as INCONCLUSIVE on app reachability (a screen negative is not a demonstrated one), not as safe. See JSON for per-item `screen_skip`.

## Screen limitations

- call graph is name-based: generic names (execute, main, run) can create spurious or missed edges
- dynamic route registration and unresolved method dispatch are not followed
- origin classifier under-flags: it keys on request.* patterns; indirect request->value flows read as unclassified
- a 'not route-reachable' result is a STATIC negative, never a demonstrated one

## Adjudication plan

- **Level-1 source review** (route→sink dataflow) is the next step for every route-reachable item and can be done statically.
- **Level-2+ dynamic tests** (boot app, `curl --path-as-is`, in-process route/dataflow tracing) were **not performed in this session**. Candidates needing them stay INCONCLUSIVE on actual-application reachability. The Phase-2 tools named in the task brief (`reach_agent.py`, `dyn_prober_v4.py`) are **not present** in the repo.
- Prior in-repo findings reconciled: nthplusio (label differs across two v1 docs — needs reconciling), civitas & leapmotor (REFUTED, with the specific hypothesis each refuted).

## AutoClip note (integrity)

The task brief describes an AutoClip `POST /api/v1/projects/../sync-data` test returning 200 with a DB row `id='..'`. **That route does not exist in the vendored AutoClip source, and no such evidence artifact (HTTP response, server log, DB-row dump) exists in this repo.** It was not run in this session. It is therefore NOT recorded as APP_CONFIRMED. AutoClip's confirmed `ai_cover` sink and its other `data/projects/{project_id}` path-construction sites are recorded as INCONCLUSIVE on application reachability (static trace in `../corpus_v2/REACHABILITY_V2.md` found the one traced route 404s on a non-existent project id).
