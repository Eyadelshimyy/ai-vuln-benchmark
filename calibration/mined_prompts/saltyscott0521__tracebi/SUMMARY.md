# Mined prompts: saltyscott0521__tracebi

105 on-topic candidate function(s) found across 124 files.
50 selected (spread across CWEs, capped at --limit 50).

- CWE-22: 17 selected (of 59 total candidates)
- CWE-502: 3 selected (of 3 total candidates)
- CWE-78: 12 selected (of 12 total candidates)
- CWE-89: 16 selected (of 29 total candidates)
- CWE-918: 3 selected (of 3 total candidates)

## Selected functions

- **serve** (CWE-22, CWE-78) -- `scripts/ui_audit.py:69`
- **read_connection_doc** (CWE-502) -- `tracebi/connections.py:81`
- **_tracebi** (CWE-78) -- `evals/agent/score.py:95`
- **list_tables** (CWE-89) -- `tracebi/connectors/duckdb_connector.py:78`
- **slack_notify** (CWE-918) -- `tracebi/_delivery.py:213`
- **main** (CWE-22) -- `site/build_docs.py:240`
- **read_model_doc** (CWE-502) -- `tracebi/model/model_spec.py:89`
- **cmd_update** (CWE-78) -- `tracebi/cli.py:837`
- **column_schema** (CWE-89) -- `tracebi/connectors/duckdb_connector.py:89`
- **_slack_request** (CWE-918) -- `tracebi/_delivery.py:404`
- **_write** (CWE-22) -- `tracebi/_gateway_log.py:68`
- **read_pipeline_doc** (CWE-502) -- `tracebi/pipeline/pipeline_spec.py:33`
- **serve** (CWE-78) -- `tracebi/mcp_server.py:2090`
- **load** (CWE-89) -- `tracebi/connectors/duckdb_connector.py:209`
- **_fetch** (CWE-918) -- `tracebi/_updates.py:88`
- **read_lines** (CWE-22) -- `tracebi/_gateway_log.py:173`
- **start** (CWE-78) -- `tracebi/pipeline/runner.py:372`
- **_check_append_decimal_types** (CWE-89) -- `tracebi/connectors/duckdb_connector.py:329`
- **_read_full_feed** (CWE-22) -- `tracebi/_session_export.py:55`
- **_current_git_sha** (CWE-78) -- `tracebi/reports/report.py:506`
- **_write_into** (CWE-89) -- `tracebi/connectors/duckdb_connector.py:359`
- **_export_markdown** (CWE-22) -- `tracebi/_session_export.py:302`
- **_refresh** (CWE-78) -- `tracebi/schedule.py:822`
- **load** (CWE-89) -- `tracebi/connectors/snowflake_connector.py:90`
- **latest_release** (CWE-22) -- `tracebi/_updates.py:105`
- **_mcp_sessions** (CWE-78) -- `tracebi/web/api/main.py:94`
- **reconcile** (CWE-89) -- `tracebi/contracts.py:159`
- **_cached_release** (CWE-22) -- `tracebi/_updates.py:171`
- **run_layer** (CWE-78) -- `tracebi/web/api/routers/pipelines.py:117`
- **_insert_returning_id** (CWE-89) -- `tracebi/pipeline/runner.py:110`
- **cmd_init** (CWE-22) -- `tracebi/cli.py:574`
- **seed_and_run** (CWE-78) -- `tracebi/web/demo_app/pipeline.py:167`
- **_add_missing_run_columns** (CWE-89) -- `tracebi/pipeline/runner.py:147`
- **_dev_in_the_app** (CWE-22) -- `tracebi/cli.py:1024`
- **seed_and_run** (CWE-78) -- `tracebi/web/demo_app/wealth_pipeline.py:75`
- **run_history** (CWE-89) -- `tracebi/pipeline/runner.py:469`
- **cmd_new_model** (CWE-22) -- `tracebi/cli.py:1236`
- **main** (CWE-78) -- `tracebi/web/run.py:31`
- **_execute_locked** (CWE-89) -- `tracebi/pipeline/runner.py:609`
- **cmd_new_transform** (CWE-22) -- `tracebi/cli.py:1271`
- **_latest_run_id** (CWE-89) -- `tracebi/pipeline/runner.py:642`
- **cmd_new_pipeline** (CWE-22) -- `tracebi/cli.py:1341`
- **_write_run_end** (CWE-89) -- `tracebi/pipeline/runner.py:677`
- **_migrate_model** (CWE-22) -- `tracebi/cli.py:1863`
- **stamp** (CWE-89) -- `tracebi/reports/embed.py:254`
- **_migrate_declarative** (CWE-22) -- `tracebi/cli.py:1902`
- **build** (CWE-89) -- `tracebi/spec.py:602`
- **_build_report_target** (CWE-22) -- `tracebi/cli.py:2505`
- **resolve** (CWE-89) -- `tracebi/spec.py:617`
- **_serve_file** (CWE-22) -- `tracebi/cli.py:2586`