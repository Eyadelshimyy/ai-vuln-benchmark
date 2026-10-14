# Mined prompts: airbytehq__airbyte

44 on-topic candidate function(s) found across 2589 files.
44 selected (spread across CWEs, capped at --limit 50).

- CWE-22: 20 selected (of 20 total candidates)
- CWE-78: 8 selected (of 8 total candidates)
- CWE-89: 8 selected (of 8 total candidates)
- CWE-918: 8 selected (of 8 total candidates)

## Selected functions

- **test_check_fails** (CWE-22) -- `airbyte-integrations/connectors/destination-glassflow/integration_tests/integration_test.py:63`
- **test_given_425_naming_report_that_differs_from_request_then_waits_and_requests_new_report** (CWE-78) -- `airbyte-integrations/connectors/source-amazon-ads/unit_tests/integrations/test_report_streams.py:326`
- **test_write** (CWE-89) -- `airbyte-integrations/connectors/destination-duckdb/integration_tests/integration_test.py:186`
- **_fetch_rdt_token** (CWE-918) -- `airbyte-integrations/connectors/source-amazon-seller-partner/components.py:164`
- **test_check_succeeds** (CWE-22) -- `airbyte-integrations/connectors/destination-glassflow/integration_tests/integration_test.py:73`
- **test_given_425_without_report_id_then_waits_and_requests_report_again** (CWE-78) -- `airbyte-integrations/connectors/source-amazon-ads/unit_tests/integrations/test_report_streams.py:388`
- **test_large_number_of_writes** (CWE-89) -- `airbyte-integrations/connectors/destination-duckdb/integration_tests/integration_test.py:289`
- **_fetch_failure_reason** (CWE-918) -- `airbyte-integrations/connectors/source-amazon-seller-partner/components.py:1036`
- **test_write** (CWE-22) -- `airbyte-integrations/connectors/destination-glassflow/integration_tests/integration_test.py:83`
- **test_given_425_without_report_id_on_report_creation_then_fails_with_transient_error** (CWE-78) -- `airbyte-integrations/connectors/source-amazon-ads/unit_tests/integrations/test_report_streams.py:420`
- **test_serialize_object_columns_prevents_empty_struct_error** (CWE-89) -- `airbyte-integrations/connectors/destination-motherduck/unit_tests/test_serialize_object_columns.py:52`
- **check_connection** (CWE-918) -- `airbyte-integrations/connectors/source-appsflyer/source_appsflyer/source.py:372`
- **decode** (CWE-22) -- `airbyte-integrations/connectors/source-bing-ads/components.py:497`
- **call** (CWE-78) -- `airbyte-integrations/connectors/source-facebook-marketing/source_facebook_marketing/api.py:214`
- **test_serialize_object_columns_normalized_column_names** (CWE-89) -- `airbyte-integrations/connectors/destination-motherduck/unit_tests/test_serialize_object_columns.py:74`
- **_do_request** (CWE-918) -- `airbyte-integrations/connectors/source-box-data-extract/source_box_data_extract/box_api.py:68`
- **_load_page_property_list** (CWE-22) -- `airbyte-integrations/connectors/source-facebook-pages/unit_tests/test_manifest_page_fields.py:23`
- **load_next_page** (CWE-78) -- `airbyte-integrations/connectors/source-facebook-marketing/source_facebook_marketing/streams/patches.py:36`
- **check** (CWE-89) -- `airbyte-integrations/connectors/destination-sqlite/destination_sqlite/destination.py:119`
- **_resolve_cloud_id** (CWE-918) -- `airbyte-integrations/connectors/source-jira/components.py:132`
- **open_file** (CWE-22) -- `airbyte-integrations/connectors/source-gcs/source_gcs/stream_reader.py:145`
- **test_call_quota_block_budget_spans_backoff_retries** (CWE-78) -- `airbyte-integrations/connectors/source-facebook-marketing/unit_tests/test_api.py:431`
- **test_api_budget_weighs_each_call_by_its_gmail_quota_cost** (CWE-89) -- `airbyte-integrations/connectors/source-gmail/unit_tests/test_streams.py:68`
- **_execute_batch_datetime_request** (CWE-918) -- `airbyte-integrations/connectors/source-microsoft-dataverse/source_microsoft_dataverse/dataverse.py:113`
- **_open_gzip_encoded_blob** (CWE-22) -- `airbyte-integrations/connectors/source-gcs/source_gcs/stream_reader.py:182`
- **test_ad_creatives_from_ads_repeated_failed_creative_remains_a_failure** (CWE-78) -- `airbyte-integrations/connectors/source-facebook-marketing/unit_tests/test_streams.py:597`
- **test_api_budget_resolves_separate_campaign_and_report_policies** (CWE-89) -- `airbyte-integrations/connectors/source-reddit-ads/unit_tests/test_manifest_config.py:19`
- **_migrate_legacy_token** (CWE-918) -- `airbyte-integrations/connectors/source-monday/components.py:102`
- **test_filename_collision_shared_tmp_dir** (CWE-22) -- `airbyte-integrations/connectors/source-gcs/unit_tests/test_zip_helper.py:83`
- **connector_setup_fixture** (CWE-78) -- `airbyte-integrations/connectors/source-gcs/integration_tests/conftest.py:63`
- **test_bulk_query_prepare** (CWE-89) -- `airbyte-integrations/connectors/source-shopify/unit_tests/graphql_bulk/test_query.py:40`
- **_create_backoff_exception** (CWE-918) -- `airbyte-integrations/connectors/source-salesforce/unit_tests/test_rate_limiting.py:524`
- **test_issue_timeline_events_collapse_a_page_into_one_record** (CWE-22) -- `airbyte-integrations/connectors/source-github/unit_tests/test_manifest_parent_child_streams.py:675`
- **test_contributor_activity_reads_the_response_fixture** (CWE-22) -- `airbyte-integrations/connectors/source-github/unit_tests/test_manifest_special_incremental_streams.py:303`
- **test_documented_v3_examples_validate_against_stream_schemas** (CWE-22) -- `airbyte-integrations/connectors/source-greenhouse/unit_tests/test_pagination.py:1038`
- **test_scorecard_question_answers_value_is_always_a_string** (CWE-22) -- `airbyte-integrations/connectors/source-greenhouse/unit_tests/test_pagination.py:1497`
- **_load_migration_template** (CWE-22) -- `airbyte-integrations/connectors/source-pinterest/unit_tests/test_config_migrations.py:24`
- **test_manifest_has_config_normalization_rules** (CWE-22) -- `airbyte-integrations/connectors/source-pinterest/unit_tests/test_config_migrations.py:281`
- **test_manifest_has_window_days_migration** (CWE-22) -- `airbyte-integrations/connectors/source-pinterest/unit_tests/test_config_migrations.py:289`
- **test_spec_default_values_are_integers** (CWE-22) -- `airbyte-integrations/connectors/source-pinterest/unit_tests/test_config_migrations.py:304`
- **_iter_run** (CWE-22) -- `airbyte-integrations/connectors/source-shopify/source_shopify/shopify_graphql/bulk/external_sort.py:115`
- **test_51004_retry_handlers_also_retry_51002** (CWE-22) -- `airbyte-integrations/connectors/source-tiktok-marketing/unit_tests/test_source.py:136`
- **configure_catalog** (CWE-22) -- `tools/schema_generator/schema_generator/configure_catalog.py:11`
- **infer_schemas** (CWE-22) -- `tools/schema_generator/schema_generator/infer_schemas.py:62`