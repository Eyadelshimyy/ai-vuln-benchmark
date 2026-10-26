#!/usr/bin/env python3
# storage_cleanup_daily.py - Daily orphaned S3 object cleanup
# Originally Python 2.7, minimally ported to Python 3 in 2021
# Lists S3 objects, compares with DynamoDB metadata, quarantines orphans,
# generates storage savings report
#
# Owner: Jake (data-team@otterworks.dev) -- Jake left mid-2020
# TODO ETL-091: Add S3 lifecycle rules instead of manual cleanup (2019-11-20)
# TODO ETL-156: Parallelize S3 listing for large buckets (deferred Q1 2020)
# TODO ETL-203: Add dry-run mode for testing (never implemented)

import configparser
import json
import sys
from datetime import datetime, timezone

import boto3


def main():
    print("[%s] storage_cleanup_daily.py starting..." % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ---- Load config ----
    config = configparser.ConfigParser()
    config.read("/opt/etl/config.ini")

    aws_access_key = config.get("aws", "access_key")
    aws_secret_key = config.get("aws", "secret_key")
    aws_region = config.get("aws", "region")

    file_storage_bucket = config.get("s3", "file_storage_bucket")
    quarantine_bucket = config.get("s3", "quarantine_bucket")
    data_lake_bucket = config.get("s3", "data_lake_bucket")

    files_prefix = "files/"
    quarantine_prefix = "quarantined"
    dynamodb_table_name = "otterworks-file-metadata"

    ds = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    # ---- List all S3 objects ----
    print("[%s] Listing objects in s3://%s/%s" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file_storage_bucket, files_prefix
    ))

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region,
    )

    all_objects = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=file_storage_bucket, Prefix=files_prefix):
        for obj in page.get("Contents", []):
            all_objects.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
            })

    total_objects = len(all_objects)
    total_size_bytes = sum(o["size"] for o in all_objects)

    print("[%s] Found %d objects in S3 (%d bytes total)" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), total_objects, total_size_bytes
    ))

    # ---- List metadata references from DynamoDB ----
    print("[%s] Scanning DynamoDB table %s for metadata references..." % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), dynamodb_table_name
    ))

    dynamodb = boto3.resource(
        "dynamodb",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region,
    )
    table = dynamodb.Table(dynamodb_table_name)

    referenced_keys = set()
    scan_kwargs = {
        "ProjectionExpression": "s3_key",
    }

    while True:
        response = table.scan(**scan_kwargs)
        for item in response.get("Items", []):
            s3_key = item.get("s3_key", "")
            if s3_key:
                referenced_keys.add(s3_key)

        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        scan_kwargs["ExclusiveStartKey"] = last_key

    print("[%s] Found %d S3 keys referenced in metadata" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(referenced_keys)
    ))

    # ---- Find orphaned objects ----
    orphaned = []
    orphaned_bytes = 0

    for obj in all_objects:
        if obj["key"] not in referenced_keys:
            orphaned.append(obj)
            orphaned_bytes += obj["size"]

    orphaned_count = len(orphaned)

    print("[%s] Found %d orphaned objects (%.2f MB)" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        orphaned_count,
        orphaned_bytes / (1024 * 1024),
    ))

    if orphaned_count == 0:
        print("[%s] No orphaned objects to quarantine" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # Still generate report even with 0 orphans
    else:
        # ---- Move orphaned objects to quarantine ----
        print("[%s] Moving %d orphaned objects to quarantine..." % (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), orphaned_count
        ))

    moved_count = 0
    failed_count = 0

    for obj in orphaned:
        source_key = obj["key"]
        dest_key = "%s/%s/%s" % (quarantine_prefix, ds, source_key)

        try:
            s3_client.copy_object(
                Bucket=quarantine_bucket,
                Key=dest_key,
                CopySource={"Bucket": file_storage_bucket, "Key": source_key},
                MetadataDirective="COPY",
            )
            s3_client.delete_object(Bucket=file_storage_bucket, Key=source_key)
            moved_count += 1
        except Exception as e:
            print("[%s] WARNING: Failed to quarantine %s: %s" % (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), source_key, str(e)
            ))
            failed_count += 1

    if orphaned_count > 0:
        print("[%s] Quarantined %d objects (%d failed) to s3://%s/%s/%s/" % (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            moved_count, failed_count,
            quarantine_bucket, quarantine_prefix, ds,
        ))

    # ---- Generate storage cleanup report ----
    savings_gb = orphaned_bytes / (1024 ** 3)
    estimated_monthly_savings = round(savings_gb * 0.023, 4)

    report = {
        "report_type": "storage_cleanup",
        "report_date": ds,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "inventory": {
            "total_objects": total_objects,
            "total_size_bytes": total_size_bytes,
            "total_size_gb": round(total_size_bytes / (1024 ** 3), 4),
        },
        "orphans": {
            "orphaned_objects": orphaned_count,
            "orphaned_bytes": orphaned_bytes,
            "orphaned_size_gb": round(savings_gb, 4),
            "orphan_percentage": round(
                (orphaned_count / total_objects * 100) if total_objects else 0, 2
            ),
        },
        "cleanup": {
            "objects_quarantined": moved_count,
            "objects_failed": failed_count,
            "quarantine_bucket": quarantine_bucket,
        },
        "savings": {
            "storage_freed_gb": round(savings_gb, 4),
            "estimated_monthly_savings_usd": estimated_monthly_savings,
        },
    }

    report_key = "reports/storage-cleanup/%s/report.json" % ds
    s3_client_report = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region,
    )
    s3_client_report.put_object(
        Bucket=data_lake_bucket,
        Key=report_key,
        Body=json.dumps(report, indent=2).encode("utf-8"),
    )

    print("[%s] Storage cleanup report: %d orphans quarantined, %.4f GB freed, ~$%.4f/month saved" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        moved_count, savings_gb, estimated_monthly_savings,
    ))
    print("[%s] storage_cleanup_daily.py completed successfully" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[%s] FATAL: %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(e)))
        sys.exit(1)
