#!/usr/bin/env python3
# audit_archive_weekly.py - Weekly audit event archival to S3 Glacier
# Originally Python 2.7, minimally ported to Python 3 in 2021
# Scans DynamoDB for old events, compresses to JSONL.gz, uploads to Glacier,
# batch-deletes from DynamoDB, generates compliance report
#
# Owner: Jake (data-team@otterworks.dev) -- Jake left mid-2020
# TODO ETL-134: Add incremental archival instead of full scan (deferred Q2 2020)
# TODO ETL-167: Handle DynamoDB throughput throttling properly (2020-04-10)
# TODO ETL-199: This script has no tests whatsoever (never prioritized)

import configparser
import gzip
import io
import json
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import boto3


class DecimalEncoder(json.JSONEncoder):
    """Handle DynamoDB Decimal types -- copied from StackOverflow"""
    def default(self, o):
        if isinstance(o, Decimal):
            if o == int(o):
                return int(o)
            return float(o)
        return super().default(o)


def main():
    print("[%s] audit_archive_weekly.py starting..." % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ---- Load config ----
    config = configparser.ConfigParser()
    config.read("/opt/etl/config.ini")

    aws_access_key = config.get("aws", "access_key")
    aws_secret_key = config.get("aws", "secret_key")
    aws_region = config.get("aws", "region")

    archive_bucket = config.get("s3", "archive_bucket")

    ds = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    retention_days = 90
    dynamodb_table_name = "otterworks-audit-events"
    dynamodb_batch_size = 25  # DynamoDB batch write limit
    s3_prefix = "audit-archive"

    cutoff_date = (
        datetime.strptime(ds, "%Y-%m-%d") - timedelta(days=retention_days)
    ).isoformat() + "Z"

    print("[%s] Archiving events older than %d days (cutoff: %s)" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), retention_days, cutoff_date
    ))

    # ---- Scan DynamoDB for old audit events ----
    dynamodb = boto3.resource(
        "dynamodb",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region,
    )
    table = dynamodb.Table(dynamodb_table_name)

    events_to_archive = []
    scan_kwargs = {
        "FilterExpression": "#ts < :cutoff",
        "ExpressionAttributeNames": {"#ts": "timestamp"},
        "ExpressionAttributeValues": {":cutoff": cutoff_date},
    }

    while True:
        response = table.scan(**scan_kwargs)
        items = response.get("Items", [])
        events_to_archive.extend(items)

        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        scan_kwargs["ExclusiveStartKey"] = last_key

    archive_count = len(events_to_archive)
    print("[%s] Found %d audit events older than %d days" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), archive_count, retention_days
    ))

    if archive_count == 0:
        print("[%s] No events to archive, exiting" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        sys.exit(0)

    # ---- Compress to JSONL.gz ----
    archive_key = "%s/year=%s/week=%s/audit_events.jsonl.gz" % (s3_prefix, ds[:4], ds)

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for event in events_to_archive:
            line = json.dumps(event, cls=DecimalEncoder)
            gz.write(line.encode("utf-8"))
            gz.write(b"\n")

    compressed_size = buf.tell()
    print("[%s] Compressed %d events to %.2f MB" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        archive_count,
        compressed_size / (1024 * 1024),
    ))

    # ---- Upload to S3 with Glacier storage class ----
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region,
    )

    s3_client.put_object(
        Bucket=archive_bucket,
        Key=archive_key,
        Body=buf.getvalue(),
        StorageClass="GLACIER",
    )

    print("[%s] Archived to s3://%s/%s (GLACIER)" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), archive_bucket, archive_key
    ))

    # ---- Batch-delete archived records from DynamoDB ----
    print("[%s] Deleting %d archived events from DynamoDB..." % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), archive_count
    ))

    deleted_count = 0
    batch = []

    for event in events_to_archive:
        key = {
            "event_id": event["event_id"],
            "timestamp": event["timestamp"],
        }
        batch.append(key)

        if len(batch) >= dynamodb_batch_size:
            try:
                with table.batch_writer() as batch_writer:
                    for k in batch:
                        batch_writer.delete_item(Key=k)
                deleted_count += len(batch)
            except:
                # TODO ETL-167: Handle throttling / partial failures
                pass
            batch = []

    # flush remaining batch
    if batch:
        try:
            with table.batch_writer() as batch_writer:
                for k in batch:
                    batch_writer.delete_item(Key=k)
            deleted_count += len(batch)
        except:
            pass

    print("[%s] Deleted %d events from DynamoDB" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), deleted_count
    ))

    # ---- Generate compliance report ----
    report = {
        "report_type": "audit_archive_compliance",
        "execution_date": ds,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "retention_policy": {
            "retention_days": retention_days,
            "cutoff_date": cutoff_date,
        },
        "results": {
            "events_scanned": archive_count,
            "events_archived": archive_count,
            "events_deleted_from_source": deleted_count,
            "archive_location": "s3://%s/%s" % (archive_bucket, archive_key),
            "archive_storage_class": "GLACIER",
            "compressed_size_bytes": compressed_size,
        },
        "compliance": {
            "gdpr_compliant": True,
            "soc2_compliant": True,
            "data_encrypted_at_rest": True,
            "data_encrypted_in_transit": True,
        },
    }

    report_key = "reports/compliance/audit-archive/%s/report.json" % ds
    s3_client_report = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region,
    )
    s3_client_report.put_object(
        Bucket=archive_bucket,
        Key=report_key,
        Body=json.dumps(report, indent=2).encode("utf-8"),
    )

    print("[%s] Compliance report: %d archived, %d deleted, stored at s3://%s/%s" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        archive_count,
        deleted_count,
        archive_bucket,
        report_key,
    ))
    print("[%s] audit_archive_weekly.py completed successfully" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[%s] FATAL: %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(e)))
        sys.exit(1)
