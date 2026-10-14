"""Tests for ABN Lookup bulk extract XML → Parquet ingest.

All XML here is synthetic — fabricated ABNs and names shaped to the
published bulk extract schema (readme v2.6). No real register data is
committed to the repository.
"""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from womblex.ingest.abn_bulk import (
    NAMES_COLUMNS,
    RECORD_COLUMNS,
    SCHEMA_VERSION,
    discover_xml_files,
    ingest_abn_directory,
    ingest_abn_xml,
)

# An individual/sole trader with two given names, GST and a trading name.
_IND_RECORD = (
    '<ABR recordLastUpdatedDate="20260301" replaced="N">'
    '<ABN status="ACT" ABNStatusFromDate="20100401">11111111111</ABN>'
    "<EntityType><EntityTypeInd>IND</EntityTypeInd>"
    "<EntityTypeText>Individual/Sole Trader</EntityTypeText></EntityType>"
    '<LegalEntity><IndividualName type="LGL">'
    "<GivenName>JANE</GivenName><GivenName>ANNE</GivenName>"
    "<FamilyName>CITIZEN</FamilyName></IndividualName>"
    "<BusinessAddress><AddressDetails><State>NSW</State>"
    "<Postcode>2000</Postcode></AddressDetails></BusinessAddress></LegalEntity>"
    '<GST status="ACT" GSTStatusFromDate="20100401"/>'
    '<OtherEntity><NonIndividualName type="TRD">'
    "<NonIndividualNameText>JANES PLUMBING</NonIndividualNameText>"
    "</NonIndividualName></OtherEntity>"
    "</ABR>"
)

# A company with ACN, DGR fund and a registered business name.
_PRV_RECORD = (
    '<ABR recordLastUpdatedDate="20260215" replaced="N">'
    '<ABN status="ACT" ABNStatusFromDate="20050701">22222222222</ABN>'
    "<EntityType><EntityTypeInd>PRV</EntityTypeInd>"
    "<EntityTypeText>Australian Private Company</EntityTypeText></EntityType>"
    '<MainEntity><NonIndividualName type="MN">'
    "<NonIndividualNameText>EXAMPLE WIDGETS PTY LTD</NonIndividualNameText>"
    "</NonIndividualName>"
    "<BusinessAddress><AddressDetails><State>VIC</State>"
    "<Postcode>3000</Postcode></AddressDetails></BusinessAddress></MainEntity>"
    '<ASICNumber ASICNumberType="undetermined">000000019</ASICNumber>'
    '<GST status="ACT" GSTStatusFromDate="20050701"/>'
    '<DGR DGRStatusFromDate="20100101" status="ACT">'
    '<NonIndividualName type="DGR">'
    "<NonIndividualNameText>EXAMPLE WIDGETS RELIEF FUND</NonIndividualNameText>"
    "</NonIndividualName></DGR>"
    '<OtherEntity><NonIndividualName type="BN">'
    "<NonIndividualNameText>WIDGETS DIRECT</NonIndividualNameText>"
    "</NonIndividualName></OtherEntity>"
    "</ABR>"
)

# A cancelled trust with no GST, ASIC, names or address postcode.
_TRT_RECORD = (
    '<ABR recordLastUpdatedDate="20190501" replaced="Y">'
    '<ABN status="CAN" ABNStatusFromDate="20190501">33333333333</ABN>'
    "<EntityType><EntityTypeInd>DIT</EntityTypeInd>"
    "<EntityTypeText>Discretionary Investment Trust</EntityTypeText></EntityType>"
    '<MainEntity><NonIndividualName type="MN">'
    "<NonIndividualNameText>The Trustee for Example Family Trust</NonIndividualNameText>"
    "</NonIndividualName>"
    "<BusinessAddress><AddressDetails><State>QLD</State>"
    "<Postcode>4000</Postcode></AddressDetails></BusinessAddress></MainEntity>"
    "</ABR>"
)


def _write_extract(path: Path, records: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<Transfer error="false"><TransferInfo></TransferInfo>'
        f"{records}</Transfer>",
        encoding="utf-8",
    )


# ── Single file ingest ──────────────────────────────────────────────────────


def test_ingest_individual_record(tmp_path: Path):
    xml = tmp_path / "20260601Public01.xml"
    _write_extract(xml, _IND_RECORD)

    result = ingest_abn_xml(xml, tmp_path / "out")
    assert result is not None
    assert result.record_count == 1
    assert result.name_count == 2  # legal name + trading name

    table = pq.read_table(str(result.records_path))
    assert table.column_names == RECORD_COLUMNS
    row = {k: v[0] for k, v in table.to_pydict().items()}
    assert row["abn"] == "11111111111"
    assert row["abn_status"] == "ACT"
    assert row["abn_status_from_date"] == "20100401"
    assert row["record_last_updated_date"] == "20260301"
    assert row["replaced"] == "N"
    assert row["entity_type_ind"] == "IND"
    assert row["name_type"] == "LGL"
    assert row["given_name_1"] == "JANE"
    assert row["given_name_2"] == "ANNE"
    assert row["family_name"] == "CITIZEN"
    assert row["non_individual_name"] == ""  # individuals have no main name
    assert row["state"] == "NSW"
    assert row["postcode"] == "2000"
    assert row["gst_status"] == "ACT"

    names = pq.read_table(str(result.names_path)).to_pydict()
    assert names["name_type"] == ["LGL", "TRD"]
    assert names["name_text"] == ["JANE ANNE CITIZEN", "JANES PLUMBING"]


def test_ingest_company_record(tmp_path: Path):
    xml = tmp_path / "20260601Public02.xml"
    _write_extract(xml, _PRV_RECORD)

    result = ingest_abn_xml(xml, tmp_path / "out")
    assert result is not None

    row = {k: v[0] for k, v in pq.read_table(str(result.records_path)).to_pydict().items()}
    assert row["abn"] == "22222222222"
    assert row["non_individual_name"] == "EXAMPLE WIDGETS PTY LTD"
    assert row["name_type"] == "MN"
    assert row["given_name_1"] == ""  # main entities have no individual name parts
    assert row["asic_number"] == "000000019"
    assert row["asic_number_type"] == "undetermined"

    names = pq.read_table(str(result.names_path)).to_pydict()
    assert names["name_type"] == ["MN", "DGR", "BN"]
    assert names["name_text"] == [
        "EXAMPLE WIDGETS PTY LTD",
        "EXAMPLE WIDGETS RELIEF FUND",
        "WIDGETS DIRECT",
    ]
    # DGR rows carry the fund's status-from date; others are empty.
    assert names["status_from_date"] == ["", "20100101", ""]


def test_absent_optionals_are_empty_strings(tmp_path: Path):
    """Missing GST / ASIC / OtherEntity become "" — never null."""
    xml = tmp_path / "20260601Public03.xml"
    _write_extract(xml, _TRT_RECORD)

    result = ingest_abn_xml(xml, tmp_path / "out")
    assert result is not None

    row = {k: v[0] for k, v in pq.read_table(str(result.records_path)).to_pydict().items()}
    assert row["abn_status"] == "CAN"
    assert row["replaced"] == "Y"
    assert row["gst_status"] == ""
    assert row["gst_status_from_date"] == ""
    assert row["asic_number"] == ""
    assert all(v is not None for v in row.values())


def test_spaced_given_name_kept_distinct(tmp_path: Path):
    """A single given name containing a space is distinguishable from two."""
    record = (
        '<ABR recordLastUpdatedDate="20260301" replaced="N">'
        '<ABN status="ACT" ABNStatusFromDate="20100401">44444444444</ABN>'
        "<EntityType><EntityTypeInd>IND</EntityTypeInd>"
        "<EntityTypeText>Individual/Sole Trader</EntityTypeText></EntityType>"
        '<LegalEntity><IndividualName type="LGL">'
        "<GivenName>ANSHPREET SINGH</GivenName><FamilyName>EXAMPLE</FamilyName>"
        "</IndividualName><BusinessAddress><AddressDetails>"
        "<State>VIC</State><Postcode>3024</Postcode>"
        "</AddressDetails></BusinessAddress></LegalEntity></ABR>"
    )
    xml = tmp_path / "20260601Public09.xml"
    _write_extract(xml, record)

    result = ingest_abn_xml(xml, tmp_path / "out")
    assert result is not None
    row = {k: v[0] for k, v in pq.read_table(str(result.records_path)).to_pydict().items()}
    assert row["given_name_1"] == "ANSHPREET SINGH"
    assert row["given_name_2"] == ""


def test_provenance_metadata(tmp_path: Path):
    xml = tmp_path / "20260601Public04.xml"
    _write_extract(xml, _IND_RECORD + _PRV_RECORD)

    result = ingest_abn_xml(xml, tmp_path / "out", compute_md5=True)
    assert result is not None
    assert result.record_count == 2

    meta = pq.read_table(str(result.records_path)).schema.metadata
    assert meta[b"abn.schema_version"] == SCHEMA_VERSION.encode()
    assert meta[b"abn.source_file"] == b"20260601Public04.xml"
    assert b"abn.source_md5" in meta

    # Row counts are appended at writer close, so they live in the parquet
    # footer key-value metadata rather than the Arrow schema metadata.
    footer = pq.ParquetFile(str(result.records_path)).metadata.metadata
    assert footer[b"abn.row_count"] == b"2"
    names_footer = pq.ParquetFile(str(result.names_path)).metadata.metadata
    assert names_footer[b"abn.row_count"] == b"5"


def test_no_md5(tmp_path: Path):
    xml = tmp_path / "20260601Public05.xml"
    _write_extract(xml, _IND_RECORD)

    result = ingest_abn_xml(xml, tmp_path / "out", compute_md5=False)
    assert result is not None
    meta = pq.read_table(str(result.records_path)).schema.metadata
    assert b"abn.source_md5" not in meta


def test_empty_extract_writes_empty_parquets(tmp_path: Path):
    """A Transfer with zero ABR records still yields readable output."""
    xml = tmp_path / "20260601Public06.xml"
    _write_extract(xml, "")

    result = ingest_abn_xml(xml, tmp_path / "out")
    assert result is not None
    assert result.record_count == 0
    table = pq.read_table(str(result.records_path))
    assert table.num_rows == 0
    assert table.column_names == RECORD_COLUMNS
    assert pq.read_table(str(result.names_path)).column_names == NAMES_COLUMNS


def test_non_transfer_root_skipped(tmp_path: Path):
    xml = tmp_path / "other.xml"
    xml.write_text("<NotABulkExtract><Thing/></NotABulkExtract>", encoding="utf-8")

    result = ingest_abn_xml(xml, tmp_path / "out")
    assert result is None
    assert not (tmp_path / "out" / "other.parquet").exists()


def test_malformed_xml_returns_none(tmp_path: Path):
    xml = tmp_path / "20260601Public07.xml"
    xml.write_text('<Transfer error="false">' + _IND_RECORD + "<ABR>", encoding="utf-8")

    result = ingest_abn_xml(xml, tmp_path / "out")
    assert result is None
    # Partial output cleaned up.
    assert not (tmp_path / "out" / "20260601Public07.parquet").exists()
    assert not (tmp_path / "out" / "20260601Public07_names.parquet").exists()


# ── Directory ingest ────────────────────────────────────────────────────────


def test_discover_xml_files(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    _write_extract(tmp_path / "a.xml", "")
    _write_extract(tmp_path / "sub" / "b.xml", "")
    (tmp_path / "c.txt").write_text("not xml")

    found = discover_xml_files(tmp_path)
    assert len(found) == 2
    assert all(p.suffix == ".xml" for p in found)


def test_ingest_abn_directory(tmp_path: Path):
    src = tmp_path / "extract"
    _write_extract(src / "20260601Public01.xml", _IND_RECORD)
    _write_extract(src / "20260601Public02.xml", _PRV_RECORD + _TRT_RECORD)
    # Skipped: not a bulk extract.
    (src / "readme.xml").write_text("<readme/>", encoding="utf-8")

    results = ingest_abn_directory(src, tmp_path / "out")
    assert len(results) == 2
    assert sum(r.record_count for r in results) == 3
    assert (tmp_path / "out" / "20260601Public01.parquet").exists()
    assert (tmp_path / "out" / "20260601Public02_names.parquet").exists()


def test_directory_continues_past_failing_file(tmp_path: Path):
    """One broken file must not stop the batch (per-file failure isolation)."""
    src = tmp_path / "extract"
    _write_extract(src / "20260601Public01.xml", _IND_RECORD)
    # Truncated mid-record: parse fails partway through.
    (src / "20260601Public02.xml").write_text(
        '<Transfer error="false">' + _PRV_RECORD + "<ABR>", encoding="utf-8",
    )
    _write_extract(src / "20260601Public03.xml", _TRT_RECORD)

    results = ingest_abn_directory(src, tmp_path / "out")
    assert len(results) == 2
    assert {r.records_path.name for r in results} == {
        "20260601Public01.parquet", "20260601Public03.parquet",
    }
    # Failed file's partial output cleaned up.
    assert not (tmp_path / "out" / "20260601Public02.parquet").exists()
    assert not (tmp_path / "out" / "20260601Public02_names.parquet").exists()


def test_ingest_empty_directory(tmp_path: Path):
    src = tmp_path / "empty"
    src.mkdir()
    results = ingest_abn_directory(src, tmp_path / "out")
    assert results == []


# ── Scale smoke ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_records", [120_000])
def test_batched_writes_round_trip(tmp_path: Path, n_records: int):
    """More records than one write batch still round-trips losslessly."""
    xml = tmp_path / "20260601Public08.xml"
    record = (
        '<ABR recordLastUpdatedDate="20260301" replaced="N">'
        '<ABN status="ACT" ABNStatusFromDate="20100401">{abn}</ABN>'
        "<EntityType><EntityTypeInd>IND</EntityTypeInd>"
        "<EntityTypeText>Individual/Sole Trader</EntityTypeText></EntityType>"
        '<LegalEntity><IndividualName type="LGL">'
        "<GivenName>TEST</GivenName><FamilyName>PERSON{i}</FamilyName>"
        "</IndividualName><BusinessAddress><AddressDetails>"
        "<State>SA</State><Postcode>5000</Postcode>"
        "</AddressDetails></BusinessAddress></LegalEntity></ABR>"
    )
    body = "".join(
        record.format(abn=str(10000000000 + i), i=i) for i in range(n_records)
    )
    _write_extract(xml, body)

    result = ingest_abn_xml(xml, tmp_path / "out", compute_md5=False)
    assert result is not None
    assert result.record_count == n_records

    table = pq.read_table(str(result.records_path))
    assert table.num_rows == n_records
    abns = table.column("abn").to_pylist()
    assert abns[0] == "10000000000"
    assert abns[-1] == str(10000000000 + n_records - 1)
