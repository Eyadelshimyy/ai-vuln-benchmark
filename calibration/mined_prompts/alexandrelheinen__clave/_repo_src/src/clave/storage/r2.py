"""Cloudflare R2 (S3-compatible) storage client using pure Python AWS SigV4.

Implements zero-dependency object storage operations against Cloudflare R2
adhering to AC-DATA-02 and AC-DATA-03.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from clave.errors import ClaveError
from clave.storage.config import R2Config


class R2Error(ClaveError):
    """An operation against Cloudflare R2 object storage failed."""


def _sign(key: bytes, msg: str) -> bytes:
    """HMAC-SHA256 signature helper."""
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signature_key(
    key: str,
    date_stamp: str,
    region_name: str,
    service_name: str,
) -> bytes:
    """Derive AWS SigV4 signing key."""
    k_date = _sign(b"AWS4" + key.encode("utf-8"), date_stamp)
    k_region = _sign(k_date, region_name)
    k_service = _sign(k_region, service_name)
    return _sign(k_service, "aws4_request")


class R2Client:
    """Client for Cloudflare R2 using standard library AWS SigV4."""

    def __init__(self, config: R2Config) -> None:
        self.config = config
        parsed = urllib.parse.urlparse(config.endpoint_url)
        self.host = parsed.netloc
        self.scheme = parsed.scheme or "https"

    def _build_request(
        self,
        method: str,
        path: str,
        query_params: dict[str, str] | None = None,
        payload: bytes = b"",
        extra_headers: dict[str, str] | None = None,
    ) -> urllib.request.Request:
        """Construct and sign an HTTP request with AWS SigV4 (AC-DATA-02)."""
        now = datetime.datetime.now(datetime.UTC)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")

        canonical_uri = urllib.parse.quote(path, safe="/-_.~")
        if not canonical_uri.startswith("/"):
            canonical_uri = "/" + canonical_uri

        canonical_querystring = ""
        if query_params:
            sorted_params = sorted(query_params.items())
            canonical_querystring = "&".join(
                f"{urllib.parse.quote(k, safe='-_.~')}="
                f"{urllib.parse.quote(v, safe='-_.~')}"
                for k, v in sorted_params
            )

        payload_hash = hashlib.sha256(payload).hexdigest()

        headers: dict[str, str] = {
            "host": self.host,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
        }
        if extra_headers:
            for k, v in extra_headers.items():
                headers[k.lower()] = v.strip()

        signed_header_keys = sorted(headers.keys())
        signed_headers = ";".join(signed_header_keys)
        canonical_headers = "".join(f"{k}:{headers[k]}\n" for k in signed_header_keys)

        canonical_request = (
            f"{method.upper()}\n"
            f"{canonical_uri}\n"
            f"{canonical_querystring}\n"
            f"{canonical_headers}\n"
            f"{signed_headers}\n"
            f"{payload_hash}"
        )

        algorithm = "AWS4-HMAC-SHA256"
        credential_scope = f"{date_stamp}/{self.config.region}/s3/aws4_request"
        string_to_sign = (
            f"{algorithm}\n"
            f"{amz_date}\n"
            f"{credential_scope}\n"
            f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        )

        signing_key = _get_signature_key(
            self.config.secret_access_key,
            date_stamp,
            self.config.region,
            "s3",
        )
        signature = hmac.new(
            signing_key,
            string_to_sign.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        authorization_header = (
            f"{algorithm} "
            f"Credential={self.config.access_key_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, "
            f"Signature={signature}"
        )

        url = f"{self.scheme}://{self.host}{canonical_uri}"
        if canonical_querystring:
            url = f"{url}?{canonical_querystring}"

        req_data = payload if method.upper() in ("PUT", "POST") else None
        req = urllib.request.Request(url, data=req_data, method=method.upper())
        req.add_header("Authorization", authorization_header)
        for k, v in headers.items():
            if k != "host":
                req.add_header(k, v)
        return req

    def head_object(self, key: str) -> dict[str, str] | None:
        """Check if an object exists and retrieve its headers. Returns None if 404."""
        clean_key = key.lstrip("/")
        path = f"/{self.config.bucket}/{clean_key}"
        req = self._build_request("HEAD", path)
        try:
            with urllib.request.urlopen(req) as resp:
                return {k.lower(): v for k, v in resp.headers.items()}
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return None
            raise R2Error(
                f"HEAD failed for {key}: HTTP {err.code} {err.reason}"
            ) from err
        except urllib.error.URLError as err:
            raise R2Error(
                f"connection error contacting R2 for {key}: {err.reason}"
            ) from err

    def put_object(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload raw bytes to an R2 object key (AC-DATA-02)."""
        clean_key = key.lstrip("/")
        path = f"/{self.config.bucket}/{clean_key}"
        extra_headers = {"content-type": content_type}
        req = self._build_request(
            "PUT", path, payload=data, extra_headers=extra_headers
        )
        try:
            with urllib.request.urlopen(req) as resp:
                if resp.status not in (200, 201, 204):
                    raise R2Error(f"PUT failed for {key}: HTTP {resp.status}")
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", errors="replace")
            raise R2Error(
                f"PUT failed for {key}: HTTP {err.code} {err.reason} - {body}"
            ) from err
        except urllib.error.URLError as err:
            raise R2Error(
                f"connection error uploading {key} to R2: {err.reason}"
            ) from err

    def get_object(self, key: str) -> bytes:
        """Download raw bytes from an R2 object key (AC-DATA-03)."""
        clean_key = key.lstrip("/")
        path = f"/{self.config.bucket}/{clean_key}"
        req = self._build_request("GET", path)
        try:
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
                return bytes(data)
        except urllib.error.HTTPError as err:
            raise R2Error(
                f"GET failed for {key}: HTTP {err.code} {err.reason}"
            ) from err
        except urllib.error.URLError as err:
            raise R2Error(
                f"connection error downloading {key} from R2: {err.reason}"
            ) from err

    def upload_file(
        self,
        local_path: Path,
        key: str,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload a local file to R2."""
        if not local_path.is_file():
            raise R2Error(f"local file not found: {local_path}")
        data = local_path.read_bytes()
        self.put_object(key, data, content_type=content_type)

    def download_file(self, key: str, local_path: Path) -> None:
        """Download an R2 object to a local file."""
        data = self.get_object(key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)

    def list_objects(self, prefix: str = "") -> list[dict[str, Any]]:
        """List objects matching a prefix."""
        path = f"/{self.config.bucket}"
        query = {"list-type": "2"}
        if prefix:
            query["prefix"] = prefix.lstrip("/")
        req = self._build_request("GET", path, query_params=query)
        try:
            with urllib.request.urlopen(req) as resp:
                body = resp.read()
        except urllib.error.HTTPError as err:
            raise R2Error(f"list_objects failed: HTTP {err.code} {err.reason}") from err
        except urllib.error.URLError as err:
            raise R2Error(f"connection error listing R2 objects: {err.reason}") from err

        root = ET.fromstring(body)
        objects: list[dict[str, Any]] = []
        for elem in root.iter():
            tag = elem.tag.split("}")[-1]
            if tag == "Contents":
                obj: dict[str, Any] = {}
                for child in elem:
                    child_tag = child.tag.split("}")[-1]
                    if child_tag == "Key" and child.text:
                        obj["key"] = child.text
                    elif child_tag == "Size" and child.text:
                        obj["size"] = int(child.text)
                if "key" in obj:
                    objects.append(obj)
        return objects
