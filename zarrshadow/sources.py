"""Record and check the files that references point into.

A reference is a URL or path plus a byte range, so it goes stale without any
error if the file behind it is replaced. generate_rfs records each source
file's size and, for remote files, its ETag under "sources". When reading,
RfsStore sends If-Match with every range request so the server refuses one
whose file has changed, compares the total size the server reports, and checks
the size of local files.
"""

from __future__ import annotations

import os
import re
import warnings

import requests

from .url_resolver import dandi_auth_headers, resolve_url


class SourceChangedError(RuntimeError):
    """A referenced file no longer matches the one the references were made from."""


_DANDI_ASSET_DOWNLOAD = re.compile(
    r"^(?P<api>https://api(?:\.sandbox)?\.dandiarchive\.org/api)/assets/(?P<asset>[0-9a-f-]+)/download/?$"
)
_CONTENT_RANGE_TOTAL = re.compile(r"bytes \d+-\d+/(\d+)")


def describe_source(url_or_path: str) -> dict:
    """Return {"size", "etag"} for a source file; "etag" only for remote files.

    DANDI asset download URLs are described from the asset's metadata, whose
    dandi-etag is the ETag S3 reports for the blob. Other URLs are described
    from a one-byte range request.
    """
    if not url_or_path.startswith(("http://", "https://")):
        return {"size": os.path.getsize(url_or_path)}
    match = _DANDI_ASSET_DOWNLOAD.match(url_or_path)
    if match:
        info = _describe_dandi_asset(match["api"], match["asset"])
        if info:
            return info
    return _describe_http(url_or_path)


def _describe_dandi_asset(api: str, asset_id: str) -> dict:
    url = f"{api}/assets/{asset_id}/"
    response = requests.get(url, headers=dandi_auth_headers(url))
    if not response.ok:
        return {}
    metadata = response.json()
    info: dict = {}
    if "contentSize" in metadata:
        info["size"] = int(metadata["contentSize"])
    etag = (metadata.get("digest") or {}).get("dandi:dandi-etag")
    if etag:
        info["etag"] = f'"{etag}"'
    return info


def _describe_http(url: str) -> dict:
    response = requests.get(resolve_url(url), headers={"Range": "bytes=0-0"}, stream=True)
    try:
        response.raise_for_status()
        info: dict = {}
        total = _content_range_total(response)
        if total is not None:
            info["size"] = total
        elif response.status_code == 200 and "Content-Length" in response.headers:
            info["size"] = int(response.headers["Content-Length"])
        etag = response.headers.get("ETag")
        if etag and not etag.startswith("W/"):
            info["etag"] = etag
        if not info:
            warnings.warn(f"{url} reports neither a size nor an ETag; changes to it cannot be detected")
        return info
    finally:
        response.close()


def _content_range_total(response: requests.Response) -> int | None:
    match = _CONTENT_RANGE_TOTAL.match(response.headers.get("Content-Range", ""))
    return int(match[1]) if match else None


class SourceChecker:
    """Checks reads against the sources recorded in a reference file system.

    Parameters
    ----------
    sources : dict
        The "sources" entry of an RFS: URL or path -> {"size", "etag"}.
    enabled : bool
        If False, nothing is checked.
    """

    def __init__(self, sources: dict[str, dict], *, enabled: bool = True) -> None:
        self._sources = sources if enabled else {}
        self._checked_local: set[str] = set()

    def request_headers(self, url: str) -> dict[str, str]:
        """Headers that make the server refuse a range request if the file changed."""
        etag = self._sources.get(url, {}).get("etag")
        return {"If-Match": etag} if etag else {}

    def check_response(self, url: str, response: requests.Response) -> None:
        """Raise SourceChangedError if a range response shows the file changed."""
        if response.status_code == 412:
            raise SourceChangedError(
                f"{url} has changed since the references were generated (ETag no longer matches)"
            )
        expected = self._sources.get(url, {}).get("size")
        total = _content_range_total(response)
        if expected is not None and total is not None and total != expected:
            raise SourceChangedError(
                f"{url} has changed since the references were generated "
                f"(size {total:,} bytes, recorded {expected:,})"
            )

    def check_local(self, path: str) -> None:
        """Raise SourceChangedError if a local file's size differs from the recorded one."""
        if path in self._checked_local:
            return
        expected = self._sources.get(path, {}).get("size")
        if expected is not None and os.path.getsize(path) != expected:
            raise SourceChangedError(
                f"{path} has changed since the references were generated "
                f"(size {os.path.getsize(path):,} bytes, recorded {expected:,})"
            )
        self._checked_local.add(path)
