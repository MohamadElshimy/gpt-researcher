"""PDF scraper: downloads the document (unchanged) and hands the bytes to the
page router, which picks a parser backend per page and returns one Markdown
string. The scraper contract -- ``scrape() -> (content, image_urls, title)`` --
is unchanged, so nothing downstream needs to change.
"""

import logging
from urllib.parse import urlparse

import requests

from gpt_researcher.scraper.pdf_router import RouterSettings, route_pdf_detailed

logger = logging.getLogger(__name__)


class PyMuPDFScraper:

    def __init__(self, link, session=None):
        """
        Initialize the scraper with a link and an optional session.

        Args:
          link (str): The URL or local file path of the PDF document.
          session (requests.Session, optional): An optional session for making HTTP requests.
        """
        self.link = link
        self.session = session
        self.failure_reason = None

    def is_url(self) -> bool:
        """
        Check if the provided `link` is a valid URL.

        Returns:
          bool: True if the link is a valid URL, False otherwise.
        """
        try:
            result = urlparse(self.link)
            return all([result.scheme, result.netloc])  # Check for valid scheme and network location
        except Exception:
            return False

    def scrape(self) -> tuple[str, list[str], str]:
        """
        Download the PDF (URL) or read it (local path), route every page to the
        cheapest parser that can handle it, and return the recombined Markdown.

        Returns:
          tuple[str, list[str], str]: (content, image_urls, title). On any
          failure the empty-result contract ``("", [], "")`` is kept and
          ``self.failure_reason`` says why.
        """
        try:
            pdf_bytes = self._fetch_bytes()
        except requests.exceptions.Timeout:
            logger.warning("Download timed out. Please check the link : %s", self.link)
            self.failure_reason = "PDF download timed out"
            return "", [], ""
        except Exception as e:
            logger.warning("Error downloading PDF : %s %s", self.link, e)
            self.failure_reason = f"PDF download failed: {e}"
            return "", [], ""

        try:
            routed = route_pdf_detailed(pdf_bytes, self.link)
        except Exception as e:
            # Malformed or non-PDF bytes: keep the historical empty-result contract.
            logger.warning("Error loading PDF : %s %s", self.link, e)
            self.failure_reason = f"PDF could not be parsed: {e}"
            return "", [], ""

        if not routed.markdown:
            self.failure_reason = "; ".join(routed.notes) or "PDF had no extractable text"
        return routed.markdown, [], routed.title

    # ------------------------------------------------------------------ fetch
    def _fetch_bytes(self) -> bytes:
        try:
            limit = RouterSettings.from_env().max_pdf_bytes
        except ValueError as e:  # a malformed PDF_ROUTER_* value must not break downloads
            logger.warning("Invalid PDF router setting, using defaults: %s", e)
            limit = RouterSettings().max_pdf_bytes
        if not self.is_url():
            with open(self.link, "rb") as f:
                data = f.read(limit + 1)
            if len(data) > limit:
                raise ValueError(f"PDF larger than {limit} bytes")
            return data

        http = self.session or requests
        try:
            response = http.get(self.link, timeout=(5, 30), stream=True)
            response.raise_for_status()
        except requests.exceptions.SSLError:
            logger.warning("SSL verification failed for %s, retrying without verification", self.link)
            response = http.get(self.link, timeout=(5, 30), stream=True, verify=False)
            response.raise_for_status()

        declared = response.headers.get("Content-Length") if hasattr(response, "headers") else None
        if declared and declared.isdigit() and int(declared) > limit:
            raise ValueError(f"PDF larger than {limit} bytes (Content-Length {declared})")

        buf = bytearray()
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            buf.extend(chunk)
            if len(buf) > limit:
                raise ValueError(f"PDF larger than {limit} bytes")
        return bytes(buf)
