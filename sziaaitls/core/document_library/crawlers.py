"""
Crawler for downloading web content into Document objects.

TODO: this module is extremely basic and needs to be extended with:
- Support for more content types (e.g., DOCX, images)
- Supporting recursive crawling
- Rate limiting and politeness policies
- Adding blacklists/whitelists for URLs
- Caching downloaded content
- Error handling improvements
- Support for custom headers and authentication
"""

import aiohttp

from .documents import (
    Document,
    DocumentMetadata,
    HtmlDocument,
    MarkdownDocument,
    PdfDocument,
)


class Crawler:
    """
    Asynchronous URL crawler producing Document objects.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    async def get_documents(self, urls: list[str]) -> list[Document]:
        """
        Fetch each URL and wrap in the appropriate Document type.

        On failure or unknown content-type, returns empty MarkdownDocument
        with crawling_error metadata.
        """
        docs: list[Document] = []
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            for url in urls:
                metadata = DocumentMetadata({"uri": url})
                try:
                    async with session.get(url) as resp:
                        content_type = resp.headers.get("Content-Type", "")
                        raw = await resp.read()
                        if "application/pdf" in content_type:
                            doc = PdfDocument(raw, metadata)
                        elif "text/html" in content_type:
                            html = raw.decode("utf-8", errors="ignore")
                            doc = HtmlDocument(html, metadata)
                        else:
                            text = raw.decode("utf-8", errors="ignore")
                            doc = MarkdownDocument(text, metadata)
                except Exception as e:
                    metadata.add_items({"crawling_error": True, "crawling_error_message": str(e)})
                    doc = MarkdownDocument("", metadata)
                docs.append(doc)
        return docs
