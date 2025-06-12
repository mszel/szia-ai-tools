"""
Simple parsers for various document types. These parsers are designed to convert
raw content from different formats (PDF, HTML, Markdown, plain text) into
Markdown format.

TODO: Future goal: finding well working existing libraries for these formats.
"""

import io
import re
from typing import Any

import html2text
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text


def clean_text(text: str, encoding: str, ignore_single_newline: bool) -> str:
    """
    A basic cleaner for plain-text/Markdown content.

    Steps:
      1. Drops any characters not representable in `encoding`.
      2. Collapses runs of 3+ newlines down to 2 newlines.
      3. Strips leading/trailing whitespace on each line and replaces
         runs of spaces/tabs with a single space.

    Args:
        text:     The raw text to clean.
        encoding: The target encoding (e.g. 'utf-8', 'latin1', 'utf-8-sig').
        ignore_single_newline: If True, removes all single newlines,
                               collapsing them into spaces.

    Returns:
        The cleaned text.
    """
    # 1) Drop un-encodable chars
    cleaned = text.encode(encoding, errors="ignore").decode(encoding, errors="ignore")

    # 2) Collapse 3+ newlines → exactly 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # 3a) Collapse spaces/tabs → single space
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # 3b) Strip whitespace at ends of lines
    lines = [line.strip() for line in cleaned.splitlines()]
    cleaned = "\n".join(lines)

    # 3c) If ignore_newline, remove all single newlines
    if ignore_single_newline:
        lines = [line.strip() for line in cleaned.split("\n\n")]
        lines = [line.replace("\n", " ") for line in lines]
        cleaned = "\n\n".join(lines)

    # Rejoin, preserving the now-normalized newlines
    return cleaned


def parse_pdf(
    content: bytes, encoding: str, ignore_single_newline: bool
) -> tuple[str, dict[str, Any]]:
    """Convert raw PDF bytes into cleaned text."""

    metadata = {"parse_error": False, "parse_error_message": None}

    try:
        fp = io.BytesIO(content)
        raw_text = extract_text(fp)
        cleaned_text = clean_text(raw_text, encoding, ignore_single_newline)

    except Exception as e:
        metadata["parse_error"] = True
        metadata["parse_error_message"] = str(e)
        cleaned_text = ""

    return cleaned_text, metadata


def parse_html(
    content: str, encoding: str, ignore_single_newline: bool
) -> tuple[str, dict[str, Any]]:
    """Convert raw HTML into cleaned Markdown text."""

    metadata = {"parse_error": False, "parse_error_message": None}

    try:
        soup = BeautifulSoup(content, "html.parser")
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            metadata["title"] = title_tag.string.strip()
        for tag in ["footer", "header"]:
            if soup.find(tag):
                soup.find(tag).extract()
        markdown_text = html2text.html2text(str(soup))
        cleaned_text = clean_text(markdown_text, encoding, ignore_single_newline)

    except Exception as e:
        metadata["parse_error"] = True
        metadata["parse_error_message"] = str(e)
        cleaned_text = ""

    return cleaned_text, metadata


def parse_markdown(
    content: str, encoding: str, ignore_single_newline: bool
) -> tuple[str, dict[str, Any]]:
    """Execute basic cleaning on Markdown content."""

    metadata = {"parse_error": False, "parse_error_message": None}

    try:
        # Markdown is already text, so we just clean it
        cleaned_text = clean_text(content, encoding, ignore_single_newline)

    except Exception as e:
        metadata["parse_error"] = True
        metadata["parse_error_message"] = str(e)
        cleaned_text = ""

    return cleaned_text, metadata


def parse_text(
    content: str, encoding: str, ignore_single_newline: bool
) -> tuple[str, dict[str, Any]]:
    """
    Execute basic cleaning on Text content.
    TODO: later, adding some automated structuring of the text
    """

    metadata = {"parse_error": False, "parse_error_message": None}

    try:
        cleaned_text = clean_text(content, encoding, ignore_single_newline)

    except Exception as e:
        metadata["parse_error"] = True
        metadata["parse_error_message"] = str(e)
        cleaned_text = ""

    return cleaned_text, metadata
