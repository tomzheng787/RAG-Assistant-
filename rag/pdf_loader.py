"""PDF loading: parse PDFs into per-page Documents with source metadata, ready for chunking and embedding."""

import logging
import re
from pathlib import Path

import pymupdf

from rag.document import Document

logger = logging.getLogger(__name__)

# Don't preserve ligatures ("ﬁ" expands to "fi", otherwise retrieval can't match it); clip text outside the page
_TEXT_FLAGS = pymupdf.TEXT_MEDIABOX_CLIP

# CJK characters and full-width punctuation: line breaks between them are joined directly, without a space
_CJK_RE = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")


def _join_lines(lines: list[str]) -> str:
    """Merge the lines of one text block into a paragraph: CJK is joined directly, other text with a space,
    and words hyphenated across a line break are rejoined."""
    merged = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if merged.endswith("-") and merged[-2:-1].isalpha() and line[0].islower():
            merged = merged[:-1]  # "gen-" + "erator" -> "generator"
        elif merged and not (_CJK_RE.match(merged[-1]) or _CJK_RE.match(line[0])):
            merged += " "
        merged += line
    return merged


def _extract_page_text(page: pymupdf.Page) -> str:
    """Extract one page's text in reading order.

    A PyMuPDF block roughly corresponds to a paragraph: line breaks inside a block are layout wraps and get merged,
    while blocks are separated by blank lines to preserve paragraph structure.
    """
    paragraphs = []
    for block in page.get_text("blocks", flags=_TEXT_FLAGS, sort=True):
        x0, y0, x1, y1, text, block_no, block_type = block
        if block_type != 0:  # 1 = image block
            continue
        paragraph = _join_lines(text.splitlines())
        if paragraph:
            paragraphs.append(paragraph)
    text = "\n\n".join(paragraphs)
    text = text.replace("\x00", "").replace("\u00a0", " ")
    return re.sub(r"[ \t]{2,}", " ", text)


def load_pdf(path: str | Path, password: str | None = None) -> list[Document]:
    """Read a single PDF, producing one Document per page.

    Pages without text (usually scanned or image-only pages) are skipped and logged; they need OCR to extract.
    """
    path = Path(path)
    documents = []
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass and not pdf.authenticate(password or ""):
            raise ValueError(f"PDF is encrypted and requires the correct password: {path}")

        title = (pdf.metadata or {}).get("title") or path.stem
        empty_pages = []
        for page in pdf:
            text = _extract_page_text(page)
            if not text:
                empty_pages.append(page.number + 1)
                continue
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source": str(path.resolve()),
                        "file_name": path.name,
                        "title": title,
                        "page": page.number + 1,  # 1-based, for display in citations
                        "total_pages": pdf.page_count,
                    },
                )
            )

    if empty_pages:
        logger.warning(
            "%s: %d/%d pages have no extractable text (possibly scanned, OCR required): %s",
            path.name, len(empty_pages), len(empty_pages) + len(documents), empty_pages,
        )
    return documents


def load_pdfs(path: str | Path, recursive: bool = True) -> list[Document]:
    """Read one PDF file, or every PDF under a directory. A file that fails to parse doesn't affect the others."""
    path = Path(path)
    if path.is_file():
        return load_pdf(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")

    pattern = "**/*" if recursive else "*"
    pdf_files = sorted(p for p in path.glob(pattern) if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdf_files:
        logger.warning("No PDF files found in directory: %s", path)

    documents = []
    for pdf_file in pdf_files:
        try:
            docs = load_pdf(pdf_file)
        except Exception as e:
            logger.error("Failed to parse, skipped %s: %s", pdf_file.name, e)
            continue
        logger.info("Loaded %s: %d pages", pdf_file.name, len(docs))
        documents.extend(docs)
    return documents


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load PDFs and preview the extracted text")
    parser.add_argument("path", help="a PDF file or a directory containing PDFs")
    parser.add_argument("--preview", type=int, default=300, help="number of characters to preview per page")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    docs = load_pdfs(args.path)
    print(f"\nTotal: {len(docs)} pages, {sum(len(d.text) for d in docs)} characters\n")
    for doc in docs:
        print(f"--- {doc.metadata['file_name']} page {doc.metadata['page']}/{doc.metadata['total_pages']} ---")
        print(doc.text[: args.preview])
        print()
