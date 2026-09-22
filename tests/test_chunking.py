from app.rag.chunking import chunk_document
from app.rag.parser import PageText, ParsedDocument


def test_chunk_document_preserves_metadata():
    parsed = ParsedDocument(
        filename="paper.pdf",
        num_pages=1,
        pages=[
            PageText(page_number=5, section="Methodology", text="A " * 800),
        ],
    )

    chunks = chunk_document("doc_001", parsed)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.document_id == "doc_001"
        assert chunk.filename == "paper.pdf"
        assert chunk.page_number == 5
        assert chunk.section == "Methodology"
        assert chunk.chunk_id.startswith("chunk_")
        assert chunk.text.strip() != ""


def test_chunk_document_skips_empty_pages():
    parsed = ParsedDocument(
        filename="paper.pdf",
        num_pages=2,
        pages=[
            PageText(page_number=1, section="Unknown", text="   "),
            PageText(page_number=2, section="Abstract", text="Some real content here."),
        ],
    )

    chunks = chunk_document("doc_002", parsed)

    assert all(chunk.page_number == 2 for chunk in chunks)
