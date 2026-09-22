from app.models.schemas import (
    Citation,
    ContentType,
    ExecutionPlan,
    FigureRecord,
    ReviewResult,
    TableRecord,
    TextChunk,
)


def test_text_chunk_defaults():
    chunk = TextChunk(
        document_id="doc_001",
        filename="paper.pdf",
        page_number=5,
        chunk_id="chunk_001",
        text="Some methodology text.",
    )
    assert chunk.content_type == ContentType.TEXT
    assert chunk.section == "Unknown"
    assert chunk.source == "pdf_text"


def test_figure_record_roundtrip():
    fig = FigureRecord(
        document_id="doc_001",
        filename="paper.pdf",
        page_number=3,
        figure_number=1,
        figure_id="fig_001",
        image_path="/tmp/fig1.png",
    )
    assert fig.content_type == ContentType.FIGURE
    data = fig.model_dump()
    assert data["figure_number"] == 1


def test_execution_plan_defaults():
    plan = ExecutionPlan()
    assert plan.query_type == "simple"
    assert plan.tasks == []
    assert plan.requires_multimodal is False
    assert plan.requires_tool is False


def test_review_result_defaults():
    result = ReviewResult(approved=True)
    assert result.missing_information == []


def test_table_record_and_citation():
    table = TableRecord(
        document_id="doc_001",
        filename="paper.pdf",
        page_number=7,
        table_number=2,
        table_id="tbl_001",
    )
    citation = Citation(filename=table.filename, page_number=table.page_number, content_type=ContentType.TABLE)
    assert citation.content_type == ContentType.TABLE
