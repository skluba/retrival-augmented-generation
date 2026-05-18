from rag_pdf_app.parsing.models import BBox, TextSpan
from rag_pdf_app.parsing.relations import nearest_neighbors


def test_nearest_vertical_text_neighbors() -> None:
    spans = [
        TextSpan(
            span_id="top-box",
            page_index=0,
            bbox=BBox(page_index=0, x0=0.0, y0=10.0, x1=180.0, y1=40.0),
            text="Introduction paragraph",
            extractor="pdfminer",
        ),
        TextSpan(
            span_id="bottom-box",
            page_index=0,
            bbox=BBox(page_index=0, x0=0.0, y0=200.0, x1=200.0, y1=240.0),
            text="Disclaimer paragraph",
            extractor="pdfminer",
        ),
    ]

    anchor = BBox(page_index=0, x0=40.0, y0=80.0, x1=120.0, y1=150.0)  # "image" bbox
    above_id, below_id = nearest_neighbors(anchor, spans_on_page=spans)

    assert above_id == "top-box"
    assert below_id == "bottom-box"
