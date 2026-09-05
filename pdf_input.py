"""MarkItDown PDF extraction with reading-order-aware prose layout."""
from markitdown import DocumentConverterResult, MarkItDown
from markitdown.converters import PdfConverter
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams


class ProsePdfConverter(PdfConverter):
    def convert(self, file_stream, stream_info, **kwargs):
        # MarkItDown's form heuristics can turn entire scientific-paper columns
        # into spurious tables. Use its existing PDFMiner backend directly,
        # preserving columns and word spacing rather than inferring form cells.
        # Some IEEE papers place all body text inside Form XObjects. Without
        # all_texts, PDFMiner emits those letters without spaces or paragraphs.
        text = extract_text(file_stream, laparams=LAParams(all_texts=True, detect_vertical=True))
        return DocumentConverterResult(markdown=text.replace("\f", "\n\n"))


def extract_pdf(path):
    converter = MarkItDown(enable_plugins=False)
    converter.register_converter(ProsePdfConverter(), priority=-1)
    return converter.convert_local(str(path)).text_content
