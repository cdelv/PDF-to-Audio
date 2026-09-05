"""Check MarkItDown installation with a generated PDF, not the user's documents."""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from pdf_input import extract_pdf
    text = b'BT /F1 12 Tf 72 720 Td (Hello. This is a PDF installation test.) Tj ET'
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
               b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
               b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
               b'<< /Length ' + str(len(text)).encode() + b' >>\nstream\n' + text + b'\nendstream']
    document = b'%PDF-1.4\n'
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(document))
        document += str(index).encode() + b' 0 obj\n' + obj + b'\nendobj\n'
    xref = len(document)
    document += b'xref\n0 6\n0000000000 65535 f \n'
    document += b''.join(f'{offset:010d} 00000 n \n'.encode() for offset in offsets[1:])
    document += f'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode()
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'installation.pdf'
        path.write_bytes(document)
        result = extract_pdf(path)
        assert 'Hello. This is a PDF installation test.' in result, result
    print('Installed MarkItDown PDF extraction passed.')


if __name__ == '__main__':
    main()
