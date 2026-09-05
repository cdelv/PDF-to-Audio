"""Real GPU integration check: two PDFs plus Markdown, using the shipped voice."""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import ROOT, defaults


def pdf(path, sentence):
    sentence = sentence.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
    stream = f'BT /F1 10 Tf 40 740 Td ({sentence}) Tj ET'.encode('cp1252')
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
               b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
               b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>',
               f'<< /Length {len(stream)} >>\nstream\n'.encode() + stream + b'\nendstream']
    data = bytearray(b'%PDF-1.4\n')
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f'{i} 0 obj\n'.encode() + obj + b'\nendobj\n')
    xref = len(data)
    data.extend(f'xref\n0 {len(offsets)}\n0000000000 65535 f \n'.encode())
    for offset in offsets[1:]:
        data.extend(f'{offset:010d} 00000 n \n'.encode())
    data.extend(f'trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF'.encode())
    path.write_bytes(data)


def main():
    output = ROOT/'test-output'
    output.mkdir(exist_ok=True)
    pdf(output/'First.pdf', 'A quiet morning is a good time to read. Every page brings a new idea.')
    pdf(output/'Second.pdf', 'The second document has its own audio file. All of this runs on your computer.')
    (output/'Notes.md').write_text('# Notes\n\nYour text can become an audiobook. Take your reading with you.')
    config = defaults()
    config.update(output=str(output/'audio'), voice=str(ROOT/'assets/voice.wav'),
                  transcript=str(ROOT/'assets/transcript.txt'), prompt=str(ROOT/'assets/prompt.txt'))
    request = dict(config=config, files=[str(output/name) for name in ('First.pdf', 'Second.pdf', 'Notes.md')])
    process = subprocess.Popen([sys.executable, '-I', '-u', str(ROOT/'worker.py')], stdin=subprocess.PIPE, text=True)
    process.communicate(json.dumps(request)+'\n')
    return process.returncode


if __name__ == '__main__':
    sys.exit(main())
