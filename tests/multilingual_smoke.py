"""Real English, Spanish, and French PDFs in one batch with an English voice."""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import ROOT, defaults
from smoke import pdf
from test_pipeline import LanguageTests


def main():
    output = ROOT/'test-output/multilingual'
    output.mkdir(exist_ok=True, parents=True)
    files = []
    for language in ('English', 'Spanish', 'French'):
        path = output/(language+'.pdf')
        pdf(path, LanguageTests.examples[language])
        files.append(str(path))
    config = defaults()
    config.update(output=str(output/'audio'), voice=str(ROOT/'assets/voice.wav'),
                  transcript=str(ROOT/'assets/transcript.txt'), prompt=str(ROOT/'assets/prompt.txt'),
                  voice_language='English', document_language='Auto')
    request = dict(config=config, files=files)
    process = subprocess.Popen([sys.executable, '-I', '-u', str(ROOT/'worker.py')],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    process.stdin.write(json.dumps(request)+'\n')
    process.stdin.close()
    events = []
    for line in process.stdout:
        print(line, end='', flush=True)
        events.append(json.loads(line))
    assert process.wait() == 0
    assert events[-1] == dict(event='finished', completed=3, failed=0), events[-1]
    for event, language in zip([e for e in events if e['event']=='done'], ('English','Spanish','French')):
        folder = Path(event['folder'])
        metadata = json.loads((folder/'languages.json').read_text())
        assert metadata == dict(document_language=language, voice_language='English'), metadata
        assert (folder/'audio.flac').stat().st_size > 1000
    print('Mixed-language PDF batch passed.')


if __name__ == '__main__':
    main()
