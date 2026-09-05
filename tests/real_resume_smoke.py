"""Cancel after a real speech batch, then verify resume keeps its exact files."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from checkpoints import digest, valid_audio
    from core import ROOT, defaults
    from worker import Cancelled, run_batch
    source = Path(sys.argv[1]).resolve()
    if source.suffix != '.md':
        raise SystemExit('Pass a short Markdown file.')
    config = defaults()
    config.update(output=str(ROOT / 'test-output/resume-audio'),
                  tts='Qwen/Qwen3-TTS-12Hz-1.7B-Base', device='cuda:0',
                  voice=str(ROOT / 'assets/voice.wav'), transcript=str(ROOT / 'assets/transcript.txt'),
                  prompt=str(ROOT / 'assets/prompt.txt'))
    folder = None
    start = time.monotonic()

    def emit(event, **data):
        nonlocal folder
        if data.get('folder'):
            folder = Path(data['folder'])
        print(json.dumps(dict(event=event, elapsed=round(time.monotonic() - start, 2), **data)), flush=True)
        if event == 'progress' and data['message'].startswith('Speaking') and data['fraction'] > 0.25:
            raise Cancelled()

    try:
        run_batch(config, [str(source)], emit)
        raise AssertionError('Expected cancellation')
    except Cancelled:
        pass
    saved = {path: (digest(path), path.stat().st_mtime_ns) for path in (folder / 'passages').glob('*.flac')}
    assert saved and not (folder / 'audio.flac').exists()
    print('Resuming with', len(saved), 'completed passages.', flush=True)
    events = []
    run_batch(config, [str(source)], lambda event, **data: events.append(dict(event=event, **data)), {str(source): str(folder)})
    assert events[-1]['completed'] == 1, events
    assert all((digest(path), path.stat().st_mtime_ns) == value for path, value in saved.items())
    assert valid_audio(folder / 'audio.flac')
    print('Real cancellation/resume passed; saved audio unchanged:', folder / 'audio.flac', flush=True)


if __name__ == '__main__':
    main()
