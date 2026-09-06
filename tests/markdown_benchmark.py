"""Time complete Markdown narration; writes only to test-output/markdown-benchmark."""
import json
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    import torch
    from core import ROOT, defaults
    from worker import run_batch
    output = ROOT / 'test-output/markdown-benchmark'
    config = defaults()
    config.update(output=str(output), tts='Qwen/Qwen3-TTS-12Hz-1.7B-Base', device='cuda:0',
                  voice=str(ROOT / 'assets/voice.wav'), transcript=str(ROOT / 'assets/transcript.txt'),
                  prompt=str(ROOT / 'assets/prompt.txt'))
    files = [str(Path(name).resolve()) for name in args.files]
    if not files or any(Path(name).suffix != '.md' for name in files):
        raise SystemExit('Pass one or more Markdown paths.')
    started = time.monotonic()

    def emit(event, **data):
        print(json.dumps(dict(event=event, elapsed=round(time.monotonic() - started, 2), **data)), flush=True)

    torch.manual_seed(42)
    run_batch(config, files, emit)


if __name__ == '__main__':
    main()
