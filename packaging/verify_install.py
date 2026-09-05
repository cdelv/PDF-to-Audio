"""Offline installation smoke check, including a short CPU inference option."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import ROOT, defaults
from worker import Cleaner, Speaker, local_model

import torch
import transformers
import qwen_tts
import markitdown
import soundfile

for name in __import__('json').loads((ROOT / 'models.json').read_text()):
    path = Path(local_model(name))
    assert (path / 'config.json').is_file(), name
    assert list(path.glob('*.safetensors')), name
assert (ROOT / 'assets/voice.wav').is_file()
print('Offline models and inference dependencies verified.', flush=True)
print('CUDA available:', torch.cuda.is_available(), flush=True)
if '--cpu-inference' in sys.argv:
    torch.set_num_threads(2)
    config = defaults()
    config.update(device='cpu', llm='Qwen/Qwen3-0.6B',
                  voice=str(ROOT / 'assets/voice.wav'), transcript=str(ROOT / 'assets/transcript.txt'),
                  prompt=str(ROOT / 'assets/prompt.txt'))
    cleaner = Cleaner(config)
    text = cleaner.clean('Hello. This is a local audio test.', lambda *_: None, 'English')
    assert text.strip()
    del cleaner
    from worker import release_gpu
    release_gpu()
    speaker = Speaker(config)
    wave, rate = speaker.speak('Hello.', 'English')
    assert len(wave) > rate / 10
    print('Offline CPU cleanup and voice-cloned speech passed.', flush=True)
