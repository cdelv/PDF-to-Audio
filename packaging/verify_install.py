"""Offline installation smoke check, including a short CPU inference option."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import ROOT, defaults
from worker import Cleaner, Speaker, local_model
from model_store import DEFAULT_MODELS, MODELS, missing_models

import torch
import transformers
import qwen_tts
import markitdown
import soundfile

if '--dependencies-only' not in sys.argv:
    for name in DEFAULT_MODELS:
        path = Path(local_model(name))
        assert (path / 'config.json').is_file(), name
        assert list(path.glob('*.safetensors')), name
    if '--defaults-only' in sys.argv:
        optional = [name for name in MODELS if name not in DEFAULT_MODELS]
        assert missing_models(optional) == optional, 'Optional models were installed unexpectedly'
assert (ROOT / 'assets/voice.wav').is_file()
print('Inference dependencies verified.' if '--dependencies-only' in sys.argv else 'Offline default models and inference dependencies verified.', flush=True)
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
