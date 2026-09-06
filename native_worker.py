"""Console helper for a windowed frozen GUI; preserves JSON pipes on Windows."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def check(inference=False):
    """Shared source/frozen installation check; full inference is opt-in."""
    import torch
    import qwen_tts  # noqa: F401 -- Verify installed dependency imports.
    import transformers  # noqa: F401
    import soundfile  # noqa: F401
    from pdf_input import extract_pdf  # noqa: F401
    from core import ROOT, defaults
    from model_store import DEFAULT_MODELS, MODELS, missing_models
    from worker import Cleaner, Speaker, release_gpu
    assert all((ROOT / 'assets' / name).is_file() for name in ('voice.wav', 'transcript.txt', 'prompt.txt', 'icon.svg'))
    print('Dependencies and assets verified. CUDA:', torch.cuda.is_available(), flush=True)
    if inference:
        assert not missing_models(DEFAULT_MODELS)
        optional = [name for name in MODELS if name not in DEFAULT_MODELS]
        assert missing_models(optional) == optional, 'Setup must install only the default models.'
        config = defaults()
        config.update(device='cpu', voice=str(ROOT / 'assets/voice.wav'),
                      transcript=str(ROOT / 'assets/transcript.txt'), prompt=str(ROOT / 'assets/prompt.txt'))
        cleaner = Cleaner(config)
        try:
            assert cleaner.clean('Hello. This is an installation test.', lambda *_: None, 'English').strip()
        finally:
            cleaner.close()
            release_gpu()
        speaker = Speaker(config)
        try:
            wave, rate = speaker.speak('Hello.', 'English')
            assert len(wave) > rate / 10
        finally:
            speaker.close()
            release_gpu()
        print('Offline CPU cleanup and speech passed.', flush=True)

if __name__ == '__main__':
    if '--check' in sys.argv or '--self-test' in sys.argv:
        check(inference='--self-test' in sys.argv)
    elif '--setup-models' in sys.argv:
        from model_store import DEFAULT_MODELS, ensure_models
        ensure_models(DEFAULT_MODELS, lambda event, **data: print(data['message'], flush=True))
    else:
        from worker import main
        sys.exit(main())
