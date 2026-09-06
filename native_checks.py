"""Installed frozen-runtime verification, also used by release automation."""
def check(inference=False):
    import torch
    import qwen_tts  # noqa: F401 -- Verify bundled dependency imports.
    import transformers  # noqa: F401
    import soundfile  # noqa: F401
    from core import ROOT, defaults
    from model_store import DEFAULT_MODELS, MODELS, missing_models
    from worker import Cleaner, Speaker, release_gpu
    from pdf_input import extract_pdf  # noqa: F401 -- Verify PDF dependencies too.
    assert (ROOT / 'assets/voice.wav').is_file()
    assert (ROOT / 'model-files.json').is_file()
    print('Frozen dependencies and assets verified. CUDA:', torch.cuda.is_available(), flush=True)
    if inference:
        assert not missing_models(DEFAULT_MODELS)
        optional = [n for n in MODELS if n not in DEFAULT_MODELS]
        assert missing_models(optional) == optional
        config = defaults()
        config.update(device='cpu', voice=str(ROOT / 'assets/voice.wav'),
                      transcript=str(ROOT / 'assets/transcript.txt'), prompt=str(ROOT / 'assets/prompt.txt'))
        cleaner = Cleaner(config)
        assert cleaner.clean('Hello. This is an installation test.', lambda *_: None, 'English').strip()
        del cleaner
        release_gpu()
        speaker = Speaker(config)
        wave, rate = speaker.speak('Hello.', 'English')
        assert len(wave) > rate / 10
        print('Frozen offline CPU narration passed.', flush=True)
