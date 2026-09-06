"""Console helper for a windowed frozen GUI; preserves JSON pipes on Windows."""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')


def check(inference=False, cuda=False, metal=False, gpu=False):
    """Shared source/frozen installation check; full inference is opt-in."""
    import torch
    import qwen_tts  # noqa: F401 -- Verify installed dependency imports.
    import transformers  # noqa: F401
    import soundfile  # noqa: F401
    from pdf_input import extract_pdf  # noqa: F401
    from core import ROOT, defaults
    from model_store import DEFAULT_MODELS, MODELS, missing_models
    from worker import Speaker, model_options, release_gpu, run_batch
    from hardware import virtual_metal
    assert all((ROOT / 'assets' / name).is_file() for name in ('voice.wav', 'transcript.txt', 'prompt.txt', 'icon.svg'))
    print('Dependencies and assets verified. CUDA:', torch.cuda.is_available(), flush=True)
    if cuda:
        assert torch.version.cuda == '12.8', 'CUDA runtime must be installed, even on a CPU-only test runner.'
    if gpu:
        assert torch.cuda.is_available(), 'Hardware test requires a working NVIDIA GPU.'
        inference = True
    if metal:
        assert torch.backends.mps.is_built(), 'macOS runtime must include Metal support.'
        if virtual_metal():
            assert model_options({'device': 'auto'})['device_map'] == 'cpu'
            try:
                model_options({'device': 'mps'})
            except ValueError as error:
                assert 'virtual Apple GPU' in str(error)
            else:
                raise AssertionError('Virtual Metal must be rejected before loading weights.')
            print('Metal runtime and virtual-GPU CPU selection verified. Metal inference SKIPPED: unsupported Apple Paravirtual GPU; physical Mac required.', flush=True)
            return
        if not torch.backends.mps.is_available():
            print('Metal runtime verified; hardware inference SKIPPED: this runner exposes no MPS device.', flush=True)
            return
        print(f'Metal recommended working memory: {torch.mps.recommended_max_memory() / 2**30:.2f} GiB', flush=True)
        inference = True
    if inference:
        assert not missing_models(DEFAULT_MODELS)
        optional = [name for name in MODELS if name not in DEFAULT_MODELS]
        if not gpu:
            assert missing_models(optional) == optional, 'Setup must install only the default models.'
        config = defaults()
        config.update(device='cuda:0' if gpu else 'mps' if metal else 'cpu', voice=str(ROOT / 'assets/voice.wav'),
                      transcript=str(ROOT / 'assets/transcript.txt'), prompt=str(ROOT / 'assets/prompt.txt'))
        import tempfile
        from checkpoints import valid_audio
        with tempfile.TemporaryDirectory(prefix='pdf-to-audio-narration-test-') as directory:
            source = Path(directory) / 'test.md'
            source.write_text('Table 1. Robot comparison.\n\n| Platform | Value |\n|---|---|\n'
                              + '| hidden_table_data | 12345 |\n' * 40
                              + '\nHello. This is an installation test.', encoding='utf-8')
            events = []
            run_batch(dict(config, output=str(Path(directory) / 'audio')), [str(source)],
                      lambda event, **data: events.append(dict(event=event, **data)))
            assert events[-1] == dict(event='finished', completed=1, failed=0), events
            folder = Path(next(event['folder'] for event in events if event['event'] == 'done'))
            assert (folder / 'cleanup.json').is_file(), 'Markdown must pass through cleanup.'
            assert 'hidden_table_data' not in (folder / 'narration.txt').read_text(encoding='utf-8')
            assert valid_audio(folder / 'audio.flac')
        speaker = Speaker(config)
        try:
            count = speaker.batch_size
            rendered = speaker.speak_batch([dict(text='Hello. This is a narration test.', language='English')] * count)
            assert len(rendered) == count
            assert all(len(wave) > rate / 10 for wave, rate in rendered)
        finally:
            speaker.close()
            release_gpu()
        print(f'Offline {config["device"]} cleanup and speech passed.', flush=True)

if __name__ == '__main__':
    if getattr(sys, 'frozen', False):
        from runtime_setup import main
        sys.exit(main())
    if any(flag in sys.argv for flag in ('--check', '--self-test', '--check-cuda', '--check-metal', '--self-test-gpu')):
        check(inference='--self-test' in sys.argv, cuda='--check-cuda' in sys.argv,
              metal='--check-metal' in sys.argv, gpu='--self-test-gpu' in sys.argv)
    elif '--setup-models' in sys.argv:
        from model_store import DEFAULT_MODELS, ensure_models
        ensure_models(DEFAULT_MODELS, lambda event, **data: print(data['message'], flush=True))
    else:
        from worker import main
        sys.exit(main())
