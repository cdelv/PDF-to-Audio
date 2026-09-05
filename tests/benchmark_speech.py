"""Compare sequential and six-item voice cloning on synthetic text; no PDFs."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from core import ROOT, defaults
    from worker import Speaker
    if not torch.cuda.is_available():
        raise SystemExit('This benchmark requires CUDA.')
    config = defaults()
    config.update(tts='Qwen/Qwen3-TTS-12Hz-1.7B-Base', device='cuda:0',
                  voice=str(ROOT / 'assets/voice.wav'), transcript=str(ROOT / 'assets/transcript.txt'))
    speaker = Speaker(config)
    speaker.speak('This is a short warm up.', 'English')
    texts = [
        'The morning sun warmed the quiet garden. A small bird landed near the window and began to sing.',
        'We opened the book and read the first chapter. It described a journey through a distant mountain village.',
        'The experiment used several small robots. Each robot moved along a boundary while a camera recorded its path.',
        'A gentle wind moved across the field. Clouds gathered slowly above the hills as the afternoon came to an end.',
        'The table caption explains the measurements. The surrounding paragraph describes how the results were obtained.',
        'At the end of the day, everyone returned home. They shared a meal and talked about the things they had learned.',
    ]
    results = []
    for size in (1, 6):
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start, duration = time.monotonic(), 0
        for offset in range(0, len(texts), size):
            batch = [dict(text=text, language='English') for text in texts[offset:offset + size]]
            duration += sum(len(wave) / rate for wave, rate in speaker.speak_batch(batch))
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        result = dict(batch=size, elapsed_seconds=elapsed, audio_seconds=duration,
                      audio_seconds_per_second=duration / elapsed,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        results.append(result)
        print(json.dumps(result), flush=True)
    print('Elapsed speedup:', results[0]['elapsed_seconds'] / results[1]['elapsed_seconds'], flush=True)
    target = ROOT / 'test-output/batching-benchmark.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
