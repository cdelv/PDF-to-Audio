"""Compare speech batch throughput on CPU or CUDA; no source documents changed."""
import argparse
import json
from pathlib import Path
import sys
import time
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 2, 6])
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1 or args.repeat < 1 or any(size < 1 or size > 6 for size in args.batches):
        parser.error('Use positive thread/repeat counts and batch sizes from 1 to 6.')
    import torch
    import soundfile as sf
    import psutil
    from core import ROOT, defaults
    from checkpoints import atomic_json, valid_audio
    from worker import Speaker, release_gpu
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise SystemExit('This benchmark requires CUDA.')
    torch.set_num_threads(args.threads)
    config = defaults()
    config.update(tts='Qwen/Qwen3-TTS-12Hz-0.6B-Base', device=args.device,
                  voice=str(ROOT / 'assets/voice.wav'), transcript=str(ROOT / 'assets/transcript.txt'))
    parent = ROOT / 'test-output/cpu-batching'
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='benchmark-', dir=parent))
    print('Output:', output, 'Device:', args.device, 'Threads:', torch.get_num_threads(), flush=True)
    speaker = Speaker(config)
    torch.set_num_threads(args.threads)
    assert speaker.model.model.device.type == args.device.split(':')[0]
    weights = next(speaker.model.model.parameters()).data_ptr()
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
    schedule = args.batches * args.repeat
    for number, size in enumerate(schedule):
        torch.manual_seed(42)
        release_gpu()
        speaker.batch_size = size
        if args.device != 'cpu':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start, duration = time.monotonic(), 0
        for offset in range(0, len(texts), size):
            batch = [dict(text=text, language='English') for text in texts[offset:offset + size]]
            rendered = speaker.speak_batch(batch)
            assert next(speaker.model.model.parameters()).data_ptr() == weights
            assert speaker.batch_size == size, 'Benchmark fell back to a smaller batch.'
            assert len(rendered) == len(batch)
            for index, (wave, rate) in enumerate(rendered, offset):
                target = output / f'run-{number}-batch-{size}-passage-{index}.flac'
                sf.write(target, wave, rate)
                assert valid_audio(target)
                duration += len(wave) / rate
            del rendered, wave
            release_gpu()
        if args.device != 'cpu':
            torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        result = dict(batch=size, threads=torch.get_num_threads(), device=args.device, model=config['tts'],
                      elapsed_seconds=elapsed, audio_seconds=duration,
                      audio_seconds_per_second=duration / elapsed,
                      rss_gib=psutil.Process().memory_info().rss / 2**30)
        results.append(result)
        print(json.dumps(result), flush=True)
        atomic_json(output / 'report.json', results)
    speaker.close()
    release_gpu()


if __name__ == '__main__':
    main()
