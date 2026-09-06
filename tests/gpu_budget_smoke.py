"""Narrate short Markdown files on CUDA under a simulated 4 GiB budget.

Keeps 512 MiB outside PyTorch's allocator for the CUDA context/libraries.
This is a budget test on the installed GPU, not physical 4 GiB hardware.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocator-gib', type=float, choices=(3.5, 4.0), default=3.5,
                        help='3.5 leaves room for CUDA overhead; 4.0 tests the full 4 GiB allocator ceiling.')
    parser.add_argument('--check-cleanup', action='store_true',
                        help='Also verify cleanup-model unloading before speech generation.')
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    files = [str(Path(name).resolve()) for name in args.files]
    if any(Path(name).suffix.lower() != '.md' or not Path(name).is_file() for name in files):
        parser.error('Pass existing short Markdown files.')

    import torch
    import worker
    from checkpoints import atomic_json, valid_audio
    from core import ROOT, defaults

    assert torch.cuda.is_available(), 'This test requires CUDA; CPU fallback is forbidden.'
    budget = int(args.allocator_gib * 2**30)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    torch.manual_seed(42)
    parent = ROOT / 'test-output/gpu-budget'
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='4gib-', dir=parent))
    config = defaults()
    config.update(output=str(output), device='cuda:0',
                  tts='Qwen/Qwen3-TTS-12Hz-0.6B-Base',
                  voice=str(ROOT / 'assets/voice.wav'),
                  transcript=str(ROOT / 'assets/transcript.txt'), prompt=str(ROOT / 'assets/prompt.txt'))
    events, samples, failures = [], [], []
    stop = threading.Event()
    started = time.monotonic()
    loads = 0

    def monitor():
        while not stop.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                         '--format=csv,noheader,nounits'], capture_output=True,
                                        text=True, check=True, timeout=5)
                for line in result.stdout.splitlines():
                    pid, memory = line.split(',', 1)
                    if int(pid.strip()) == os.getpid():
                        samples.append(float(memory.strip()))
            except Exception as error:
                failures.append(str(error))
            stop.wait(0.5)

    class CheckedSpeaker(worker.Speaker):
        def __init__(self, config):
            nonlocal loads
            loads += 1
            super().__init__(config)
            assert self.model.model.device.type == 'cuda'
            assert self.batch_size == 6
            self.weight_address = next(self.model.model.parameters()).data_ptr()

        def speak_batch(self, passages):
            assert next(self.model.model.parameters()).data_ptr() == self.weight_address
            result = super().speak_batch(passages)
            assert self.batch_size == 6
            return result

        def speak(self, *args, **kwargs):
            assert next(self.model.model.parameters()).data_ptr() == self.weight_address
            return super().speak(*args, **kwargs)

    def emit(event, **data):
        torch.cuda.synchronize()
        row = dict(event=event, elapsed=round(time.monotonic() - started, 2),
                   allocated_MiB=round(torch.cuda.memory_allocated() / 2**20, 1),
                   reserved_MiB=round(torch.cuda.memory_reserved() / 2**20, 1), **data)
        events.append(row)
        print(json.dumps(row), flush=True)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        if args.check_cleanup:
            cleaner = worker.Cleaner(config)
            model = weakref.ref(cleaner.model)
            assert cleaner.clean('This is a test of document narration. Keep the original language.',
                                 lambda *_: None, 'English').strip()
            cleaner.close()
            del cleaner
            worker.release_gpu()
            assert model() is None, 'Cleanup weights are still referenced.'
            assert torch.cuda.memory_allocated() < 64 * 2**20
            emit('cleanup_unloaded')
        with patch('worker.Speaker', CheckedSpeaker):
            worker.run_batch(config, files, emit)
    finally:
        stop.set()
        thread.join(timeout=6)
        report = dict(model=config['tts'], files=files, model_loads=loads, batch_size=6,
                      serial_decode=True, cleanup_checked=args.check_cleanup,
                      allocator_limit_MiB=budget / 2**20,
                      peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
                      peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20,
                      sampled_process_peak_MiB=max(samples, default=0),
                      monitor_errors=failures, events=events)
        atomic_json(output / 'report.json', report)
        print('Report:', output / 'report.json', flush=True)
    assert loads == 1, report
    assert samples and not failures, report
    assert max(samples) <= 4096, report
    assert report['peak_reserved_MiB'] <= budget / 2**20, report
    assert events[-1]['completed'] == len(files) and events[-1]['failed'] == 0, report
    assert events[-1]['allocated_MiB'] < 64, 'Speech weights were not unloaded.'
    assert all(valid_audio(e['audio']) for e in events if e['event'] == 'done')
    print('4 GiB budget test passed; all final audio files verified.', flush=True)


if __name__ == '__main__':
    main()
