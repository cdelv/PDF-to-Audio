import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hardware import batch_size
from worker import Cancelled, Cleaner, Speaker, model_options, run_batch, serial_audio_decoder


class GpuLifecycleTests(unittest.TestCase):
    def test_cpu_uses_all_available_threads(self):
        for count, expected in ((16, 16), (2, 2), (None, 1)):
            with patch('os.cpu_count', return_value=count), patch('torch.set_num_threads') as setter:
                self.assertEqual(model_options({'device': 'cpu'})['device_map'], 'cpu')
                setter.assert_called_once_with(expected)

    def test_fixed_batch_size_with_only_the_small_gpu_exception(self):
        self.assertEqual(batch_size(), 6)
        self.assertEqual(batch_size(4 * 2**30 - 1), 1)
        self.assertEqual(batch_size(4 * 2**30), 6)
        self.assertEqual(batch_size(12 * 2**30), 6)

    def test_serial_decoder_preserves_order_without_reference_cycle(self):
        class Tokenizer:
            def decode(self, codes):
                self.calls.append(codes)
                return [codes[0]], 24000

        tokenizer = Tokenizer()
        tokenizer.calls = []
        reference = weakref.ref(tokenizer)
        tokenizer.decode = serial_audio_decoder(tokenizer)
        self.assertEqual(tokenizer.decode([1, 2, 3]), ([1, 2, 3], 24000))
        self.assertEqual(tokenizer.calls, [[1], [2], [3]])
        del tokenizer
        self.assertIsNone(reference())

    def test_close_drops_weights_and_voice_prompt(self):
        class Payload:
            pass

        cleaner, speaker = Cleaner.__new__(Cleaner), Speaker.__new__(Speaker)
        cleaner.model = Payload()
        cleaner.tokenizer = Payload()
        speaker.model = Payload()
        speaker.prompt = Payload()
        references = [weakref.ref(value) for value in
                      (cleaner.model, cleaner.tokenizer, speaker.model, speaker.prompt)]
        cleaner.close()
        speaker.close()
        self.assertTrue(all(reference() is None for reference in references))

    def test_models_do_not_overlap_and_close_on_success_failure_or_cancel(self):
        for failure in (None, RuntimeError, Cancelled):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                events = []
                root = Path(temp)
                files = [root / 'first.pdf', root / 'second.pdf']
                for path in files:
                    path.write_bytes(b'PDF placeholder; extraction is mocked.')

                class FakeCleaner:
                    warnings = []

                    def __init__(self, config):
                        events.append('load cleanup')

                    def clean(self, text, *args, **kwargs):
                        events.append('clean')
                        return text

                    def close(self):
                        events.append('close cleanup')

                class FakeSpeaker:
                    batch_size = 6
                    voice_language = 'English'

                    def __init__(self, config):
                        self.closed = 'close cleanup' in events
                        if not self.closed:
                            raise AssertionError('Cleanup must unload before speech loads.')
                        events.append('load speech')

                    def speak_batch(self, passages):
                        events.append('speak')
                        if failure:
                            raise failure('test')
                        return [(np.ones(2400, dtype=np.float32), 24000) for _ in passages]

                    def close(self):
                        events.append('close speech')

                with patch('worker.Cleaner', FakeCleaner), patch('worker.Speaker', FakeSpeaker), \
                     patch('pdf_input.extract_pdf', return_value='This is a sentence for narration. ' * 60), \
                     patch('worker.release_gpu', side_effect=lambda: events.append('release GPU')):
                    try:
                        run_batch(dict(output=str(root / 'audio')), [str(p) for p in files], lambda *_args, **_kw: None)
                    except Cancelled:
                        self.assertIs(failure, Cancelled)
                self.assertEqual(events.count('load cleanup'), 1)
                self.assertEqual(events.count('close cleanup'), 1)
                self.assertEqual(events.count('load speech'), 1)
                self.assertEqual(events.count('close speech'), 1)
                self.assertEqual(events[-2:], ['close speech', 'release GPU'])
                closed = events.index('close cleanup')
                self.assertEqual(events[closed:closed + 3], ['close cleanup', 'release GPU', 'load speech'])


if __name__ == '__main__':
    unittest.main()
