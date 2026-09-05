import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from checkpoints import fingerprint, valid_audio
from core import speech_plan
from worker import Cancelled, Cleaner, Speaker, run_batch


class ResumeTests(unittest.TestCase):
    def test_cancel_then_resume_reuses_completed_passages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'book.md'
            source.write_text('This is a complete sentence for narration. ' * 55)
            config = dict(output=str(root / 'audio'))
            events = []
            emit = lambda event, **data: events.append(dict(event=event, **data))
            speaker = Mock(batch_size=1, voice_language='English')
            wave = np.ones(2400, dtype=np.float32) * 0.1
            speaker.speak.side_effect = [(wave, 24000), (wave, 24000), Cancelled()]
            with patch('worker.Speaker', return_value=speaker), patch('worker.release_gpu'):
                with self.assertRaises(Cancelled):
                    run_batch(config, [str(source)], emit)
            folder = Path(next(e['folder'] for e in events if e.get('folder')))
            first = folder / 'passages/000001.flac'
            before = first.read_bytes(), first.stat().st_mtime_ns
            total = len(json.loads((folder / 'passages.json').read_text()))
            speaker.speak.side_effect = None
            speaker.speak.return_value = (wave, 24000)
            speaker.speak.reset_mock()
            with patch('worker.Speaker', return_value=speaker), patch('worker.release_gpu'):
                run_batch(config, [str(source)], emit, {str(source): str(folder)})
            self.assertEqual(speaker.speak.call_count, total - 2)
            self.assertEqual((first.read_bytes(), first.stat().st_mtime_ns), before)
            self.assertTrue(valid_audio(folder / 'audio.flac'))
            self.assertEqual(json.loads((folder / 'job.json').read_text())['status'], 'complete')
            source.write_text('The document has changed.')
            with patch('worker.Speaker', side_effect=AssertionError('Changed input must not render')), patch('worker.release_gpu'):
                run_batch(config, [str(source)], emit, {str(source): str(folder)})
            self.assertEqual(events[-1]['failed'], 1)
            self.assertTrue(any('Cannot resume' in e.get('message', '') for e in events))

    def test_cleanup_excerpts_survive_cancellation(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / 'cleanup.json'
            cleaner = Cleaner.__new__(Cleaner)
            cleaner.clean_part = Mock(side_effect=['First output.', Cancelled()])
            text = 'Complete source sentence. ' * 180
            with self.assertRaises(Cancelled):
                cleaner.clean(text, lambda *_: None, 'English', checkpoint)
            self.assertEqual(len(json.loads(checkpoint.read_text())), 1)
            cleaner.clean_part = Mock(return_value='Later output.')
            result = cleaner.clean(text, lambda *_: None, 'English', checkpoint)
            self.assertTrue(result.startswith('First output.'))
            self.assertEqual(cleaner.clean_part.call_count, len(json.loads(checkpoint.read_text())) - 1)

    def test_batch_oom_retries_smaller_and_preserves_order(self):
        import torch
        speaker = Speaker.__new__(Speaker)
        speaker.batch_size, speaker.prompt = 6, ['voice']
        speaker.model = Mock()
        wave = np.ones(2400, dtype=np.float32)
        speaker.model.generate_voice_clone.side_effect = [torch.OutOfMemoryError('test'), ([wave.copy()], 24000), ([wave.copy()], 24000)]
        passages = [dict(text='First sentence.', language='English'), dict(text='Second sentence.', language='French')]
        with patch('worker.release_gpu'):
            result = speaker.speak_batch(passages)
        self.assertEqual(len(result), 2)
        self.assertEqual(speaker.batch_size, 1)
        calls = speaker.model.generate_voice_clone.call_args_list
        self.assertEqual(calls[0].kwargs['text'], ['First sentence.', 'Second sentence.'])
        self.assertEqual(calls[1].kwargs['text'], 'First sentence.')
        self.assertEqual(calls[2].kwargs['language'], 'French')


if __name__ == '__main__':
    unittest.main()
