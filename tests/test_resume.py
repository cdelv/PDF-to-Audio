import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from checkpoints import atomic_json, digest, fingerprint, open_job, valid_audio
from worker import Cancelled, Cleaner, Speaker, run_batch


class ResumeTests(unittest.TestCase):
    def test_cancel_then_resume_reuses_completed_passages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'book.md'
            source.write_text('This is a complete sentence for narration. ' * 55)
            config = dict(output=str(root / 'audio'), llm='Qwen/Qwen3-1.7B',
                          tts='Qwen/Qwen3-TTS-12Hz-1.7B-Base')
            events = []
            emit = lambda event, **data: events.append(dict(event=event, **data))
            speaker = Mock(batch_size=1, voice_language='English')
            wave = np.ones(2400, dtype=np.float32) * 0.1
            speaker.speak_batch.side_effect = [[(wave, 24000)], [(wave, 24000)], Cancelled()]
            cleaner = Mock(warnings=[])
            cleaner.clean.side_effect = lambda text, *_a, **_k: text
            with patch('worker.Speaker', return_value=speaker), patch('worker.release_gpu'), \
                 patch('worker.Cleaner', return_value=cleaner):
                with self.assertRaises(Cancelled):
                    run_batch(config, [str(source)], emit)
            folder = Path(next(e['folder'] for e in events if e.get('folder')))
            first = folder / 'passages/000001.flac'
            before = first.read_bytes(), first.stat().st_mtime_ns
            total = len(json.loads((folder / 'passages.json').read_text()))
            speaker.speak_batch.side_effect = None
            speaker.speak_batch.return_value = [(wave, 24000)]
            speaker.speak_batch.reset_mock()
            config.update(llm='Qwen/Qwen3-0.6B', tts='Qwen/Qwen3-TTS-12Hz-0.6B-Base')
            with patch('worker.Speaker', return_value=speaker) as load_speech, patch('worker.release_gpu'), \
                 patch('worker.Cleaner', side_effect=AssertionError('Resume must reuse prepared narration')):
                run_batch(config, [str(source)], emit, {str(source): str(folder)})
            load_speech.assert_called_once_with(config)
            self.assertEqual(speaker.speak_batch.call_count, total - 2)
            self.assertEqual((first.read_bytes(), first.stat().st_mtime_ns), before)
            self.assertTrue(valid_audio(folder / 'audio.flac'))
            self.assertEqual(json.loads((folder / 'job.json').read_text())['status'], 'complete')
            source.write_text('The document has changed.')
            with patch('worker.Speaker', side_effect=AssertionError('Changed input must not render')), patch('worker.release_gpu'):
                run_batch(config, [str(source)], emit, {str(source): str(folder)})
            self.assertEqual(events[-1]['failed'], 1)
            self.assertTrue(any('Cannot resume' in e.get('message', '') for e in events))

    def test_model_switch_during_cleanup_reuses_saved_excerpts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'book.md'
            source.write_text('Complete source sentence. ' * 650)
            config = dict(output=str(root / 'audio'), llm='Qwen/Qwen3-1.7B',
                          tts='Qwen/Qwen3-TTS-12Hz-1.7B-Base')
            events = []
            def emit(event, **data):
                events.append(dict(event=event, **data))
            cleaner = Cleaner.__new__(Cleaner)
            cleaner.batch_size = 6
            calls = 0
            def clean(parts, language):
                nonlocal calls
                calls += 1
                if calls > 1:
                    raise Cancelled()
                return [(part, []) for part in parts]
            cleaner.clean_batch = clean
            with patch('worker.Cleaner', return_value=cleaner), patch('worker.release_gpu'):
                with self.assertRaises(Cancelled):
                    run_batch(config, [str(source)], emit)
            folder = Path(next(e['folder'] for e in events if e.get('folder')))
            saved = json.loads((folder / 'cleanup.json').read_text())
            self.assertEqual(len(saved), 6)
            config.update(llm='Qwen/Qwen3-0.6B', tts='Qwen/Qwen3-TTS-12Hz-0.6B-Base')
            replacement = Cleaner.__new__(Cleaner)
            replacement.batch_size = 6
            replacement.clean_batch = Mock(side_effect=lambda parts, _language: [(part, []) for part in parts])
            speaker = Mock(batch_size=6, voice_language='English')
            speaker.speak_batch.side_effect = lambda parts: [(np.ones(2400, dtype=np.float32) * 0.1, 24000) for _ in parts]
            with patch('worker.Cleaner', return_value=replacement) as load, \
                 patch('worker.Speaker', return_value=speaker), patch('worker.release_gpu'):
                run_batch(config, [str(source)], emit, {str(source): str(folder)})
            load.assert_called_once_with(config)
            replacement.clean_batch.assert_called_once()
            self.assertEqual(json.loads((folder / 'cleanup.json').read_text())[:6], saved)
            self.assertEqual(events[-1], dict(event='finished', completed=1, failed=0))

    def test_legacy_model_switch_migrates_without_weakening_input_checks(self):
        import hashlib
        from model_store import MODELS
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source.txt'
            source.write_text('Original document.')
            config = dict(llm='Qwen/Qwen3-1.7B', tts='Qwen/Qwen3-TTS-12Hz-1.7B-Base',
                          document_language='Auto', voice_language='English')
            for name in ('voice', 'transcript', 'prompt'):
                path = root / name
                path.write_text('Original ' + name)
                config[name] = str(path)
            # Reproduce the previous release's hash independently of migration code.
            old = dict(version=1, source=digest(source))
            for key in ('llm', 'tts', 'voice_language', 'document_language'):
                old[key] = config[key]
            for key in ('llm', 'tts'):
                old[key + '_revision'] = MODELS[config[key]]['revision']
            for key in ('voice', 'transcript', 'prompt'):
                old[key] = digest(config[key])
            legacy = dict(version=1, fingerprint=hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest(),
                          source=str(source), status='ready', plan_digest='saved-plan')
            changed = dict(config, llm='Qwen/Qwen3-0.6B', tts='Qwen/Qwen3-TTS-12Hz-0.6B-Base')
            atomic_json(root / 'job.json', legacy)
            upgraded = open_job(root, source, changed, resume=True)
            self.assertEqual(upgraded['version'], 2)
            self.assertEqual(upgraded['plan_digest'], 'saved-plan')
            self.assertEqual(upgraded['fingerprint'], fingerprint(source, changed))
            self.assertEqual(open_job(root, source, config, resume=True), upgraded)
            for saved_job in (legacy, upgraded):
                for key in ('voice', 'transcript', 'prompt', 'document_language', 'voice_language', 'source'):
                    with self.subTest(version=saved_job['version'], key=key):
                        atomic_json(root / 'job.json', saved_job)
                        bad = dict(changed)
                        altered = root / 'altered'
                        altered.write_text('Different content.')
                        target = source
                        if key in ('voice', 'transcript', 'prompt'):
                            bad[key] = str(altered)
                        elif key == 'source':
                            target = altered
                        else:
                            bad[key] = 'Spanish'
                        before = (root / 'job.json').read_bytes()
                        with self.assertRaisesRegex(ValueError, 'Cannot resume'):
                            open_job(root, target, bad, resume=True)
                        self.assertEqual((root / 'job.json').read_bytes(), before)

    def test_cleanup_excerpts_survive_cancellation(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / 'cleanup.json'
            cleaner = Cleaner.__new__(Cleaner)
            cleaner.batch_size = 6
            cleaner.clean_batch = Mock(side_effect=[[('First output.', [])] * 6, Cancelled()])
            text = 'Complete source sentence. ' * 650
            with self.assertRaises(Cancelled):
                cleaner.clean(text, lambda *_: None, 'English', checkpoint)
            self.assertEqual(len(json.loads(checkpoint.read_text())), 6)
            cleaner.clean_batch = Mock(side_effect=lambda parts, _language: [('Later output.', []) for _ in parts])
            result = cleaner.clean(text, lambda *_: None, 'English', checkpoint)
            self.assertTrue(result.startswith('First output.'))
            remaining = len(json.loads(checkpoint.read_text())) - 6
            self.assertEqual(cleaner.clean_batch.call_count, (remaining + 5) // 6)

    def test_memory_errors_do_not_change_the_batch_size(self):
        import torch
        for error in (torch.OutOfMemoryError('test'), MemoryError('test'), RuntimeError('test')):
            speaker = Speaker.__new__(Speaker)
            speaker.batch_size, speaker.prompt = 6, ['voice']
            speaker.model = Mock()
            speaker.model.generate_voice_clone.side_effect = error
            passages = [dict(text='Sentence.', language='English')] * 6
            with self.assertRaises(type(error)):
                speaker.speak_batch(passages)
            self.assertEqual(speaker.batch_size, 6)
            self.assertEqual(speaker.model.generate_voice_clone.call_count, 1)


if __name__ == '__main__':
    unittest.main()
