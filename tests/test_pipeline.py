import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace

import core


class TextTests(unittest.TestCase):
    def test_cleanup_can_split_malformed_pdf_without_losing_text(self):
        for text in ('word ' * 1000, 'x' * 5000, 'Author A, Author B,\n' * 500):
            chunks = core.cleanup_chunks(text)
            self.assertEqual(''.join(chunks), text)
            self.assertTrue(all(len(chunk) <= 2000 for chunk in chunks))

    def test_markdown_hard_breaks_remain_speech_boundaries(self):
        text = 'Long author list  \nAnother author list  \nLast authors'
        self.assertEqual(core.plain_text(text), 'Long author list\n\nAnother author list\n\nLast authors')

    def test_tables_omit_data_but_keep_captions(self):
        source = ('Figure 1. Setup.\n\nTable 1. Results.\n\n'
                  '| Method | Score |\n| --- | ---: |\n| Zebra | 98765 |\n\n'
                  'Surrounding prose.\n<table><caption>Tabla 2. Más resultados.</caption>'
                  '<tr><td>Otter</td><td>54321</td></tr></table>\nFinal prose.')
        result = core.omit_tables(source)
        for caption in ('Figure 1. Setup.', 'Table 1. Results.', 'Tabla 2. Más resultados.',
                        'Surrounding prose.', 'Final prose.'):
            self.assertIn(caption, result)
        for data in ('Method', 'Score', 'Zebra', '98765', 'Otter', '54321', '<table>'):
            self.assertNotIn(data, result)

    def test_table_filter_preserves_pipe_prose(self):
        text = 'Choose A | B.\nThis is ordinary prose.\n'
        self.assertEqual(core.omit_tables(text), text)

    def test_long_document_is_lossless_and_bounded(self):
        text = ('A complete sentence with several words. Another sentence ends here.\n\n' * 8000) + 'Final words'
        sections = core.split_text(text, 100_000)
        self.assertGreater(len(sections), 5)
        self.assertEqual(''.join(sections), text)
        self.assertTrue(all(len(part) <= 100_000 for part in sections))
        plan = core.speech_plan(text)
        self.assertEqual(''.join(p['text'] for p in plan), text)
        self.assertTrue(all(len(p['text']) <= core.SPEECH_CHARS for p in plan))
        self.assertTrue(all(p['text'].rstrip().endswith('.') for p in plan[:-1]))

    def test_paragraphs_quotes_and_decimals(self):
        for text in ('No punctuation here\n\nNew paragraph here\n\nLast paragraph',
                     'The value is 3.14. "That is correct." Next sentence.',
                     'One! Two? Three.\n\nFour.', '第一句话。 第二句话。 最后一句。'):
            chunks = core.split_text(text, 28)
            self.assertEqual(''.join(chunks), text)
            self.assertTrue(all(len(c) <= 28 for c in chunks))

    def test_cjk_sentences_without_spaces(self):
        text = '这是第一句话。第二句话没有空格。第三句话也没有空格。' * 100
        chunks = core.split_text(text, 40)
        self.assertEqual(''.join(chunks), text)
        self.assertTrue(all(c.endswith('。') for c in chunks))

    def test_abbreviation_is_not_a_sentence_end(self):
        text = 'Meet Dr. Smith. Next sentence.'
        self.assertEqual(core.split_text(text, 18), ['Meet Dr. Smith.', ' Next sentence.'])

    def test_no_silent_mid_sentence_cut(self):
        with self.assertRaisesRegex(ValueError, 'sentence or paragraph exceeds'):
            core.split_text('word ' * 200, 100)
        with self.assertRaises(ValueError):
            core.split_text('Hi.', 0)
        self.assertEqual(core.split_text(' \n ', 100), [])

    def test_markdown_does_not_read_link_targets(self):
        self.assertEqual(core.plain_text('# Title\n\n**Hello** [reader](https://example.com).'), 'Title\n\nHello reader.')


class HardwareTests(unittest.TestCase):
    def test_model_labels_on_four_gib_card(self):
        from hardware import MODEL_VRAM, model_label
        for name, estimate in MODEL_VRAM.items():
            text, red = model_label(name, 4.0)
            self.assertIn(f'~{estimate:g} GiB VRAM', text)
            self.assertEqual(red, estimate > 4.0)
        self.assertEqual(model_label('custom/model', 4.0), ('custom/model — VRAM unknown', False))
        self.assertFalse(model_label('Qwen/Qwen3-1.7B', None)[1])

    def test_automatic_cuda_selection_and_older_cuda_dtype(self):
        import torch
        from worker import model_options
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.is_bf16_supported', return_value=False):
            options = model_options({'device': 'auto'})
        self.assertEqual(options['device_map'], 'cuda:0')
        self.assertEqual(options['dtype'], torch.float16)


class PipelineTests(unittest.TestCase):
    def test_speech_cap_is_reported_without_accepting_truncation_or_retrying(self):
        from worker import Speaker
        speaker = Speaker.__new__(Speaker)
        speaker.prompt = 'reference'
        generate = Mock(side_effect=ValueError('Speech reached its generation limit.'))
        speaker.model = SimpleNamespace(generate_voice_clone=generate)
        text = 'The first sentence is complete. The second sentence is complete.'
        with self.assertRaisesRegex(ValueError, 'generation limit'):
            speaker.speak(text, 'English')
        self.assertEqual(generate.call_count, 1)

    def test_failed_cleanup_keeps_source_with_a_warning(self):
        import torch
        from transformers import BatchEncoding
        from worker import Cleaner
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = 'chat'
        tokenizer.encode.return_value = [1, 2]
        tokenizer.return_value = BatchEncoding({'input_ids': torch.tensor([[1]])})
        tokenizer.decode.return_value = 'Esta es una historia sobre una biblioteca. Cada libro nos invita a descubrir nuevas ideas y aventuras.'
        cleaner = Cleaner.__new__(Cleaner)
        cleaner.tokenizer = tokenizer
        cleaner.prompt = 'Preserve original language.'
        cleaner.model = SimpleNamespace(device='cpu', config=SimpleNamespace(max_position_embeddings=40960),
            generation_config=SimpleNamespace(eos_token_id=0), generate=Mock(return_value=torch.tensor([[1, 2, 0]])))
        source = 'The source must remain in its original language.'
        self.assertEqual(cleaner.clean(source, lambda *_: None, 'English'), source)
        self.assertEqual(len(cleaner.warnings), 1)

    def test_stitch_preserves_samples_and_order(self):
        import numpy as np
        import soundfile as sf
        from worker import stitch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b, final = root/'a.flac', root/'b.flac', root/'final.flac'
            sf.write(a, np.full(1234, 0.25), 24000)
            sf.write(b, np.full(2345, -0.25), 24000)
            stitch([a, b], final)
            wave, rate = sf.read(final)
            self.assertEqual(rate, 24000)
            self.assertEqual(len(wave), 3579)
            self.assertTrue(np.all(wave[:1234] == 0.25))
            self.assertTrue(np.all(wave[1234:] == -0.25))

    def test_batch_skips_llm_for_text_and_continues_after_failure(self):
        import numpy as np
        from worker import run_batch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = root/'book.md'
            text.write_text('# Test\n\nHello there. This is a sentence.')
            bad = root/'bad.txt'
            bad.write_bytes(b'\xff\xfe\x00')
            events = []
            with patch('worker.Cleaner', side_effect=AssertionError('LLM must not run')), \
                 patch('worker.Speaker') as speaker, patch('worker.release_gpu'):
                speaker.return_value.batch_size = 6
                speaker.return_value.speak_batch.return_value = [(np.ones(2400, dtype=np.float32) * 0.1, 24000)]
                speaker.return_value.voice_language = 'English'
                run_batch(dict(output=str(root/'output')), [str(bad), str(text), str(text)],
                          lambda event, **data: events.append(dict(event=event, **data)))
            self.assertEqual(events[-1], dict(event='finished', completed=2, failed=1))
            outputs = [e for e in events if e['event'] == 'done']
            self.assertNotEqual(outputs[0]['folder'], outputs[1]['folder'])
            for event in outputs:
                self.assertTrue(Path(event['audio']).is_file())
                plan = json.loads((Path(event['folder'])/'passages.json').read_text())
                self.assertEqual(''.join(p['text'] for p in plan), 'Test\n\nHello there. This is a sentence.')


class LanguageTests(unittest.TestCase):
    examples = {
        'English': 'A quiet morning is a good time to read. Every page brings a new idea and a new adventure.',
        'Spanish': 'Esta es una historia sobre una biblioteca. Cada libro nos invita a descubrir nuevas ideas y aventuras.',
        'French': 'Le matin est un bon moment pour lire. Chaque livre nous invite à découvrir de nouvelles idées.',
        'German': 'Dies ist eine Geschichte über eine Bibliothek. Jedes Buch bringt neue Gedanken und spannende Abenteuer.',
        'Portuguese': 'Esta é uma história sobre uma biblioteca. Cada livro nos convida a descobrir novas ideias e aventuras.',
        'Italian': 'Questa è una storia su una biblioteca. Ogni libro ci invita a scoprire nuove idee e nuove avventure.',
        'Russian': 'Это история о библиотеке. Каждая книга приглашает нас открыть новые идеи и отправиться в путешествие.',
        'Japanese': '図書館にはたくさんの本があります。新しい本を読むことで、私たちは知らなかった世界を発見することができます。',
        'Chinese': '图书馆里有许多有趣的书籍。阅读可以帮助我们了解不同的文化，也可以让我们发现新的想法和故事。',
        'Korean': '도서관에는 재미있는 책이 많이 있습니다. 새로운 책을 읽으면 다양한 문화를 배우고 새로운 생각을 발견할 수 있습니다.',
    }

    def test_detect_supported_languages(self):
        from languages import resolve_language
        for name, text in self.examples.items():
            with self.subTest(name=name):
                self.assertEqual(resolve_language(text), name)

    def test_ambiguous_and_unsupported(self):
        from languages import resolve_language
        self.assertEqual(resolve_language('12345'), 'Auto')
        self.assertEqual(resolve_language('Hello'), 'Auto')
        self.assertEqual(resolve_language('Hello', 'English'), 'English')
        with self.assertRaisesRegex(ValueError, 'does not support'):
            resolve_language('هذه قصة عن مكتبة جميلة. تحتوي المكتبة على العديد من الكتب التي تساعدنا على اكتشاف أفكار جديدة وثقافات مختلفة.')
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            resolve_language('Hello', 'Klingon')

    def test_reference_is_independent_and_cleanup_cannot_translate(self):
        from languages import check_cleanup_language, resolve_language
        self.assertEqual(resolve_language(self.examples['English'], 'English', 'Voice sample transcript'), 'English')
        self.assertEqual(resolve_language(self.examples['Spanish']), 'Spanish')
        with self.assertRaisesRegex(ValueError, 'Voice sample transcript appears'):
            resolve_language(self.examples['English'], 'French', 'Voice sample transcript')
        with self.assertRaisesRegex(ValueError, 'changed the language'):
            check_cleanup_language('Spanish', self.examples['English'])

    def test_mixed_batch_passes_each_documents_language_to_speech(self):
        import numpy as np
        from worker import run_batch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for name in ('English', 'Spanish', 'French'):
                path = root / (name + '.txt')
                path.write_text(self.examples[name])
                files.append(str(path))
            events = []
            with patch('worker.Cleaner', side_effect=AssertionError('Text must bypass LLM')), \
                 patch('worker.Speaker') as speaker, patch('worker.release_gpu'):
                speaker.return_value.voice_language = 'English'
                speaker.return_value.batch_size = 6
                speaker.return_value.speak_batch.return_value = [(np.ones(100, dtype=np.float32)*0.1, 24000)]
                run_batch(dict(output=str(root/'out'), document_language='Auto', voice_language='English'), files,
                          lambda event, **data: events.append(dict(event=event, **data)))
                self.assertEqual([c.args[0][0]['language'] for c in speaker.return_value.speak_batch.call_args_list],
                                 ['English', 'Spanish', 'French'])
            self.assertEqual(events[-1], dict(event='finished', completed=3, failed=0))

    def test_existing_english_default_migrates_to_per_document_auto(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'settings.json'
            path.write_text(json.dumps(dict(language='English', output='/keep/my/output', python='/untrusted/python')))
            with patch.object(core, 'CONFIG', path), patch.object(core, 'DATA', root):
                config = core.load_settings()
                self.assertEqual(config['document_language'], 'Auto')
                self.assertEqual(config['voice_language'], 'Auto')
                self.assertEqual(config['output'], '/keep/my/output')
                self.assertNotIn('language', config)
                self.assertNotIn('python', config)
                core.save_settings({**config, 'python': '/untrusted/python'})
                self.assertNotIn('python', json.loads(path.read_text()))


if __name__ == '__main__':
    unittest.main()
