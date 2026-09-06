import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers import BatchEncoding

from worker import Cleaner, model_batch_size


class CleanupBatchTests(unittest.TestCase):
    def test_scheduling_order_and_remainder(self):
        pieces = [f'Excerpt number {i}.' for i in range(13)]
        for size, counts in ((6, [6, 6, 1]), (1, [1] * 13)):
            cleaner = Cleaner.__new__(Cleaner)
            cleaner.batch_size = size
            cleaner.clean_batch = Mock(side_effect=lambda parts, _language: [(part, []) for part in parts])
            with patch('worker.cleanup_chunks', return_value=pieces), patch('worker.release_gpu') as release:
                result = cleaner.clean('Input.', lambda *_: None, 'English')
            self.assertEqual(result, '\n\n'.join(pieces))
            self.assertEqual([len(call.args[0]) for call in cleaner.clean_batch.call_args_list], counts)
            self.assertEqual(release.call_count, len(counts))

    def test_both_models_share_cpu_cuda_and_metal_policy(self):
        for device in ('cpu', 'cuda:0', 'mps'):
            for capacity in (3, 4, 8):
                model = SimpleNamespace(device=torch.device(device))
                with patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(total_memory=capacity * 2**30)), \
                     patch('torch.mps.recommended_max_memory', return_value=capacity * 2**30):
                    self.assertEqual(model_batch_size(model), 6 if device == 'cpu' or capacity > 4 else 1)

    def test_padded_generation_orders_outputs_and_checks_each_eos(self):
        cleaner = Cleaner.__new__(Cleaner)
        cleaner.prompt = 'Keep the original language.'
        cleaner.tokenizer = Mock()
        cleaner.tokenizer.apply_chat_template.side_effect = ['English chat', 'Spanish chat']
        cleaner.tokenizer.encode.return_value = [1, 2]
        cleaner.tokenizer.return_value = BatchEncoding({
            'input_ids': torch.tensor([[0, 0, 1], [2, 3, 4]]),
            'attention_mask': torch.tensor([[0, 0, 1], [1, 1, 1]])})
        cleaner.tokenizer.decode.side_effect = ['First output.', 'Second output.']
        # First row finished early then padded; the second really hit its cap.
        result = torch.tensor([[0, 0, 1, 10, 99] + [0] * 510,
                               [2, 3, 4] + [11] * 512])
        cleaner.model = SimpleNamespace(device='cpu', config=SimpleNamespace(max_position_embeddings=4096),
                                       generation_config=SimpleNamespace(eos_token_id=[99]),
                                       generate=Mock(return_value=result))
        parts = ['English words ' * 30, 'Palabras originales ' * 30]
        with patch('worker.detect_language', side_effect=['English', 'Spanish']):
            outputs = cleaner.generate_batch(parts, 'Auto')
        self.assertEqual(outputs, [('First output.', 'English', False), ('Second output.', 'Spanish', True)])
        cleaner.tokenizer.assert_called_once_with(['English chat', 'Spanish chat'], padding=True, return_tensors='pt')
        self.assertTrue(torch.equal(cleaner.model.generate.call_args.kwargs['attention_mask'],
                                    torch.tensor([[0, 0, 1], [1, 1, 1]])))
        self.assertEqual(cleaner.model.generate.call_count, 1)

    def test_warnings_belong_to_their_own_excerpt(self):
        cleaner = Cleaner.__new__(Cleaner)
        cleaner.batch_size = 6
        parts = ['The first excerpt stays unchanged.', 'The second excerpt stays unchanged.']
        cleaner.generate_batch = Mock(return_value=[(parts[0], 'English', False), ('', 'English', False)])
        outputs = cleaner.clean_batch(parts, 'English')
        self.assertEqual(outputs[0], (parts[0], []))
        self.assertEqual(outputs[1][0], parts[1])
        self.assertEqual(len(outputs[1][1]), 1)

    def test_memory_errors_do_not_reduce_batch_size_or_retry(self):
        for error in (torch.OutOfMemoryError('test'), MemoryError('test')):
            cleaner = Cleaner.__new__(Cleaner)
            cleaner.batch_size = 6
            cleaner.generate_batch = Mock(side_effect=error)
            with self.assertRaises(type(error)):
                cleaner.clean_batch(['Sentence.'] * 6, 'English')
            self.assertEqual(cleaner.batch_size, 6)
            cleaner.generate_batch.assert_called_once()


if __name__ == '__main__':
    unittest.main()
