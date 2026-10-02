"""Exercise the actual run.py entry point with simulated CUDA/distributed APIs."""
import argparse
import ast
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


source = Path(__file__).resolve().parents[1] / 'run.py'
tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
entry = next(node for node in tree.body if isinstance(node, ast.If)
             and ast.unparse(node.test) == "__name__ == '__main__'")
# Execute the production entry point without importing model/training services.
entry_code = compile(ast.Module(body=entry.body, type_ignores=[]), str(source), 'exec')


class TrainingShutdownTests(unittest.TestCase):
    def setUp(self):
        self.distributed = Mock()
        self.distributed.is_initialized.return_value = True
        self.run_loop = Mock()
        self.cuda = Mock()
        self.globals = {'argparse': argparse, 'os': os, 'dist': self.distributed,
                        'torch': types.SimpleNamespace(cuda=self.cuda),
                        'run_loop': self.run_loop}

    def run_entry(self):
        with patch.dict(os.environ, {'LOCAL_RANK': '0', 'POST_MORTEM_DEBUG': 'False'}), \
             patch.object(sys, 'argv', ['run.py', '--config_file', 'overall/ID_deepspeed.yaml',
                                       'IDNet/llama_id.yaml', '--baseline_train', 'True']):
            exec(entry_code, self.globals)

    def test_success_releases_process_group_after_training(self):
        self.run_loop.side_effect = lambda **kwargs: self.distributed.destroy_process_group.assert_not_called()
        self.run_entry()
        self.cuda.set_device.assert_called_once_with(0)
        self.distributed.init_process_group.assert_called_once_with(backend='nccl')
        self.run_loop.assert_called_once_with(local_rank=0,
            config_file=['overall/ID_deepspeed.yaml', 'IDNet/llama_id.yaml'],
            extra_args=['--baseline_train', 'True'])
        self.distributed.destroy_process_group.assert_called_once_with()

    def test_training_exception_propagates_and_releases_group(self):
        error = RuntimeError('training failure')
        self.run_loop.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            self.run_entry()
        self.assertIs(raised.exception, error)
        self.distributed.destroy_process_group.assert_called_once_with()

    def test_keyboard_interrupt_releases_group(self):
        self.run_loop.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_entry()
        self.distributed.destroy_process_group.assert_called_once_with()

    def test_failed_initialization_does_not_destroy_nonexistent_group(self):
        self.distributed.init_process_group.side_effect = RuntimeError('initialization failed')
        self.distributed.is_initialized.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'initialization failed'):
            self.run_entry()
        self.run_loop.assert_not_called()
        self.distributed.destroy_process_group.assert_not_called()

    def test_already_destroyed_group_is_not_destroyed_twice(self):
        self.run_loop.side_effect = lambda **kwargs: setattr(
            self.distributed.is_initialized, 'return_value', False)
        self.run_entry()
        self.distributed.destroy_process_group.assert_not_called()


if __name__ == '__main__':
    unittest.main()
