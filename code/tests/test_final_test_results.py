"""Final metric output remains usable without application logging or a GPU."""
from collections import OrderedDict
import contextlib
import importlib.util
import io
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    'final_results', Path(__file__).resolve().parents[1] / 'REC' / 'utils' / 'results.py')
results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results)


class FinalTestResultsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / 'checkpoint with spaces'
        self.config = {'checkpoint_dir': str(self.directory), 'model': 'LLMIDRec', 'dataset': 'amazon'}
        self.metrics = OrderedDict((f'{metric}@{k}', i / 100)
            for i, (metric, k) in enumerate(
                ((metric, k) for metric in ('recall', 'ndcg') for k in (5, 10, 50, 200)), 1))
        self.validation = {'ndcg@200': 0.1234567}

    def report(self, **kwargs):
        return results.report_final_test_results(self.config, self.metrics, **kwargs)

    def test_metrics_print_and_save_even_with_logging_disabled(self):
        previous_disable = logging.root.manager.disable
        self.addCleanup(logging.disable, previous_disable)
        logging.disable(logging.CRITICAL)
        output = io.StringIO()
        checkpoint = self.directory / 'LLMIDRec-0.pth'
        with contextlib.redirect_stdout(output), patch('builtins.print', wraps=print) as printer:
            destination = self.report(best_valid_result=self.validation, checkpoint_path=checkpoint)
        self.assertEqual(destination, self.directory / 'test_results.json')
        data = json.loads(destination.read_text(encoding='utf-8'))
        self.assertEqual(data['test_result'], self.metrics)
        self.assertEqual(data['best_valid_result'], self.validation)
        self.assertEqual(data['checkpoint'], str(checkpoint))
        self.assertEqual(data['model'], 'LLMIDRec')
        self.assertEqual(data['dataset'], 'amazon')
        self.assertIn('completed_at_utc', data)
        self.assertIn('FINAL TEST RESULTS', output.getvalue())
        for name in self.metrics:
            self.assertIn(name, output.getvalue())
        self.assertIn(str(destination.resolve()), output.getvalue())
        self.assertTrue(all(call.kwargs.get('flush') for call in printer.call_args_list))

    def test_evaluation_only_has_no_invented_validation_metrics(self):
        with contextlib.redirect_stdout(io.StringIO()):
            data = json.loads(self.report().read_text(encoding='utf-8'))
        self.assertIsNone(data['best_valid_result'])
        self.assertIsNone(data['checkpoint'])

    def test_empty_evaluation_does_not_claim_success(self):
        output = io.StringIO()
        self.metrics = None
        with contextlib.redirect_stdout(output), self.assertRaisesRegex(ValueError, 'no metrics'):
            self.report()
        self.assertEqual(output.getvalue(), '')
        self.assertFalse((self.directory / 'test_results.json').exists())

    def test_nonfinite_evaluation_does_not_write_invalid_json(self):
        self.metrics['ndcg@200'] = float('nan')
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(ValueError):
            self.report()
        self.assertEqual(output.getvalue(), '')
        self.assertFalse((self.directory / 'test_results.json').exists())

    def test_write_failure_preserves_console_metrics_and_prior_result(self):
        self.directory.mkdir()
        destination = self.directory / 'test_results.json'
        prior = '{"previous_run": true}\n'
        destination.write_text(prior, encoding='utf-8')
        output = io.StringIO()
        with contextlib.redirect_stdout(output), \
             patch.object(results.os, 'replace', side_effect=OSError('cannot replace')), \
             self.assertRaisesRegex(OSError, 'cannot replace'):
            self.report()
        self.assertIn('FINAL TEST RESULTS', output.getvalue())
        self.assertNotIn('Test results saved to:', output.getvalue())
        self.assertEqual(destination.read_text(encoding='utf-8'), prior)
        self.assertEqual(list(self.directory.glob('.test_results.*.tmp')), [])

    def test_completed_run_replaces_prior_metrics(self):
        with contextlib.redirect_stdout(io.StringIO()):
            destination = self.report()
            self.metrics['recall@5'] = 0.7654321
            self.report()
        self.assertEqual(json.loads(destination.read_text())['test_result'], self.metrics)
        self.assertEqual(list(self.directory.glob('.test_results.*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
