"""Exercise application logging without importing the GPU training stack."""
import contextlib
import importlib.util
import io
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class PlainColorFormatter(logging.Formatter):
    """Replace only optional terminal coloring; use real logging handlers."""

    def __init__(self, fmt, datefmt, **kwargs):
        super().__init__(fmt.replace('%(log_color)s', ''), datefmt)


class TrainingLoggerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.rank = 0
        self.root = logging.getLogger()
        self.original_handlers = self.root.handlers[:]
        self.original_level = self.root.level
        self.original_disabled = self.root.disabled
        self.original_filters = self.root.filters[:]
        self.root.handlers = []
        self.root.filters = []
        self.root.disabled = False
        self.addCleanup(self.restore_logging)

        # GPU/distributed setup and colors are unrelated to handler routing.
        # Isolate them while executing the repository's real logger function.
        dependencies = {
            'colorlog': SimpleNamespace(ColoredFormatter=PlainColorFormatter),
            'colorama': SimpleNamespace(init=lambda **kwargs: None),
            'torch': SimpleNamespace(distributed=SimpleNamespace(
                get_rank=lambda: self.rank, barrier=lambda: None)),
            'REC.utils.utils': SimpleNamespace(
                get_local_time=lambda: 'Oct-02-2026_00-48-19',
                ensure_dir=lambda path: Path(path).mkdir(parents=True, exist_ok=True)),
        }
        spec = importlib.util.spec_from_file_location(
            'training_logger_under_test',
            Path(__file__).resolve().parents[1] / 'REC' / 'utils' / 'logger.py')
        self.logger_module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, dependencies):
            spec.loader.exec_module(self.logger_module)

        self.config = {
            'checkpoint_dir': str(self.directory),
            'model': 'LLMIDRec',
            'log_path': None,
            'state': 'INFO',
        }

    def restore_logging(self):
        for handler in self.root.handlers[:]:
            self.root.removeHandler(handler)
            handler.close()
        self.root.handlers = self.original_handlers
        self.root.filters = self.original_filters
        self.root.setLevel(self.original_level)
        self.root.disabled = self.original_disabled

    def log_text(self):
        return next(self.directory.rglob('*.log')).read_text()

    def test_preexisting_warning_handler_does_not_hide_final_metrics(self):
        old_stream = io.StringIO()
        old_handler = logging.StreamHandler(old_stream)
        self.root.addHandler(old_handler)
        self.root.setLevel(logging.WARNING)
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            self.logger_module.init_logger(self.config)
            self.root.info('test result: recall@5=0.0153497')

        self.assertEqual(output.getvalue().count('test result:'), 1)
        self.assertIn('recall@5=0.0153497', self.log_text())
        self.assertEqual(old_stream.getvalue(), '')
        self.assertTrue(old_handler._closed)

    def test_reinitializing_closes_previous_handlers_without_duplicates(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.logger_module.init_logger(self.config)
            old_handlers = self.root.handlers[:]
            self.logger_module.init_logger(self.config)
            self.root.info('best valid: ndcg@200=0.0304453')

        self.assertEqual(len(self.root.handlers), 2)
        self.assertTrue(all(handler._closed for handler in old_handlers))
        self.assertEqual(output.getvalue().count('best valid:'), 1)
        self.assertEqual(self.log_text().count('best valid:'), 1)

    def test_nonzero_rank_keeps_info_suppressed(self):
        (self.directory / self.config['model']).mkdir()
        self.rank = 1
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.logger_module.init_logger(self.config)
            self.root.info('worker metric should remain hidden')
            self.root.warning('worker warning remains visible')

        self.assertNotIn('worker metric', output.getvalue())
        self.assertNotIn('worker metric', self.log_text())
        self.assertIn('worker warning', output.getvalue())
        self.assertIn('worker warning', self.log_text())

    def test_configured_warning_level_is_preserved(self):
        self.config['state'] = 'WARNING'
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.logger_module.init_logger(self.config)
            self.root.info('hidden info')
            self.root.warning('visible warning')

        self.assertNotIn('hidden info', output.getvalue())
        self.assertNotIn('hidden info', self.log_text())
        self.assertIn('visible warning', output.getvalue())
        self.assertIn('visible warning', self.log_text())


if __name__ == '__main__':
    unittest.main()
