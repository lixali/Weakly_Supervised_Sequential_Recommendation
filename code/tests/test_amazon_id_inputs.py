"""Data checks run without importing CUDA/training dependencies."""
import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'amazon_id_inputs', Path(__file__).resolve().parents[1] / 'check_amazon_id_inputs.py')
inputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inputs)


class InteractionChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'interactions.csv'
        self.rows = [[f'item{i}', f'user{i // 4}', str(i)] for i in range(200)]

    def check(self, header=None):
        with self.path.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow(header or ['item_id', 'user_id', 'timestamp'])
            writer.writerows(self.rows)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            inputs.check_interactions(self.path)
        return output.getvalue()

    def test_valid_data_with_bom_and_integer_timestamps(self):
        self.assertIn('50 users, 200 items, 200 interactions', self.check())

    def test_reordered_columns_rejected(self):
        with self.assertRaisesRegex(ValueError, 'columns'):
            self.check(['user_id', 'item_id', 'timestamp'])

    def test_empty_identifier_rejected(self):
        self.rows[0][0] = ''
        with self.assertRaisesRegex(ValueError, 'row 2'):
            self.check()

    def test_noninteger_timestamp_rejected(self):
        self.rows[0][2] = 'not a timestamp'
        with self.assertRaisesRegex(ValueError, 'integer'):
            self.check()

    def test_short_user_history_rejected(self):
        self.rows.append(['newitem', 'newuser', '201'])
        with self.assertRaisesRegex(ValueError, 'at least 4'):
            self.check()

    def test_insufficient_catalog_rejected(self):
        self.rows = self.rows[:196]
        with self.assertRaisesRegex(ValueError, 'at least 200'):
            self.check()

    def test_empty_data_rejected(self):
        self.rows = []
        with self.assertRaisesRegex(ValueError, 'at least 4'):
            self.check()


if __name__ == '__main__':
    unittest.main()
