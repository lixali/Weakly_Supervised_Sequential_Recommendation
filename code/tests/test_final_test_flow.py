"""Exercise final test orchestration after real fit/early-stopping control flow.

Only GPU setup, model kernels, and reporting side effects are replaced. Extracting
the production functions keeps these tests runnable without the training stack.
"""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
from time import time
import types
import unittest
from unittest.mock import Mock, patch


CODE = Path(__file__).resolve().parents[1]


def production_function(relative_path, name, namespace, class_name=None):
    """Compile an unchanged function from the repository's source file."""
    source = CODE / relative_path
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    if class_name is not None:
        tree = next(node for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'),
         namespace)
    return namespace[name]


class Config(dict):
    """Match Config's missing-key behavior without importing model registries."""

    def __getitem__(self, name):
        return self.get(name)


class Loader:
    def __init__(self, name):
        self.name = name
        self.sampler = Mock()

    def __len__(self):
        return 2


class FinalTestFlowTests(unittest.TestCase):
    def make_run(self, scores, epochs, rank=0, evaluation_error=None):
        events = []
        logger = Mock()
        config = Config(
            model='LLMIDRec', dataset='amazon_fixture', checkpoint_dir='checkpoints',
            seed=2020, reproducibility=True, baseline_train=True,
            finetune_clueweb=False, clueweb_pretrain=False, val_only=False,
            gen_relevance_score=False, show_progress=True, valid_metric_bigger=True,
            strategy='deepspeed', precision='bf16-mixed', stage=2,
        )
        loaders = [Loader(name) for name in ('train', 'valid', 'test')]
        model = object()
        trainer = types.SimpleNamespace(
            config=config, model=model, optimizer=object(), logger=logger,
            scheduler_config=None, rank=rank, start_epoch=0, epochs=epochs,
            eval_step=1, stopping_step=1, valid_metric_bigger=True,
            best_valid_score=float('-inf'), best_valid_result=None, cur_step=0,
            train_loss_dict={}, saved_model_file='checkpoints/LLMIDRec-0.pth',
            tensorboard=Mock(), wandblogger=Mock(),
        )

        def train_epoch(loader, epoch, **kwargs):
            self.assertIs(loader, loaders[0])
            events.append(('train', epoch))
            return 1.0

        validation_scores = iter(scores)

        def valid_epoch(loader, **kwargs):
            self.assertIs(loader, loaders[1])
            score = next(validation_scores)
            events.append(('valid', score))
            return score, {'ndcg@200': score, 'recall@200': score + 0.1}

        trainer._train_epoch = Mock(side_effect=train_epoch)
        trainer._valid_epoch = Mock(side_effect=valid_epoch)
        trainer._save_checkpoint = Mock(
            side_effect=lambda epoch, **kwargs: events.append(('save', epoch)))
        trainer._generate_train_loss_output = Mock(return_value='training loss')
        trainer._add_train_loss_to_tensorboard = Mock()
        fabric = Mock()
        fabric.setup.side_effect = lambda actual_model, optimizer: (actual_model, optimizer)
        fit_namespace = {
            'os': os, 'time': time, 'set_color': lambda text, color: text,
            'dict2str': str, 'DeepSpeedStrategy': Mock(), 'DDPStrategy': Mock(),
            'L': types.SimpleNamespace(Fabric=Mock(return_value=fabric)),
        }
        fit_namespace['early_stopping'] = production_function(
            'REC/utils/utils.py', 'early_stopping', {})
        fit = production_function('REC/trainer/trainer.py', 'fit', fit_namespace,
                                  class_name='Trainer')
        trainer.fit = types.MethodType(fit, trainer)

        test_result = {'ndcg@200': 0.025, 'recall@200': 0.1}

        def evaluate(loader, **kwargs):
            self.assertIs(loader, loaders[2])
            events.append(('test', kwargs['load_best_model']))
            if evaluation_error is not None:
                raise evaluation_error
            return test_result

        trainer.evaluate = Mock(side_effect=evaluate)
        reporter = Mock(side_effect=lambda *args, **kwargs: events.append(('report',)))
        distributed = types.SimpleNamespace(get_world_size=lambda: 2,
                                            get_rank=lambda: rank)
        namespace = {
            'Config': lambda **kwargs: Config(config),
            'torch': types.SimpleNamespace(
                device=lambda *args: 'fake_cuda', distributed=distributed),
            'dist': distributed, 'json': json, 'os': os,
            'init_logger': Mock(), 'init_seed': Mock(), 'getLogger': lambda: logger,
            'set_color': lambda text, color: text,
            'load_data': Mock(return_value=object()),
            'bulid_dataloader': Mock(return_value=loaders),
            'get_model': Mock(return_value=Mock(return_value=model)),
            'Trainer': Mock(return_value=trainer),
            'report_final_test_results': reporter,
        }
        run_loop = production_function('run.py', 'run_loop', namespace)

        def run(saved=True):
            with patch.dict(os.environ, {'WORLD_SIZE': '2', 'LOCAL_WORLD_SIZE': '2'}), \
                 contextlib.redirect_stdout(io.StringIO()):
                return run_loop(0, config_file=['baseline.yaml'], saved=saved)

        return types.SimpleNamespace(
            run=run, events=events, trainer=trainer, logger=logger, reporter=reporter,
            test_result=test_result, loaders=loaders,
        )

    def assert_one_best_checkpoint_test_and_report(self, scenario, best_score):
        scenario.trainer.evaluate.assert_called_once_with(
            scenario.loaders[2], load_best_model=True, show_progress=True)
        scenario.reporter.assert_called_once()
        args, kwargs = scenario.reporter.call_args
        self.assertEqual(args[0]['dataset'], 'amazon_fixture')
        self.assertIs(args[1], scenario.test_result)
        self.assertEqual(kwargs['best_valid_result']['ndcg@200'], best_score)
        self.assertEqual(kwargs['checkpoint_path'], scenario.trainer.saved_model_file)
        self.assertEqual(scenario.events[-2:], [('test', True), ('report',)])

    def test_early_stopping_automatically_tests_best_checkpoint_then_reports(self):
        scenario = self.make_run(scores=[0.4, 0.3, 0.2], epochs=201)
        result = scenario.run()

        self.assertEqual(scenario.trainer._train_epoch.call_count, 3)
        self.assertEqual(scenario.trainer._valid_epoch.call_count, 3)
        scenario.trainer._save_checkpoint.assert_called_once_with(0, verbose=True)
        self.assertTrue(any('Finished training' in str(call)
                            for call in scenario.logger.info.call_args_list))
        self.assertEqual(result['best_valid_score'], 0.4)
        self.assertIs(result['test_result'], scenario.test_result)
        self.assert_one_best_checkpoint_test_and_report(scenario, best_score=0.4)

    def test_max_epochs_completion_also_tests_best_checkpoint_then_reports(self):
        scenario = self.make_run(scores=[0.2, 0.4, 0.3], epochs=3)
        result = scenario.run()

        self.assertEqual(scenario.trainer._train_epoch.call_count, 3)
        self.assertEqual(scenario.trainer._save_checkpoint.call_count, 2)
        self.assertFalse(any('Finished training' in str(call)
                             for call in scenario.logger.info.call_args_list))
        self.assertEqual(result['best_valid_score'], 0.4)
        self.assert_one_best_checkpoint_test_and_report(scenario, best_score=0.4)

    def test_nonzero_rank_participates_in_test_without_duplicate_report(self):
        scenario = self.make_run(scores=[0.4, 0.3, 0.2], epochs=201, rank=1)
        result = scenario.run()

        scenario.trainer.evaluate.assert_called_once_with(
            scenario.loaders[2], load_best_model=True, show_progress=True)
        scenario.reporter.assert_not_called()
        self.assertEqual(scenario.events[-1], ('test', True))
        self.assertIs(result['test_result'], scenario.test_result)

    def test_evaluation_failure_propagates_without_false_results(self):
        error = RuntimeError('test checkpoint evaluation failed')
        scenario = self.make_run(scores=[0.4, 0.3, 0.2], epochs=201,
                                 evaluation_error=error)
        with self.assertRaises(RuntimeError) as raised:
            scenario.run()

        self.assertIs(raised.exception, error)
        scenario.trainer.evaluate.assert_called_once()
        scenario.reporter.assert_not_called()
        self.assertFalse(any('test result' in str(call)
                             for call in scenario.logger.info.call_args_list))

    def test_unsaved_run_reports_without_claiming_a_checkpoint(self):
        scenario = self.make_run(scores=[0.2, 0.4, 0.3], epochs=3)
        scenario.run(saved=False)

        scenario.trainer._save_checkpoint.assert_not_called()
        scenario.trainer.evaluate.assert_called_once_with(
            scenario.loaders[2], load_best_model=False, show_progress=True)
        scenario.reporter.assert_called_once()
        self.assertIsNone(scenario.reporter.call_args.kwargs['checkpoint_path'])


if __name__ == '__main__':
    unittest.main()
