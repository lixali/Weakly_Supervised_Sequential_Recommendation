"""Exercise the real HSTU model without optional compiled recommendation kernels.

Run: python -m unittest discover -s code/tests -p test_hstu_dense.py

Only REC's unrelated logging-service imports are stubbed. Model construction,
attention, embeddings, NCE loss, gradients, optimizer, and prediction are real.
"""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
utils_stub = types.ModuleType("REC.utils")
utils_stub.__path__ = [str(CODE / "REC" / "utils")]
utils_stub.set_color = lambda message, *args, **kwargs: message
spec = importlib.util.spec_from_file_location(
    "_hstu_dense_regression", CODE / "REC" / "model" / "IDNet" / "hstu.py"
)
hstu = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"REC.utils": utils_stub, "fbgemm_gpu": None}):
    spec.loader.exec_module(hstu)


def tiny_model(item_embedding_size=16, num_negatives=512):
    config = {
        "item_embedding_size": item_embedding_size,
        "hstu_embedding_size": 16,
        "MAX_ITEM_LIST_LENGTH": 10,
        "n_layers": 2,
        "n_heads": 4,
        "hidden_act": "silu",
        "hidden_dropout_prob": 0.0,
        "attn_dropout_prob": 0.0,
        "enable_relative_attention_bias": True,
        "loss": "nce",
        "fix_temp": True,
        "nce_thres": 0.99,
        "num_negatives": num_negatives,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        return hstu.HSTU(config, types.SimpleNamespace(item_num=700))


class HSTUDenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(23)

    def test_nce_512_backward_optimizer_and_prediction(self):
        for item_embedding_size in (8, 16):
            with self.subTest(item_embedding_size=item_embedding_size):
                model = tiny_model(item_embedding_size).train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                items = torch.tensor([
                    [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                ])
                mask = items[:, :-1].ne(0).long()
                output = model((items, torch.zeros_like(items[:, :-1]), mask))
                self.assertTrue(torch.isfinite(output["loss"]))
                before = model.item_embedding.weight.detach().clone()
                output["loss"].backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(model._hstu._attention_layers[0]._uvqk.grad.abs().sum(), 0)
                optimizer.step()
                self.assertFalse(torch.equal(before, model.item_embedding.weight))
                model.eval()
                with contextlib.redirect_stdout(io.StringIO()):
                    features = model.compute_item_all()
                scores = model.predict(items[:, 1:], None, features)
                self.assertEqual(scores.shape, (2, 700))
                self.assertTrue(torch.isfinite(scores).all())

    def test_shared_batch_negatives_with_one_rank_backward(self):
        model = tiny_model(num_negatives=0).train()
        items = torch.arange(1, 23).reshape(2, 11)
        negatives = torch.arange(101, 123).reshape(2, 11)
        mask = torch.ones(2, 10, dtype=torch.long)
        mask[0, :6] = 0
        # Exercise the real BaseModel.all_gather single-rank branch. The test
        # only supplies the world size normally set by the torchrun launcher.
        with patch("torch.distributed.get_world_size", return_value=1):
            output = model((items, negatives, mask))
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        self.assertTrue(torch.isfinite(model.item_embedding.weight.grad).all())
        self.assertGreater(model.item_embedding.weight.grad[101:123].abs().sum(), 0)
        self.assertGreater(model._hstu._attention_layers[0]._uvqk.grad.abs().sum(), 0)

    def test_dense_attention_matches_explicit_silu_reference_and_gradients(self):
        # HSTU uses SiLU-normalized attention, not softmax attention/SDPA.
        tensors = [torch.randn(2, 4, 8, requires_grad=True) for _ in range(3)]
        reference = [value.detach().clone().requires_grad_() for value in tensors]
        mask = torch.tril(torch.ones(2, 1, 4, 4, dtype=torch.bool))
        mask[0, :, :, 0] = False
        actual = hstu._hstu_attention_maybe_from_cache(2, 4, 4, *tensors, mask)
        q, k, v = [value.view(2, 4, 2, 4).transpose(1, 2) for value in reference]
        weights = torch.nn.functional.silu(q @ k.transpose(-1, -2)) / 4
        expected = ((weights * mask) @ v).transpose(1, 2).reshape(2, 4, 8)
        torch.testing.assert_close(actual, expected)
        probe = torch.randn_like(actual)
        (actual * probe).sum().backward()
        (expected * probe).sum().backward()
        for value, ref in zip(tensors, reference):
            torch.testing.assert_close(value.grad, ref.grad)

    def test_attention_excludes_future_tokens(self):
        model = tiny_model().eval()
        tokens = torch.randn(2, 4, 16)
        mask = model.get_attention_mask(torch.ones(2, 4, dtype=torch.long))
        expected = model._hstu(tokens, mask)
        changed = tokens.clone()
        changed[:, 3] = torch.randn(2, 16) * 100
        actual = model._hstu(changed, mask)
        torch.testing.assert_close(actual[:, :3], expected[:, :3])

    def test_prediction_excludes_left_padding_embeddings(self):
        model = tiny_model().eval()
        items = torch.tensor([[0, 0, 1, 2], [0, 3, 4, 5]])
        with contextlib.redirect_stdout(io.StringIO()):
            features = model.compute_item_all()
        expected = model.predict(items, None, features)
        with torch.no_grad():
            model.item_embedding.weight[0].fill_(100)
        actual = model.predict(items, None, features)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
