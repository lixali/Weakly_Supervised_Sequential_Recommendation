"""Offline CPU regression tests for standard and paper-sized SASRec paths.

Run with PyTorch and Transformers 4.x installed:
    python -m unittest discover -s code/tests -p test_sasrec_colab.py

Tiny random configurations exercise the real models without model downloads.
Only unrelated training-service imports and the one-worker gather are stubbed.
"""

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import LlamaConfig


SOURCE = Path(__file__).resolve().parents[1] / "REC" / "model"
PACKAGE = "_sasrec_colab_regression"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(SOURCE / "HLLM")]
sys.modules[PACKAGE] = package


def load_module(name, source):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


llama = load_module("modeling_llama", SOURCE / "HLLM" / "modeling_llama.py")
layers = load_module("layers", SOURCE / "layers.py")
enum_stub = types.ModuleType("REC.utils.enum_type")
enum_stub.InputType = types.SimpleNamespace(SEQ="SEQ")
base_stub = types.ModuleType("REC.model.basemodel")
base_stub.BaseModel = nn.Module
base_stub.all_gather = lambda value, **kwargs: value.unsqueeze(0)
with patch.dict(sys.modules, {
    "REC.utils.enum_type": enum_stub,
    "REC.model.basemodel": base_stub,
    "REC.model.layers": layers,
}):
    SASRec = load_module("sasrec", SOURCE / "IDNet" / "sasrec.py").SASRec
    LLMIDRec = load_module("llmidrec", SOURCE / "IDNet" / "llmidrec.py").LLMIDRec


def legacy_encode(self, inputs_embeds, attention_mask):
    return self.user_llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    ).hidden_states[-1]


class SASRecColabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.config_dir = tempfile.TemporaryDirectory()
        LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=16, pad_token_id=0,
        ).save_pretrained(cls.config_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.config_dir.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.data = types.SimpleNamespace(item_num=32)
        self.config = {
            "loss": "nce", "fix_temp": True, "nce_thres": 0.99,
            "num_negatives": None, "MAX_ITEM_LIST_LENGTH": 5,
            "user_pretrain_dir": self.config_dir.name,
            "gradient_checkpointing": False, "use_ft_flash_attn": False,
            "user_llm_init": False, "item_embed_dim": 8,
        }
        self.items = torch.tensor([[9, 8, 7, 1, 2, 3], [7, 6, 2, 4, 5, 6]])
        self.negatives = torch.tensor([[10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21]])
        self.mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
        self.sequence = torch.tensor([[0, 0, 1, 2, 3], [0, 2, 4, 5, 6]])

    def llmidrec(self, **kwargs):
        # Resolve the production local model import without importing unrelated
        # REC training services such as W&B or Lightning.
        with patch.dict(sys.modules, {"REC.model.HLLM.modeling_llama": llama}):
            return LLMIDRec(self.config | kwargs, self.data)

    def assert_close(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)

    def test_config_only_initialization_needs_no_pretrained_weights(self):
        with patch.object(llama.LlamaForCausalLM, "from_pretrained", side_effect=AssertionError("weights requested")):
            model = self.llmidrec()
        self.assertFalse(model.user_llm.config.use_cache)
        self.assertFalse(model.user_llm.config.output_hidden_states)
        self.assertEqual(model.user_llm.config.hidden_size, 16)

    def test_backbone_matches_legacy_loss_and_parameter_gradients(self):
        model = self.llmidrec().float().eval()
        legacy = copy.deepcopy(model)
        legacy._encode_user = types.MethodType(legacy_encode, legacy)
        current = model((self.items, self.negatives, self.mask))
        expected = legacy((self.items, self.negatives, self.mask))
        self.assert_close(current["loss"], expected["loss"])
        current["loss"].backward()
        expected["loss"].backward()
        for (name, parameter), (_, reference) in zip(model.named_parameters(), legacy.named_parameters()):
            with self.subTest(parameter=name):
                if parameter.grad is None:
                    self.assertIsNone(reference.grad)
                else:
                    self.assert_close(parameter.grad, reference.grad)

    def test_full_sort_scores_match_legacy_and_skip_vocabulary_head(self):
        model = self.llmidrec().float().eval()
        legacy = copy.deepcopy(model)
        legacy._encode_user = types.MethodType(legacy_encode, legacy)
        with patch.object(model.user_llm.lm_head, "forward", side_effect=AssertionError("unused vocabulary head called")):
            current = model.predict(self.sequence, None, model.compute_item_all())
        expected = legacy.predict(self.sequence, None, legacy.compute_item_all())
        self.assertEqual(current.shape, (2, 32))
        self.assert_close(current, expected)

    def test_backbone_does_not_collect_all_hidden_states(self):
        model = self.llmidrec().float().eval()
        observed = []
        handle = model.user_llm.base_model.register_forward_hook(lambda module, args, output: observed.append(output))
        try:
            model((self.items, self.negatives, self.mask))
        finally:
            handle.remove()
        self.assertEqual(len(observed), 1)
        self.assertIsNone(observed[0].hidden_states)
        self.assertIsNone(observed[0].past_key_values)

    def test_bf16_training_optimizer_and_prediction_with_checkpointing(self):
        for checkpointing in (False, True):
            for negatives in (None, 7):
                with self.subTest(checkpointing=checkpointing, negatives=negatives):
                    model = self.llmidrec(gradient_checkpointing=checkpointing, num_negatives=negatives)
                    self.assertEqual(model.user_llm.model.gradient_checkpointing, checkpointing)
                    self._train_and_predict(model)

    def test_standard_sasrec_training_optimizer_and_prediction(self):
        for negatives in (None, 7):
            with self.subTest(negatives=negatives):
                config = self.config | {
                    "n_layers": 2, "n_heads": 4, "embedding_size": 16, "inner_size": 2,
                    "hidden_dropout_prob": 0.1, "attn_dropout_prob": 0.1,
                    "hidden_act": "gelu", "layer_norm_eps": 1e-12,
                    "initializer_range": 0.02, "num_negatives": negatives,
                }
                self._train_and_predict(SASRec(config, self.data))

    def _train_and_predict(self, model):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.item_embedding.weight.detach().clone()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model((self.items, self.negatives, self.mask))
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(before, model.item_embedding.weight))
        model.eval()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            scores = model.predict(self.sequence, None, model.compute_item_all())
        self.assertEqual(scores.shape, (2, 32))
        self.assertTrue(torch.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
