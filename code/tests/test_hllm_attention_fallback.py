"""CPU regression tests for item attention without external FlashAttention.

Run with the training environment's PyTorch and Transformers 4.x installed:
    python -m unittest discover -s code/tests -p test_hllm_attention_fallback.py

Models are initialized with tiny random weights; no checkpoint or GPU is needed.
Only the unrelated REC training-service imports are stubbed.
"""

import copy
import importlib.util
import logging
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import LlamaConfig


SOURCE = Path(__file__).resolve().parents[1] / "REC" / "model" / "HLLM"
PACKAGE_NAME = "_hllm_attention_regression"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(SOURCE)]
sys.modules[PACKAGE_NAME] = package


def load_module(name):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE_NAME}.{name}", SOURCE / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


llama = load_module("modeling_llama")
enum_stub = types.ModuleType("REC.utils.enum_type")
enum_stub.InputType = types.SimpleNamespace(SEQ="SEQ")
base_stub = types.ModuleType("REC.model.basemodel")
base_stub.BaseModel = nn.Module
base_stub.all_gather = lambda value, **kwargs: value
with patch.dict(sys.modules, {
    "REC.utils.enum_type": enum_stub,
    "REC.model.basemodel": base_stub,
}):
    HLLM = load_module("hllm").HLLM


def tiny_llama(request_flash=False):
    config = LlamaConfig(
        vocab_size=48,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        use_cache=False,
        use_ft_flash_attn=request_flash,
    )
    # Test the missing-extension condition even on hosts with flash-attn installed.
    with patch.object(llama, "compute_flash_attention", None):
        return llama.LlamaForCausalLM(config)


def recommender(checkpointing=False):
    model = HLLM.__new__(HLLM)
    nn.Module.__init__(model)
    model.gradient_checkpointing = checkpointing
    model.logger = logging.getLogger(__name__)
    return model


class AttentionFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(23)

    def assert_close(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)

    def test_packed_capability_requires_every_attention_layer(self):
        rec = recommender()
        missing_extension = tiny_llama(request_flash=True)
        rec._finalize_llm(missing_extension, supports_cu_input_lens=True)
        self.assertFalse(missing_extension.supports_cu_input_lens)

        # Eligibility depends on the realized layers, not the requested config.
        for flags, advertised, expected in (
            ([], True, False),
            ([True, True], True, True),
            ([True, False], True, False),
            ([False, False], True, False),
            ([True, True], False, False),
        ):
            with self.subTest(flags=flags, advertised=advertised):
                model = nn.Module()
                model.layers = nn.ModuleList([nn.Module() for _ in flags])
                for layer, enabled in zip(model.layers, flags):
                    layer.use_ft_flash_attn = enabled
                rec._finalize_llm(model, supports_cu_input_lens=advertised)
                self.assertEqual(model.supports_cu_input_lens, expected)

    def test_sdpa_matches_eager_outputs_and_gradients_with_causal_padding(self):
        for padding in ("none", "right", "left"):
            with self.subTest(padding=padding):
                model = tiny_llama().train()
                reference = copy.deepcopy(model).train()
                inputs = torch.randint(1, 48, (2, 5))
                mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
                if padding == "none":
                    mask.fill_(1)
                elif padding == "left":
                    mask[1] = torch.tensor([0, 0, 1, 1, 1])
                actual = model.base_model(
                    input_ids=inputs, attention_mask=mask,
                    output_attentions=False,
                ).last_hidden_state
                eager = reference.base_model(
                    input_ids=inputs, attention_mask=mask,
                    output_attentions=True,
                )
                self.assertEqual(eager.attentions[0].shape, (2, 4, 5, 5))
                # Fully padded query positions do not contribute to the loss.
                self.assert_close(actual[mask.bool()], eager.last_hidden_state[mask.bool()])
                probe = torch.randn_like(actual) * mask.unsqueeze(-1)
                (actual * probe).sum().backward()
                (eager.last_hidden_state * probe).sum().backward()
                for name, value in model.named_parameters():
                    expected = dict(reference.named_parameters())[name]
                    if value.grad is not None:
                        self.assert_close(value.grad, expected.grad)

    def test_pooled_items_match_independent_sequences_and_gradients(self):
        lengths = torch.tensor([3, 5, 2], dtype=torch.int32)
        positions = torch.cat([torch.arange(int(n)) for n in lengths])
        inputs = torch.randint(1, 48, (int(lengths.sum()),))
        for special_token in (False, True):
            for checkpointing in (False, True):
                with self.subTest(special_token=special_token,
                                  checkpointing=checkpointing):
                    rec = recommender(checkpointing)
                    model = rec._finalize_llm(tiny_llama(True), True).train()
                    reference = copy.deepcopy(model).train()
                    token = nn.Parameter(torch.randn(1, 1, 16) * 0.02)
                    reference_token = nn.Parameter(token.detach().clone())
                    actual = rec.forward_item_emb(
                        inputs, positions, lengths, int(special_token), token, model
                    )
                    expected_items = []
                    start = 0
                    for length in lengths.tolist():
                        end = start + length
                        embeddings = reference.get_input_embeddings()(inputs[start:end])
                        if special_token:
                            embeddings = torch.cat((embeddings[:-1], reference_token[0]))
                        hidden = rec._forward_backbone(
                            reference,
                            inputs_embeds=embeddings.unsqueeze(0),
                            position_ids=positions[start:end].unsqueeze(0),
                            attention_mask=torch.ones(1, length, dtype=torch.long),
                        )[0]
                        expected_items.append(hidden[-1] if special_token else hidden.mean(0))
                        start = end
                    expected = torch.stack(expected_items)
                    self.assert_close(actual, expected)
                    probe = torch.randn_like(actual)
                    (actual * probe).sum().backward()
                    (expected * probe).sum().backward()
                    self.assert_close(
                        model.get_input_embeddings().weight.grad,
                        reference.get_input_embeddings().weight.grad,
                    )
                    for current_layer, expected_layer in zip(
                        model.model.layers, reference.model.layers
                    ):
                        for projection in ("q_proj", "k_proj", "v_proj"):
                            self.assert_close(
                                getattr(current_layer.self_attn, projection).weight.grad,
                                getattr(expected_layer.self_attn, projection).weight.grad,
                            )
                    if special_token:
                        self.assertIsNotNone(token.grad)
                        self.assert_close(token.grad, reference_token.grad)
                    if checkpointing:
                        self.assertTrue(model.is_gradient_checkpointing)

    def test_items_are_isolated_and_attention_uses_maximum_item_length(self):
        rec = recommender()
        model = rec._finalize_llm(tiny_llama(True), True).eval()
        lengths = torch.tensor([3, 5, 2], dtype=torch.int32)
        positions = torch.cat([torch.arange(int(n)) for n in lengths])
        inputs = torch.randint(1, 48, (int(lengths.sum()),))
        token = torch.randn(1, 1, 16) * 0.02
        changed = inputs.clone()
        changed[:3] = changed[:3].remainder(47) + 1
        for special_token in (False, True):
            with self.subTest(special_token=special_token):
                with torch.no_grad(), patch.object(
                    llama.F, "scaled_dot_product_attention",
                    wraps=llama.F.scaled_dot_product_attention,
                ) as attention:
                    actual = rec.forward_item_emb(
                        inputs, positions, lengths, int(special_token), token, model
                    )
                    modified = rec.forward_item_emb(
                        changed, positions, lengths, int(special_token), token, model
                    )
                self.assertGreater(attention.call_count, 0)
                for call in attention.call_args_list:
                    query, key = call.args[:2]
                    self.assertEqual(query.shape[0], len(lengths))
                    self.assertEqual(query.shape[-2], int(lengths.max()))
                    self.assertEqual(key.shape[-2], int(lengths.max()))
                self.assert_close(actual[1:], modified[1:])
                self.assertFalse(torch.allclose(actual[0], modified[0]))

    def test_direct_packed_model_input_rejected_before_dense_mask(self):
        model = tiny_llama(True).base_model
        inputs = torch.randint(1, 48, (1, 7))
        with patch.object(model, "_prepare_decoder_attention_mask") as mask:
            mask.side_effect = AssertionError("dense mask allocated for packed tokens")
            with self.assertRaisesRegex(ValueError, "[Pp]acked|[Ff]lash|cu_input_lens"):
                model(input_ids=inputs, cu_input_lens=torch.tensor([3, 4]))
            mask.assert_not_called()

    def test_causal_prefix_is_unchanged_when_future_tokens_change(self):
        model = tiny_llama().base_model.eval()
        inputs = torch.randint(1, 48, (2, 6))
        changed = inputs.clone()
        changed[:, 3:] = changed[:, 3:].remainder(47) + 1
        with torch.no_grad():
            actual = model(input_ids=inputs).last_hidden_state
            modified = model(input_ids=changed).last_hidden_state
        self.assert_close(actual[:, :3], modified[:, :3])
        self.assertFalse(torch.allclose(actual[:, 3:], modified[:, 3:]))

    def test_cached_single_token_matches_full_sequence_and_eager(self):
        model = tiny_llama().base_model.eval()
        inputs = torch.randint(1, 48, (2, 5))
        with torch.no_grad():
            expected = model(input_ids=inputs).last_hidden_state[:, -1:]
            prefix = model(input_ids=inputs[:, :-1], use_cache=True)
            kwargs = dict(
                input_ids=inputs[:, -1:],
                attention_mask=torch.ones_like(inputs),
                past_key_values=prefix.past_key_values,
                use_cache=True,
            )
            actual = model(**kwargs).last_hidden_state
            eager = model(**kwargs, output_attentions=True).last_hidden_state
        self.assert_close(actual, expected)
        self.assert_close(actual, eager)

    def test_hllm_training_loss_backward_through_both_models(self):
        rec = recommender(checkpointing=True)
        rec.item_llm = rec._finalize_llm(tiny_llama(True), True)
        rec.user_llm = rec._finalize_llm(tiny_llama(True), True)
        rec.item_emb_token_n = 1
        rec.item_emb_tokens = nn.Parameter(torch.randn(1, 1, 16) * 0.02)
        rec.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
        rec.nce_thres = 0.99
        interaction = {"attention_mask": torch.ones(2, 2, dtype=torch.long)}
        # Two users, two history items and a target each; two negatives per user.
        for prefix, sizes in (("pos", [3, 2, 4, 3, 5, 2]), ("neg", [3, 4, 2, 5])):
            interaction[f"{prefix}_cu_input_lens"] = torch.tensor(sizes, dtype=torch.int32)
            interaction[f"{prefix}_input_ids"] = torch.randint(1, 48, (sum(sizes),))
            interaction[f"{prefix}_position_ids"] = torch.cat([
                torch.arange(size) for size in sizes
            ])
        result = rec.train()(interaction)
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertGreater(result["loss"].item(), 0)
        result["loss"].backward()
        gradients = [rec.item_emb_tokens.grad]
        for model in (rec.item_llm, rec.user_llm):
            gradients.extend(layer.self_attn.q_proj.weight.grad for layer in model.model.layers)
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_direct_packed_attention_rejected_before_dense_attention(self):
        attention = tiny_llama(True).model.layers[0].self_attn
        with patch.object(llama.torch, "matmul") as matmul, patch.object(
            llama.F, "scaled_dot_product_attention"
        ) as sdpa:
            matmul.side_effect = AssertionError("dense packed attention allocated")
            sdpa.side_effect = AssertionError("packed attention reached SDPA")
            with self.assertRaisesRegex(ValueError, "[Pp]acked|[Ff]lash|cu_input_lens"):
                attention(
                    hidden_states=torch.randn(1, 7, 16),
                    position_ids=torch.tensor([[0, 1, 2, 0, 1, 2, 3]]),
                    cu_input_lens=torch.tensor([3, 4]),
                )
            matmul.assert_not_called()
            sdpa.assert_not_called()


if __name__ == "__main__":
    unittest.main()
