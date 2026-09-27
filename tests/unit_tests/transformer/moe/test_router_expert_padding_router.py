# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""S2: router-side wiring of the route-2 expert-multiple padding.

The stub router keeps the test CPU-only: it replaces the gating GEMM (the only GPU-bound
step) with a deterministic logits tensor, so the padding wiring, the metadata hand-off and
the aux-loss/expert-bias interaction can be checked without a device.
"""

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import (
    check_expert_pad_invariants,
    get_default_model_comm_pgs,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

M = 8
NUM_EXPERTS = 4
NUM_TOKENS = 24


class _StubGatingRouter(TopKRouter):
    """TopKRouter whose gating returns fixed logits so no device is required."""

    def setup_logits(self, logits: torch.Tensor):
        self._stub_logits = logits

    def gating(self, input: torch.Tensor):
        return self._stub_logits


def _config(**overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=12,
        num_attention_heads=4,
        num_moe_experts=NUM_EXPERTS,
        use_cpu_initialization=True,
        moe_router_load_balancing_type="aux_loss",
        moe_router_topk=2,
        moe_aux_loss_coeff=0,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        moe_token_dispatcher_type="alltoallsdc",
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _router(**overrides):
    router = _StubGatingRouter(config=_config(**overrides), model_comm_pgs=get_default_model_comm_pgs())
    router.setup_logits(torch.randn(NUM_TOKENS, NUM_EXPERTS))
    return router


@pytest.fixture(autouse=True)
def _init_parallel():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def test_padding_disabled_by_default():
    router = _router()
    assert router.expert_pad_multiple is None

    probs, routing_map = router(torch.randn(NUM_TOKENS, 12))

    assert probs.shape == (NUM_TOKENS, NUM_EXPERTS)
    assert routing_map.shape == (NUM_TOKENS, NUM_EXPERTS)
    assert router.get_expert_pad_meta() is None


def test_padding_enabled_pads_probs_and_map():
    router = _router(moe_router_expert_pad_multiple=M)
    assert router.expert_pad_multiple == M

    probs, routing_map = router(torch.randn(NUM_TOKENS, 12))

    meta = router.get_expert_pad_meta()
    assert meta is not None
    assert meta.multiple == M
    assert meta.original_num_tokens == NUM_TOKENS
    assert meta.num_experts == NUM_EXPERTS

    padded_rows = NUM_TOKENS + NUM_EXPERTS * M
    assert probs.shape == (padded_rows, NUM_EXPERTS)
    assert routing_map.shape == (padded_rows, NUM_EXPERTS)

    # Dummy rows never carry a probability; a non-zero one would pollute the expert weights.
    assert probs[NUM_TOKENS:].abs().sum().item() == 0
    # Real rows keep their probabilities verbatim.
    assert probs.shape[0] == meta.padded_num_tokens
    assert routing_map.dtype == torch.bool
    assert check_expert_pad_invariants(routing_map[:NUM_TOKENS], routing_map, meta) == []


def test_padding_leaves_real_token_probs_unchanged():
    router = _router(moe_router_expert_pad_multiple=M)
    hidden = torch.randn(NUM_TOKENS, 12)

    padded_probs, padded_map = router(hidden)

    # Same router, padding switched off: the real rows must be bit-identical.
    router.expert_pad_multiple = None
    reference_probs, reference_map = router(hidden)

    assert reference_map.shape == (NUM_TOKENS, NUM_EXPERTS)
    assert torch.equal(padded_map[:NUM_TOKENS], reference_map)
    assert torch.allclose(padded_probs[:NUM_TOKENS], reference_probs)


def test_padding_runs_after_expert_bias_accumulation():
    """local_tokens_per_expert must only ever see real tokens, never the dummy rows."""
    router = _router(
        moe_router_expert_pad_multiple=M,
        moe_router_load_balancing_type="none",
        moe_router_enable_expert_bias=True,
        moe_router_score_function="sigmoid",
    )
    router.local_tokens_per_expert.zero_()

    # Must run with grad enabled: the accumulation is skipped under torch.no_grad().
    probs, routing_map = router(torch.randn(NUM_TOKENS, 12))

    real_map = routing_map[:NUM_TOKENS]
    assert torch.equal(router.local_tokens_per_expert, real_map.sum(dim=0).float())
    assert int(router.local_tokens_per_expert.sum().item()) == NUM_TOKENS * router.topk


def test_padding_requires_the_multiple_to_be_positive():
    router = _router()
    router.expert_pad_multiple = 0
    with pytest.raises(AssertionError):
        router.pad_for_expert_multiple(
            torch.rand(NUM_TOKENS, NUM_EXPERTS), torch.zeros(NUM_TOKENS, NUM_EXPERTS, dtype=torch.bool)
        )
