# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""S0: configuration/validation tests for `moe_router_expert_pad_multiple`.

These tests are pure CPU (no distributed init, no CUDA): they only exercise the
`TransformerConfig.__post_init__` argument validation for the route-2
"router_select pads each expert to an m8 multiple" switch.
"""

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig


def _cfg(**kwargs):
    base = dict(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_router_topk=2,
        add_bias_linear=False,
    )
    base.update(kwargs)
    return TransformerConfig(**base)


def test_switch_defaults_to_none_and_is_inert():
    """Switch off (default) => no new validation is triggered, any dispatcher is fine."""
    cfg = _cfg()
    assert cfg.moe_router_expert_pad_multiple is None
    assert _cfg(moe_token_dispatcher_type="alltoallsdc").moe_router_expert_pad_multiple is None


def test_switch_accepts_alltoallsdc():
    cfg = _cfg(moe_token_dispatcher_type="alltoallsdc", moe_router_expert_pad_multiple=8)
    assert cfg.moe_router_expert_pad_multiple == 8


@pytest.mark.parametrize("multiple", [0, -1, -8])
def test_switch_must_be_positive(multiple):
    with pytest.raises(ValueError, match="must be a positive integer"):
        _cfg(moe_token_dispatcher_type="alltoallsdc", moe_router_expert_pad_multiple=multiple)


@pytest.mark.parametrize("dispatcher_type", ["allgather", "alltoall", "flex"])
def test_switch_requires_alltoallsdc(dispatcher_type):
    with pytest.raises(ValueError, match="only supported by the 'alltoallsdc'"):
        _cfg(moe_token_dispatcher_type=dispatcher_type, moe_router_expert_pad_multiple=8)


def test_switch_rejects_expert_capacity_factor():
    with pytest.raises(ValueError, match="moe_expert_capacity_factor"):
        _cfg(
            moe_token_dispatcher_type="alltoallsdc",
            moe_router_expert_pad_multiple=8,
            moe_expert_capacity_factor=0.5,
        )


def test_switch_rejects_capacity_padding_combination():
    """`moe_pad_expert_input_to_capacity` implies a capacity factor, so the capacity-factor
    rejection above already covers the drop-and-pad path; assert the combination is rejected."""
    with pytest.raises(ValueError, match="moe_expert_capacity_factor"):
        _cfg(
            moe_token_dispatcher_type="alltoallsdc",
            moe_router_expert_pad_multiple=8,
            moe_expert_capacity_factor=0.5,
            moe_pad_expert_input_to_capacity=True,
        )


def test_switch_rejects_router_padding_for_fp8():
    """fp8 in-place padding flips existing zeros; route-2 appends rows => mutually exclusive."""
    with pytest.raises(ValueError, match="moe_router_padding_for_fp8"):
        _cfg(
            moe_token_dispatcher_type="alltoallsdc",
            moe_router_expert_pad_multiple=8,
            moe_router_padding_for_fp8=True,
        )
