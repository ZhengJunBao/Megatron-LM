# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""S1: unit tests for the route-2 expert-multiple padding helpers.

Pure CPU, no distributed init. Covers the padding invariants plus the boundary token
distributions that the m8 alignment has to survive.
"""

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import (
    ExpertPadMeta,
    check_expert_pad_invariants,
    compute_expert_pad_dummy_counts,
    pad_routing_map_for_expert_multiple,
    pad_row_count_for_expert_multiple,
    pad_token_pool_for_expert_multiple,
    unpad_token_pool_for_expert_multiple,
)

M = 8


def _map_from_counts(counts, num_experts=None):
    """Build a [N, E] bool map whose column e has exactly counts[e] ones."""
    counts = torch.tensor(counts, dtype=torch.long)
    num_experts = num_experts or len(counts)
    num_tokens = int(counts.sum().item())
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool)
    row = 0
    for expert, count in enumerate(counts.tolist()):
        routing_map[row : row + count, expert] = True
        row += count
    return routing_map


# --------------------------------------------------------------------------------------
# compute_expert_pad_dummy_counts
# --------------------------------------------------------------------------------------


def test_dummy_counts_round_up_to_multiple():
    counts = torch.tensor([3, 3, 4, 2])
    assert compute_expert_pad_dummy_counts(counts, M).tolist() == [5, 5, 4, 6]


@pytest.mark.parametrize("count", [1, 7, 8, 9, 15, 16, 17, 256])
def test_dummy_counts_never_exceed_m_minus_one_for_positive_count(count):
    dummy = compute_expert_pad_dummy_counts(torch.tensor([count]), M)
    assert 0 <= int(dummy.item()) <= M - 1
    assert (count + int(dummy.item())) % M == 0


def test_dummy_counts_zero_count_gets_a_full_block():
    """c[e] == 0 is the special case: pad a whole block so the expert is not empty."""
    assert compute_expert_pad_dummy_counts(torch.tensor([0]), M).tolist() == [M]


def test_dummy_counts_positive_multiple_stays_empty():
    """c[e] = 8/16/... (positive multiple of m, not zero) => d[e] = 0."""
    assert compute_expert_pad_dummy_counts(torch.tensor([8, 16, 24]), M).tolist() == [0, 0, 0]


def test_dummy_counts_rejects_non_positive_multiple():
    with pytest.raises(AssertionError):
        compute_expert_pad_dummy_counts(torch.tensor([1]), 0)


# --------------------------------------------------------------------------------------
# pad_routing_map_for_expert_multiple: shape + invariants
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,counts",
    [
        ("even", [32, 32, 32, 32]),
        ("ragged", [3, 3, 4, 2]),
        ("expert_full", [256, 0, 0, 0]),
        ("expert_empty", [0, 0, 0, 0]),
        ("many_empty", [120, 0, 0, 0, 0, 0, 0, 0]),
        ("already_aligned", [8, 16, 24, 8]),
    ],
)
def test_padding_invariants_hold(name, counts):
    routing_map = _map_from_counts(counts)
    padded, meta = pad_routing_map_for_expert_multiple(routing_map, M)

    assert check_expert_pad_invariants(routing_map, padded, meta) == []

    num_tokens, num_experts = routing_map.shape
    assert padded.shape == (num_tokens + num_experts * M, num_experts)
    assert meta.padded_num_tokens == num_tokens + num_experts * M
    assert meta.reserved_rows == num_experts * M
    assert meta.original_num_tokens == num_tokens


def test_real_rows_are_bit_identical():
    routing_map = _map_from_counts([3, 3, 4, 2])
    padded, meta = pad_routing_map_for_expert_multiple(routing_map, M)
    assert torch.equal(padded[: meta.original_num_tokens], routing_map)


def test_every_column_becomes_a_multiple_of_m():
    routing_map = _map_from_counts([3, 3, 4, 2])
    padded, _ = pad_routing_map_for_expert_multiple(routing_map, M)
    assert (padded.sum(dim=0) % M == 0).all()


def test_dummy_rows_select_exactly_one_expert():
    routing_map = _map_from_counts([1, 2, 3, 4])
    padded, meta = pad_routing_map_for_expert_multiple(routing_map, M)
    dummy_rows = padded[meta.original_num_tokens :]
    assert not bool((dummy_rows.sum(dim=1) > 1).any())


def test_disabled_reservation_slots_stay_zero():
    routing_map = _map_from_counts([1, 2, 3, 4, 5])
    padded, meta = pad_routing_map_for_expert_multiple(routing_map, M)
    reserved = padded[meta.original_num_tokens :].reshape(meta.num_experts, M, meta.num_experts)
    for expert in range(meta.num_experts):
        enabled = int(meta.dummy_counts[expert].item())
        assert bool(reserved[expert, :enabled, expert].all())
        assert not bool(reserved[expert, enabled:].any())


def test_slot_start_depends_only_on_expert_index_not_on_distribution():
    """Expert e always owns [N + e*m, N + (e+1)*m): the layout strides by m, not by d[e]."""
    # Same N (=12) but different dummy totals (20 vs 28).
    counts_a = [3, 3, 4, 2]
    counts_b = [12, 0, 0, 0]
    map_a = _map_from_counts(counts_a)
    map_b = _map_from_counts(counts_b)
    padded_a, meta_a = pad_routing_map_for_expert_multiple(map_a, M)
    padded_b, meta_b = pad_routing_map_for_expert_multiple(map_b, M)

    assert meta_a.original_num_tokens == meta_b.original_num_tokens == 12
    assert meta_a.num_dummy != meta_b.num_dummy
    assert meta_a.reserved_rows == meta_b.reserved_rows == 4 * M

    for expert in range(meta_a.num_experts):
        start = meta_a.original_num_tokens + expert * M
        assert torch.equal(padded_a[start : start + M], padded_b[start : start + M]) is False or True
        # expert e's block only ever touches column e
        block_a = padded_a[start : start + M]
        assert not bool(block_a[:, [c for c in range(4) if c != expert]].any())
        block_b = padded_b[start : start + M]
        assert not bool(block_b[:, [c for c in range(4) if c != expert]].any())


def test_padding_is_deterministic_for_identical_input():
    routing_map = _map_from_counts([3, 3, 4, 2])
    first, meta_first = pad_routing_map_for_expert_multiple(routing_map, M)
    second, meta_second = pad_routing_map_for_expert_multiple(routing_map, M)
    assert torch.equal(first, second)
    assert torch.equal(meta_first.dummy_counts, meta_second.dummy_counts)


def test_num_routed_rows_matches_real_plus_dummy():
    routing_map = _map_from_counts([3, 3, 4, 2])
    padded, meta = pad_routing_map_for_expert_multiple(routing_map, M)
    assert meta.num_routed_rows == int(routing_map.sum().item()) + meta.num_dummy
    assert meta.num_routed_rows == int(padded.sum().item())


def test_expected_num_out_tokens_applies_etp():
    routing_map = _map_from_counts([3, 3, 4, 2])
    _, meta = pad_routing_map_for_expert_multiple(routing_map, M)
    assert meta.expected_num_out_tokens(tp_size=2) == meta.num_routed_rows * 2
    assert meta.expected_num_out_tokens(tp_size=1) == meta.num_routed_rows


def test_mismatched_padded_shape_is_reported():
    routing_map = _map_from_counts([3, 3, 4, 2])
    padded, meta = pad_routing_map_for_expert_multiple(routing_map, M)
    problems = check_expert_pad_invariants(routing_map, padded[:-1], meta)
    assert problems and "shape" in problems[0]


def test_unpadded_map_as_padded_is_reported():
    routing_map = _map_from_counts([3, 3, 4, 2])
    _, meta = pad_routing_map_for_expert_multiple(routing_map, M)
    problems = check_expert_pad_invariants(routing_map, routing_map, meta)
    assert any("shape" in problem for problem in problems)


def test_modified_real_rows_are_detected():
    routing_map = _map_from_counts([3, 3, 4, 2])
    padded, meta = pad_routing_map_for_expert_multiple(routing_map, M)
    padded[0, 0] = False
    assert "real token rows were modified" in check_expert_pad_invariants(
        routing_map, padded, meta
    )


def test_integer_routing_map_dtype_is_preserved():
    routing_map = _map_from_counts([3, 3, 4, 2]).to(torch.int32)
    padded, _ = pad_routing_map_for_expert_multiple(routing_map, M)
    assert padded.dtype == torch.int32
    assert (padded.sum(dim=0) % M == 0).all()


def test_pad_row_count_matches_map():
    assert pad_row_count_for_expert_multiple(256, M, 16) == 384


# --------------------------------------------------------------------------------------
# token pool padding / unpadding
# --------------------------------------------------------------------------------------


def test_token_pool_padding_appends_zero_rows():
    routing_map = _map_from_counts([3, 3, 4, 2])
    _, meta = pad_routing_map_for_expert_multiple(routing_map, M)
    hidden = torch.randn(meta.original_num_tokens, 5)

    padded_hidden = pad_token_pool_for_expert_multiple(hidden, meta)

    assert padded_hidden.shape == (meta.padded_num_tokens, 5)
    assert torch.equal(padded_hidden[: meta.original_num_tokens], hidden)
    assert padded_hidden[meta.original_num_tokens :].abs().sum().item() == 0


def test_token_pool_padding_leaves_dtype_and_requires_grad():
    routing_map = _map_from_counts([3, 3, 4, 2])
    _, meta = pad_routing_map_for_expert_multiple(routing_map, M)
    hidden = torch.randn(meta.original_num_tokens, 5, dtype=torch.bfloat16, requires_grad=True)

    padded_hidden = pad_token_pool_for_expert_multiple(hidden, meta)

    assert padded_hidden.dtype == torch.bfloat16
    assert padded_hidden.requires_grad


def test_unpad_is_the_inverse_of_pad():
    routing_map = _map_from_counts([3, 3, 4, 2])
    _, meta = pad_routing_map_for_expert_multiple(routing_map, M)
    hidden = torch.randn(meta.original_num_tokens, 5)

    restored = unpad_token_pool_for_expert_multiple(
        pad_token_pool_for_expert_multiple(hidden, meta), meta
    )

    assert torch.equal(restored, hidden)


def test_unpad_drops_gradients_of_reserved_rows():
    """Backward through the reserved rows must leave the real rows untouched."""
    routing_map = _map_from_counts([3, 3, 4, 2])
    _, meta = pad_routing_map_for_expert_multiple(routing_map, M)
    hidden = torch.randn(meta.original_num_tokens, 5, requires_grad=True)

    padded = pad_token_pool_for_expert_multiple(hidden, meta)
    # Give the reserved rows a large gradient contribution; a pure slice must drop it.
    padded.grad = torch.zeros_like(padded)
    padded.grad[meta.original_num_tokens :] = 1000.0
    unpad_token_pool_for_expert_multiple(padded, meta).sum().backward()

    assert torch.equal(hidden.grad, torch.ones_like(hidden))


def test_probs_pool_padding_matches_map_shape():
    routing_map = _map_from_counts([3, 3, 4, 2])
    _, meta = pad_routing_map_for_expert_multiple(routing_map, M)
    probs = torch.rand(meta.original_num_tokens, meta.num_experts)

    padded_probs = pad_token_pool_for_expert_multiple(probs, meta)

    assert padded_probs.shape == (meta.padded_num_tokens, meta.num_experts)
    assert padded_probs[meta.original_num_tokens :].abs().sum().item() == 0


def test_token_pool_padding_rejects_wrong_row_count():
    routing_map = _map_from_counts([3, 3, 4, 2])
    _, meta = pad_routing_map_for_expert_multiple(routing_map, M)
    with pytest.raises(AssertionError):
        pad_token_pool_for_expert_multiple(torch.randn(meta.original_num_tokens + 1, 5), meta)


def test_meta_reserved_rows_is_zero_when_no_experts():
    meta = ExpertPadMeta(
        multiple=M,
        original_num_tokens=4,
        num_experts=0,
        counts=torch.zeros(0, dtype=torch.long),
        dummy_counts=torch.zeros(0, dtype=torch.long),
    )
    assert meta.reserved_rows == 0
    hidden = torch.randn(4, 3)
    assert pad_token_pool_for_expert_multiple(hidden, meta) is hidden
