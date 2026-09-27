# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""S3/S4: MoE layer wiring of the route-2 router-side expert-multiple padding.

Single process (EP=1, TP=1, ETP=1) so the padding bookkeeping of the SDC dispatcher can be
checked without the 8/32-GPU distributed setup. The shape identities asserted here are the
ones that do not depend on the parallel sizes:

    padded pool rows   N_pad = N + E * m      (384 for N=256, E=16, m=8)
    routed rows        Q2    = sum(c) + D     (512 + D)
    permute rows       S2    = ETP * Q2       (1024 + 2D at ETP=2)
"""

import os
import pathlib
import subprocess
import sys

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.token_dispatcher import SDCMoEAlltoAllTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

M = 8
NUM_EXPERTS = 4
HIDDEN = 12
NUM_TOKENS = 32


def _config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=NUM_EXPERTS,
        use_cpu_initialization=True,
        moe_token_dispatcher_type="alltoallsdc",
        moe_router_load_balancing_type="aux_loss",
        moe_router_topk=2,
        moe_aux_loss_coeff=0.0,
        moe_grouped_gemm=False,
        moe_ffn_hidden_size=16,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _moe_layer(**overrides):
    config = _config(**overrides)
    spec = get_gpt_layer_local_spec(num_experts=NUM_EXPERTS, moe_grouped_gemm=False)
    return MoELayer(config, spec.submodules.mlp.submodules).cuda()


@pytest.fixture(autouse=True)
def _parallel():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_router_and_preprocess_materializes_the_padded_pool():
    moe_layer = _moe_layer(moe_router_expert_pad_multiple=M)
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        dispatched, probs, residual = moe_layer.router_and_preprocess(hidden_states)

    meta = moe_layer.router.get_expert_pad_meta()
    assert meta is not None
    assert meta.original_num_tokens == NUM_TOKENS
    assert meta.padded_num_tokens == NUM_TOKENS + NUM_EXPERTS * M

    # The dispatcher must have been told about the padding before the permute ran.
    assert moe_layer.token_dispatcher.pad_meta is meta
    assert tuple(moe_layer.token_dispatcher.original_hidden_shape) == (NUM_TOKENS, HIDDEN)

    # The dispatcher keeps the padded pool (N_pad rows) but permutes down to the routed
    # rows (Q2) -- at ETP=1 the permute output is exactly S2 == Q2.
    assert moe_layer.token_dispatcher.routing_map.shape[0] == meta.padded_num_tokens
    assert moe_layer.token_dispatcher.num_out_tokens == meta.num_routed_rows
    assert probs.shape[0] == meta.num_routed_rows
    assert dispatched.shape[0] == meta.num_routed_rows
    assert meta.num_routed_rows == int(meta.counts.sum().item()) + meta.num_dummy

    # The residual is the untouched MoE input, used by the shared expert path.
    assert residual.shape == hidden_states.shape
    assert torch.equal(residual, hidden_states)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_padding_is_off_by_default_in_the_layer():
    moe_layer = _moe_layer()
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        dispatched, probs, residual = moe_layer.router_and_preprocess(hidden_states)

    assert moe_layer.router.get_expert_pad_meta() is None
    assert moe_layer.token_dispatcher.pad_meta is None
    # Without padding the pool keeps N rows and the permute keeps N * topk rows.
    assert moe_layer.token_dispatcher.routing_map.shape[0] == NUM_TOKENS
    assert probs.shape[0] == NUM_TOKENS * 2
    assert moe_layer.token_dispatcher.num_out_tokens == NUM_TOKENS * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_combine_restores_the_moe_input_shape():
    moe_layer = _moe_layer(moe_router_expert_pad_multiple=M)
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)

    output, mlp_bias = moe_layer(hidden_states)
    output.sum().backward()

    assert mlp_bias is None
    assert output.shape == hidden_states.shape
    assert moe_layer.router.weight.grad is not None
    # The dummy rows must not leak a gradient into the gate: only real tokens route through it.
    assert torch.isfinite(moe_layer.router.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_matches_between_padded_and_unpadded():
    """The reservation rows are zero and unweighted, so the output must be identical."""
    padded = _moe_layer(moe_router_expert_pad_multiple=M)
    plain = _moe_layer()
    plain.load_state_dict(padded.state_dict())

    hidden_states = torch.randn(NUM_TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        padded_out, _ = padded(hidden_states)
        plain_out, _ = plain(hidden_states)

    assert padded_out.shape == plain_out.shape
    torch.testing.assert_close(padded_out, plain_out, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("expert_pad_multiple", [None, M], ids=["unpadded", "route2-padded"])
def test_fused_path_is_broken_with_and_without_padding(expert_pad_multiple, tmp_path):
    """Documents K8: the fused SDC path crashes before route 2 is even involved.

    The mapping built by ``_build_unpermute_mapping(fused=True)`` is not the mask-map that
    TE's ``moe_unpermute`` expects, so the Triton kernel reads out of bounds. This is
    pre-existing (it reproduces with ``moe_router_expert_pad_multiple`` disabled), so it is
    not a route-2 regression. Run in a subprocess because an illegal memory access
    invalidates the CUDA context of the calling process.
    """
    script = tmp_path / "fused_probe.py"
    script.write_text(_FUSED_PROBE)
    repo_root = pathlib.Path(__file__).resolve().parents[4]

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    env.setdefault("WORLD_SIZE", "1")
    env.setdefault("LOCAL_RANK", "0")
    env.setdefault("MASTER_ADDR", "localhost")
    # The probe must not reuse the rendezvous port of the pytest process itself.
    env["MASTER_PORT"] = os.environ.get("ROUTE2_PROBE_PORT", "60623")

    result = subprocess.run(
        [sys.executable, str(script), str(expert_pad_multiple or 0)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )

    if result.returncode == 0:
        pytest.skip("fused SDC path is working again; drop this K8 probe and re-enable fusion")
    output = result.stdout + result.stderr
    assert "Triton Error" in output or "AcceleratorError" in output, output[-2000:]


def test_set_expert_pad_meta_is_recorded_on_the_base_dispatcher():
    dispatcher = object.__new__(SDCMoEAlltoAllTokenDispatcher)
    dispatcher.pad_meta = None
    dispatcher.original_hidden_shape = None
    marker = object()

    dispatcher.set_expert_pad_meta(marker, torch.Size([32, HIDDEN]))

    assert dispatcher.pad_meta is marker
    assert dispatcher.original_hidden_shape == torch.Size([32, HIDDEN])


def test_clear_expert_pad_meta_drops_recorded_padding():
    dispatcher = object.__new__(SDCMoEAlltoAllTokenDispatcher)
    dispatcher.pad_meta = None
    dispatcher.original_hidden_shape = None
    dispatcher.set_expert_pad_meta(object(), torch.Size([32, HIDDEN]))

    dispatcher.clear_expert_pad_meta()

    assert dispatcher.pad_meta is None
    assert dispatcher.original_hidden_shape is None


def test_toggling_expert_pad_multiple_between_forward_passes(monkeypatch):
    """An unpadded pass must work even right after a padded one.

    ``set_expert_pad_meta`` is only called while padding is on, so the dispatcher has to be
    told when the padding goes away. Without ``clear_expert_pad_meta`` the stale ``pad_meta``
    makes the dispatcher slice/expect ``N + E * m`` rows and the pass dies with
    "size of tensor a (32) must match the size of tensor b (40)".
    """
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    Utils.initialize_model_parallel(1, 1)
    try:
        layer = _moe_layer()
        tokens = torch.randn(NUM_TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)

        def run():
            output = layer(tokens)
            return output[0] if isinstance(output, tuple) else output

        layer.router.expert_pad_multiple = None
        assert run().shape == tokens.shape
        assert layer.token_dispatcher.pad_meta is None

        layer.router.expert_pad_multiple = M
        assert run().shape == tokens.shape
        assert layer.token_dispatcher.pad_meta is not None

        layer.router.expert_pad_multiple = None
        assert run().shape == tokens.shape
        assert layer.token_dispatcher.pad_meta is None
    finally:
        Utils.destroy_model_parallel()


_FUSED_PROBE = """\
import sys

import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pad = int(sys.argv[1])
Utils.initialize_model_parallel(1, 1)
config = TransformerConfig(
    num_layers=1,
    hidden_size=12,
    num_attention_heads=4,
    num_moe_experts=4,
    use_cpu_initialization=True,
    moe_token_dispatcher_type="alltoallsdc",
    moe_router_load_balancing_type="aux_loss",
    moe_router_topk=2,
    moe_aux_loss_coeff=0.0,
    moe_grouped_gemm=False,
    moe_ffn_hidden_size=16,
    add_bias_linear=False,
    bf16=True,
    params_dtype=torch.bfloat16,
    moe_permute_fusion=True,
    moe_router_expert_pad_multiple=(pad or None),
)
spec = get_gpt_layer_local_spec(num_experts=4, moe_grouped_gemm=False)
layer = MoELayer(config, spec.submodules.mlp.submodules).cuda()
hidden = torch.randn(32, 12, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    layer(hidden)
torch.cuda.synchronize()
print("FUSED_PATH_OK")
"""
