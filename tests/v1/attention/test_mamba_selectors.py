# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for mamba attention backend selectors."""

import pytest

from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionBackend
from vllm.v1.attention.backends.mamba_selectors import get_mamba_attn_backend


# TODO: add other mamba layer types
@pytest.mark.parametrize(argnames=[
    "layer_cls", "mamba_kwargs", "expected_backend", "expected_mamba_type"
],
                         argvalues=[(
                             MambaMixer,
                             dict(
                                 hidden_size=128,
                                 ssm_state_size=16,
                                 conv_kernel_size=4,
                                 intermediate_size=256,
                                 time_step_rank=8,
                                 use_conv_bias=True,
                                 use_bias=False,
                                 use_rms_norm=True,
                             ),
                             Mamba1AttentionBackend,
                             "mamba1",
                         )])
def test_get_mamba_attn_backend_mamba(dist_init, layer_cls, mamba_kwargs,
                                      expected_backend, expected_mamba_type):
    layer = layer_cls(**mamba_kwargs)
    assert expected_backend == layer.get_attn_backend()
    assert layer.mamba_type == expected_mamba_type


def test_get_mamba_attn_backend_unsupported():
    unsupported_types = ["mamba", ""]

    for mamba_type in unsupported_types:
        err_message = f"Mamba Attention type {mamba_type} is not supported yet."
        with pytest.raises(NotImplementedError, match=err_message):
            get_mamba_attn_backend(mamba_type)
