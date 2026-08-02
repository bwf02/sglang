from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmRunnerInput,
    DeepGemmRunnerOutput,
    post_permute_deep_gemm_to_standard,
    pre_permute_standard_to_deep_gemm,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend


_SPARSE_GEMM_MOE_PATH_ENV = "SGLANG_SPARSE_GEMM_MOE_PATH"
_SPARSE_GEMM_KERNEL_ENV = "SGLANG_SPARSE_GEMM_KERNEL"


@dataclass
class SparseGemmMoeQuantInfo(MoeQuantInfo):
    gate_weight: object
    up_weight: object
    down_weight: object
    dense_w13_weight: torch.Tensor
    block_shape: Optional[list[int]] = None

    @property
    def w13_weight(self) -> torch.Tensor:
        # The standard->DeepGEMM pre-permute path only needs dtype/device and
        # block_shape from quant_info. SparseGEMM keeps gate/up physically split.
        return self.dense_w13_weight


class SparseGemmRunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        if self.config.activation != "silu" or not self.config.is_gated:
            raise ValueError("SparseGEMM MoE currently supports gated SiLU only")

    def run(
        self,
        runner_input: RunnerInput,
        quant_info: MoeQuantInfo,
        running_state: dict,
        hooks=None,
    ) -> RunnerOutput:
        if not isinstance(runner_input, DeepGemmRunnerInput):
            raise TypeError("SparseGEMM runner expects DeepGEMM-style input")
        if not isinstance(quant_info, SparseGemmMoeQuantInfo):
            raise TypeError("SparseGEMM runner expects SparseGemmMoeQuantInfo")
        if not runner_input.use_masked_gemm:
            raise NotImplementedError(
                "SparseGEMM MoE currently supports the masked grouped path only"
            )
        hidden_states = self._run_masked_bf16_gemm(runner_input, quant_info)
        return DeepGemmRunnerOutput(hidden_states=hidden_states)

    def _run_masked_bf16_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: SparseGemmMoeQuantInfo,
    ) -> torch.Tensor:
        hidden_states = runner_input.hidden_states
        masked_m = runner_input.masked_m
        expected_m = runner_input.expected_m
        if hidden_states.dtype != torch.bfloat16:
            raise TypeError("SparseGEMM MoE currently expects BF16 activations")
        if masked_m is None:
            raise ValueError("masked_m is required for SparseGEMM masked grouped GEMM")

        gate_output = _grouped_masked_gemm(
            hidden_states,
            quant_info.gate_weight,
            masked_m,
            expected_m,
        )
        up_output = _grouped_masked_gemm(
            hidden_states,
            quant_info.up_weight,
            masked_m,
            expected_m,
        )

        down_input = (F.silu(gate_output.float()) * up_output.float()).to(
            torch.bfloat16
        )
        del gate_output, up_output

        down_output = _grouped_masked_gemm(
            down_input,
            quant_info.down_weight,
            masked_m,
            expected_m,
        )
        return down_output

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.SPARSE_GEMM


def _grouped_masked_gemm(
    activation: torch.Tensor,
    packed_weight: object,
    masked_m: torch.Tensor,
    expected_m: Optional[int],
) -> torch.Tensor:
    kernel = os.environ.get(_SPARSE_GEMM_KERNEL_ENV, "wgmma_tma")
    if kernel == "wgmma_tma":
        from sparse_gemm.hybrid_sparse import (
            hybrid_block_sparse_grouped_masked_wgmma_tma,
        )

        return hybrid_block_sparse_grouped_masked_wgmma_tma(
            activation, packed_weight, masked_m, expected_m=expected_m
        )
    if kernel == "naive":
        from sparse_gemm.hybrid_sparse import hybrid_block_sparse_grouped_masked_naive

        return hybrid_block_sparse_grouped_masked_naive(
            activation, packed_weight, masked_m
        )
    if kernel == "ref":
        from sparse_gemm.hybrid_sparse import hybrid_block_sparse_grouped_masked_ref

        return hybrid_block_sparse_grouped_masked_ref(activation, packed_weight, masked_m)
    raise ValueError(
        f"{_SPARSE_GEMM_KERNEL_ENV} must be one of 'wgmma_tma', 'naive', or 'ref'"
    )


def load_sparse_gemm_moe_weight(
    *,
    layer_id: int,
    projection: str,
    device: torch.device,
) -> object:
    manifest_root = os.environ.get(_SPARSE_GEMM_MOE_PATH_ENV)
    if not manifest_root:
        raise RuntimeError(
            f"{_SPARSE_GEMM_MOE_PATH_ENV} must point to a MosaicMoE SparseGEMM export"
        )
    manifest_path = Path(manifest_root)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    logical_name = f"model.layers.{layer_id}.mlp.experts.{projection}.weight"
    for entry in manifest["weights"]:
        if entry["logical_name"] != logical_name:
            continue
        payload_path = manifest_path.parent / entry["file"]
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        return _payload_to_sparse_weight(payload, device)
    raise KeyError(f"{logical_name} not found in {manifest_path}")


def _payload_to_sparse_weight(payload: dict, device: torch.device) -> object:
    from sparse_gemm.hybrid_sparse import (
        HybridBlockSparseLayout,
        HybridBlockSparseWeight,
    )

    def move(name: str) -> torch.Tensor:
        return payload[name].to(device=device, non_blocking=True).contiguous()

    hardware_metadata = payload["hardware_metadata"]
    if hardware_metadata is not None:
        hardware_metadata = hardware_metadata.to(
            device=device, non_blocking=True
        ).contiguous()

    return HybridBlockSparseWeight(
        original_shape=tuple(payload["original_shape"]),
        layout=HybridBlockSparseLayout(**payload["layout"]),
        block_selector=move("block_selector"),
        dense_values=move("dense_values"),
        sparse_values=move("sparse_values"),
        sparse_metadata=move("sparse_metadata"),
        hardware_metadata=hardware_metadata,
    )


@register_pre_permute("standard", "sparse_gemm")
def pre_permute_standard_to_sparse_gemm(
    dispatch_output,
    quant_info: SparseGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    return pre_permute_standard_to_deep_gemm(
        dispatch_output, quant_info, runner_config, running_state
    )


@register_post_permute("sparse_gemm", "standard")
def post_permute_sparse_gemm_to_standard(
    runner_output: DeepGemmRunnerOutput,
    quant_info: SparseGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    return post_permute_deep_gemm_to_standard(
        runner_output, quant_info, runner_config, running_state
    )
