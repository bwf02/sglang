from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

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
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend


_SPARSE_GEMM_MOE_PATH_ENV = "SGLANG_SPARSE_GEMM_MOE_PATH"
_SPARSE_GEMM_KERNEL_ENV = "SGLANG_SPARSE_GEMM_KERNEL"
_SPARSE_GEMM_LAYOUT_ENV = "SGLANG_SPARSE_GEMM_LAYOUT"
_SPARSE_GEMM_M_ALIGNMENT_ENV = "SGLANG_SPARSE_GEMM_M_ALIGNMENT"
_SPARSE_GEMM_MASKED_M_ALIGNMENT_ENV = "SGLANG_SPARSE_GEMM_MASKED_M_ALIGNMENT"
_SPARSE_GEMM_CONTIGUOUS_MIN_M_ENV = "SGLANG_SPARSE_GEMM_CONTIGUOUS_MIN_M"

@dataclass
class SparseGemmRunnerInput(DeepGemmRunnerInput):
    grouped_layout: Optional[torch.Tensor] = None
    m_alignment: int = 128


@dataclass
class SparseGemmMoeQuantInfo(MoeQuantInfo):
    w13_weight: object
    down_weight: object
    dense_w13_weight: torch.Tensor
    moe_tp_rank: int = 0
    block_shape: Optional[list[int]] = None

    @property
    def dense_w13(self) -> torch.Tensor:
        # The standard->DeepGEMM pre-permute path only needs dtype/device and
        # block_shape from quant_info. SparseGEMM keeps the real sparse w13
        # payload separately because the preprocess only needs dense metadata.
        return self.dense_w13_weight

    def down_input_padding(self, actual_columns: int) -> tuple[int, int]:
        padded_columns = self.down_weight.original_shape[-1]
        if actual_columns == padded_columns:
            return 0, 0
        block_width = self.down_weight.layout.block_w
        block_group = self.down_weight.layout.block_m
        actual_blocks = actual_columns // block_width
        left_blocks = (self.moe_tp_rank * actual_blocks) % block_group
        left = left_blocks * block_width
        right = padded_columns - actual_columns - left
        if left < 0 or right < 0:
            raise ValueError(
                f"invalid SparseGEMM down padding ({left}, {right}) for "
                f"{actual_columns} -> {padded_columns} columns"
            )
        return left, right


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
        if runner_input.use_masked_gemm:
            hidden_states = self._run_masked_bf16_gemm(runner_input, quant_info)
        else:
            if not isinstance(runner_input, SparseGemmRunnerInput):
                raise TypeError(
                    "SparseGEMM contiguous path expects SparseGemmRunnerInput"
                )
            hidden_states = self._run_contiguous_bf16_gemm(runner_input, quant_info)
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

        gateup_output = _grouped_masked_gemm(
            hidden_states,
            quant_info.w13_weight,
            masked_m,
            expected_m,
        )

        from sglang.srt.layers.moe.ep_moe.kernels import (
            silu_and_mul_masked_padded_fwd,
        )

        actual_columns = gateup_output.shape[2] // 2
        left, _ = quant_info.down_input_padding(actual_columns)
        down_input = silu_and_mul_masked_padded_fwd(
            gateup_output,
            masked_m,
            quant_info.down_weight.original_shape[-1],
            left,
        )
        del gateup_output

        down_output = _grouped_masked_gemm(
            down_input,
            quant_info.down_weight,
            masked_m,
            expected_m,
        )
        return down_output

    def _run_contiguous_bf16_gemm(
        self,
        runner_input: SparseGemmRunnerInput,
        quant_info: SparseGemmMoeQuantInfo,
    ) -> torch.Tensor:
        hidden_states = runner_input.hidden_states
        grouped_layout = runner_input.grouped_layout
        m_alignment = runner_input.m_alignment
        if hidden_states.dtype != torch.bfloat16:
            raise TypeError("SparseGEMM MoE currently expects BF16 activations")
        if grouped_layout is None:
            raise ValueError("grouped_layout is required for contiguous SparseGEMM")

        gateup_output = _grouped_contiguous_gemm(
            hidden_states,
            quant_info.w13_weight,
            grouped_layout,
            m_alignment,
        )

        from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_padded_fwd

        actual_columns = gateup_output.shape[1] // 2
        left, right = quant_info.down_input_padding(actual_columns)
        down_input = silu_and_mul_padded_fwd(
            gateup_output,
            quant_info.down_weight.original_shape[-1],
            left,
        )
        del gateup_output

        down_output = _grouped_contiguous_gemm(
            down_input,
            quant_info.down_weight,
            grouped_layout,
            m_alignment,
            active_tail_block=(
                left // quant_info.down_weight.layout.block_w
                if left + right == quant_info.down_weight.layout.block_w
                else -1
            ),
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


def _grouped_contiguous_gemm(
    activation: torch.Tensor,
    packed_weight: object,
    grouped_layout: torch.Tensor,
    m_alignment: int,
    active_tail_block: int = -1,
) -> torch.Tensor:
    kernel = os.environ.get(_SPARSE_GEMM_KERNEL_ENV, "wgmma_tma")
    if kernel == "wgmma_tma":
        from sparse_gemm.hybrid_sparse import (
            hybrid_block_sparse_grouped_contiguous_wgmma_tma,
        )

        return hybrid_block_sparse_grouped_contiguous_wgmma_tma(
            activation,
            packed_weight,
            grouped_layout,
            m_alignment,
            active_tail_block=active_tail_block,
        )
    if kernel == "naive":
        from sparse_gemm.hybrid_sparse import hybrid_block_sparse_grouped_contiguous_naive

        return hybrid_block_sparse_grouped_contiguous_naive(
            activation, packed_weight, grouped_layout, m_alignment
        )
    if kernel == "ref":
        from sparse_gemm.hybrid_sparse import hybrid_block_sparse_grouped_contiguous_ref

        return hybrid_block_sparse_grouped_contiguous_ref(
            activation, packed_weight, grouped_layout, m_alignment
        )
    raise ValueError(
        f"{_SPARSE_GEMM_KERNEL_ENV} must be one of 'wgmma_tma', 'naive', or 'ref'"
    )


def load_sparse_gemm_moe_weight(
    *,
    layer_id: int,
    projection: str,
    device: torch.device,
    expert_start: int = 0,
    num_local_experts: Optional[int] = None,
    moe_tp_rank: int = 0,
    moe_tp_size: int = 1,
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
    if projection == "w13_weight":
        logical_name = f"model.layers.{layer_id}.mlp.experts.w13_weight"
    else:
        logical_name = f"model.layers.{layer_id}.mlp.experts.{projection}.weight"
    for entry in manifest["weights"]:
        if entry["logical_name"] != logical_name:
            continue
        payload_path = manifest_path.parent / entry["file"]
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        return _payload_to_sparse_weight(
            payload,
            device,
            projection=projection,
            expert_start=expert_start,
            num_local_experts=num_local_experts,
            moe_tp_rank=moe_tp_rank,
            moe_tp_size=moe_tp_size,
        )
    raise KeyError(f"{logical_name} not found in {manifest_path}")


def load_sparse_gemm_shared_weight(
    *,
    layer_id: int,
    projection: str,
    device: torch.device,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> object:
    manifest_root = os.environ.get(_SPARSE_GEMM_MOE_PATH_ENV)
    if not manifest_root:
        raise RuntimeError(
            f"{_SPARSE_GEMM_MOE_PATH_ENV} must point to a MosaicMoE SparseGEMM export"
        )
    if projection not in ("gate_up_proj", "down_proj"):
        raise ValueError(f"unsupported shared expert projection: {projection}")
    manifest_path = Path(manifest_root)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    logical_name = (
        f"model.layers.{layer_id}.mlp.shared_expert.{projection}.weight"
    )
    for entry in manifest["weights"]:
        if entry["logical_name"] != logical_name:
            continue
        payload = torch.load(
            manifest_path.parent / entry["file"],
            map_location="cpu",
            weights_only=True,
        )
        return _payload_to_sparse_shared_weight(
            payload,
            device,
            projection=projection,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
    raise KeyError(f"{logical_name} not found in {manifest_path}")


def _payload_to_sparse_shared_weight(
    payload: dict,
    device: torch.device,
    *,
    projection: str,
    tp_rank: int,
    tp_size: int,
) -> object:
    from sparse_gemm.hybrid_sparse import (
        HybridBlockSparseLayout,
        HybridBlockSparseWeight,
    )

    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"invalid shared expert TP rank/size ({tp_rank}, {tp_size})")
    layout = HybridBlockSparseLayout(**payload["layout"])
    original_rows, original_columns = payload["original_shape"]

    if projection == "gate_up_proj":
        if original_rows % (2 * layout.block_h) != 0:
            raise ValueError(
                "shared gate/up rows must be divisible by 2 * block_h"
            )
        half_blocks = original_rows // (2 * layout.block_h)
        if half_blocks % tp_size != 0:
            raise ValueError("shared gate/up rows must divide evenly across TP ranks")
        local_blocks = half_blocks // tp_size
        block_start = tp_rank * local_blocks
        block_end = block_start + local_blocks

        def local(tensor: torch.Tensor) -> torch.Tensor:
            half = tensor.shape[0] // 2
            return torch.cat(
                (
                    tensor[block_start:block_end],
                    tensor[half + block_start : half + block_end],
                ),
                dim=0,
            )

        local_shape = (2 * local_blocks * layout.block_h, original_columns)
    elif projection == "down_proj":
        if original_columns % layout.block_w != 0:
            raise ValueError("shared down columns must be divisible by block_w")
        total_blocks = original_columns // layout.block_w
        if total_blocks % tp_size != 0:
            raise ValueError("shared down columns must divide evenly across TP ranks")
        local_blocks = total_blocks // tp_size
        block_start = tp_rank * local_blocks
        block_end = block_start + local_blocks
        if block_start % layout.block_m or block_end % layout.block_m:
            raise ValueError(
                "shared down TP boundaries must align to SparseGEMM block groups"
            )
        group_start = block_start // layout.block_m
        group_end = block_end // layout.block_m

        def local(tensor: torch.Tensor) -> torch.Tensor:
            return tensor[:, group_start:group_end]

        local_shape = (original_rows, local_blocks * layout.block_w)
    else:
        raise ValueError(f"unsupported shared expert projection: {projection}")

    def move(name: str) -> torch.Tensor:
        return local(payload[name]).to(
            device=device, non_blocking=True
        ).contiguous()

    hardware_metadata = payload["hardware_metadata"]
    if hardware_metadata is not None:
        hardware_metadata = local(hardware_metadata).to(
            device=device, non_blocking=True
        ).contiguous()

    return HybridBlockSparseWeight(
        original_shape=local_shape,
        layout=layout,
        block_selector=move("block_selector"),
        dense_values=move("dense_values"),
        sparse_values=move("sparse_values"),
        sparse_metadata=move("sparse_metadata"),
        hardware_metadata=hardware_metadata,
    )


def _payload_to_sparse_weight(
    payload: dict,
    device: torch.device,
    *,
    projection: str,
    expert_start: int = 0,
    num_local_experts: Optional[int] = None,
    moe_tp_rank: int = 0,
    moe_tp_size: int = 1,
) -> object:
    from sparse_gemm.hybrid_sparse import (
        HybridBlockSparseLayout,
        HybridBlockSparseWeight,
    )

    global_experts = payload["original_shape"][0]
    if num_local_experts is None:
        num_local_experts = global_experts
    expert_end = expert_start + num_local_experts
    if expert_start < 0 or expert_end > global_experts:
        raise ValueError(
            f"invalid local expert range [{expert_start}, {expert_end}) for "
            f"SparseGEMM weight with {global_experts} experts"
        )

    if moe_tp_size <= 0 or not 0 <= moe_tp_rank < moe_tp_size:
        raise ValueError(
            f"invalid MoE TP rank/size ({moe_tp_rank}, {moe_tp_size})"
        )

    layout = HybridBlockSparseLayout(**payload["layout"])
    original_rows, original_columns = payload["original_shape"][-2:]
    if projection == "w13_weight":
        if original_rows % (2 * layout.block_h) != 0:
            raise ValueError(
                "gated w13 rows must be divisible by 2 * block_h, got "
                f"{original_rows} and block_h={layout.block_h}"
            )
        intermediate_blocks = original_rows // (2 * layout.block_h)
        if intermediate_blocks % moe_tp_size != 0:
            raise ValueError(
                "gated w13 intermediate blocks must divide evenly across MoE TP ranks"
            )
        local_block_count = intermediate_blocks // moe_tp_size
        block_start = moe_tp_rank * local_block_count
        block_end = block_start + local_block_count
        group_start = group_end = local_group_count = 0
    elif projection == "down_proj":
        total_blocks = original_columns // layout.block_w
        if original_columns % layout.block_w != 0 or total_blocks % moe_tp_size != 0:
            raise ValueError(
                "down projection blocks must divide evenly across MoE TP ranks"
            )
        local_block_count = total_blocks // moe_tp_size
        block_start = moe_tp_rank * local_block_count
        block_end = block_start + local_block_count
        group_start = block_start // layout.block_m
        group_end = (block_end + layout.block_m - 1) // layout.block_m
        local_group_count = group_end - group_start
    else:
        raise ValueError(
            f"unsupported SparseGEMM MoE projection for TP sharding: {projection}"
        )

    def local(name: str) -> torch.Tensor:
        tensor = payload[name][expert_start:expert_end]
        if moe_tp_size == 1:
            return tensor
        if projection == "w13_weight":
            half_block_rows = tensor.shape[1] // 2
            return torch.cat(
                (
                    tensor[:, block_start:block_end],
                    tensor[
                        :,
                        half_block_rows + block_start : half_block_rows + block_end,
                    ],
                ),
                dim=1,
            )
        result = tensor[:, :, group_start:group_end]
        if name not in ("dense_values", "sparse_values"):
            return result

        result = result.clone()
        selector = payload["block_selector"][
            expert_start:expert_end, :, group_start:group_end
        ]
        for local_group in range(local_group_count):
            global_group = group_start + local_group
            for slot in range(layout.block_m):
                global_block = global_group * layout.block_m + slot
                if block_start <= global_block < block_end:
                    continue
                is_sparse = ((selector[:, :, local_group] >> slot) & 1).bool()
                selected = ~is_sparse if name == "dense_values" else is_sparse
                if not selected.any():
                    continue

                sparse_prefix = sum(
                    (selector[:, :, local_group] >> previous_slot) & 1
                    for previous_slot in range(slot)
                )
                value_index = (
                    slot - sparse_prefix
                    if name == "dense_values"
                    else sparse_prefix
                )
                for compressed_index in range(result.shape[3]):
                    mask = selected & (value_index == compressed_index)
                    result[:, :, local_group, compressed_index][mask] = 0
        return result

    def move(name: str) -> torch.Tensor:
        return local(name).to(device=device, non_blocking=True).contiguous()

    hardware_metadata = payload["hardware_metadata"]
    if hardware_metadata is not None:
        hardware_metadata = local("hardware_metadata").to(
            device=device, non_blocking=True
        ).contiguous()

    if projection == "w13_weight":
        local_rows = 2 * local_block_count * layout.block_h
        local_shape = (num_local_experts, local_rows, original_columns)
    else:
        local_columns = local_group_count * layout.block_w * layout.block_m
        local_shape = (num_local_experts, original_rows, local_columns)

    return HybridBlockSparseWeight(
        original_shape=local_shape,
        layout=layout,
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
    from sglang.srt.layers.moe.ep_moe.kernels import (
        moe_ep_deepgemm_preprocess,
        moe_ep_sparse_gemm_contiguous_preprocess,
    )

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )
    topk_weights, topk_ids, _ = topk_output

    hidden_states_shape = hidden_states.shape
    hidden_states_dtype = hidden_states.dtype
    hidden_states_device = hidden_states.device

    output_dtype = (
        torch.bfloat16
        if quant_info.dense_w13.dtype == torch.bfloat16
        else torch.float8_e4m3fn
    )
    layout = os.environ.get(_SPARSE_GEMM_LAYOUT_ENV, "auto")
    if layout == "auto":
        contiguous_min_m = int(os.environ.get(_SPARSE_GEMM_CONTIGUOUS_MIN_M_ENV, "4096"))
        if contiguous_min_m < 0:
            raise ValueError(f"{_SPARSE_GEMM_CONTIGUOUS_MIN_M_ENV} must be >= 0")
        use_contiguous = topk_ids.numel() >= contiguous_min_m
    elif layout == "contiguous":
        use_contiguous = True
    elif layout == "masked":
        use_contiguous = False
    else:
        raise ValueError(
            f"{_SPARSE_GEMM_LAYOUT_ENV} must be 'auto', 'masked', or 'contiguous'"
        )

    if use_contiguous:
        if output_dtype != torch.bfloat16:
            raise TypeError("SparseGEMM contiguous layout currently supports BF16 only")
        m_alignment = int(os.environ.get(_SPARSE_GEMM_M_ALIGNMENT_ENV, "128"))
        if m_alignment <= 0 or m_alignment % 64 != 0:
            raise ValueError(
                f"{_SPARSE_GEMM_M_ALIGNMENT_ENV} must be positive and divisible by 64"
            )
        grouped_layout, m_alignment, src2dst, packed_hidden_states = (
            moe_ep_sparse_gemm_contiguous_preprocess(
                topk_ids,
                runner_config.num_local_experts,
                hidden_states,
                runner_config.top_k,
                m_alignment=m_alignment,
                output_dtype=output_dtype,
            )
        )
        hidden_states_scale = None
        masked_m = None
        expected_m = None
        use_masked_gemm = False
    else:
        masked_m_alignment = int(
            os.environ.get(_SPARSE_GEMM_MASKED_M_ALIGNMENT_ENV, "64")
        )
        if masked_m_alignment <= 0 or masked_m_alignment % 64 != 0:
            raise ValueError(
                f"{_SPARSE_GEMM_MASKED_M_ALIGNMENT_ENV} must be positive and "
                "divisible by 64"
            )
        masked_m, expected_m, src2dst, packed_hidden_states, hidden_states_scale = (
            moe_ep_deepgemm_preprocess(
                topk_ids,
                runner_config.num_local_experts,
                hidden_states,
                runner_config.top_k,
                quant_info.block_shape,
                output_dtype=output_dtype,
                m_alignment=masked_m_alignment,
            )
        )
        grouped_layout = None
        m_alignment = 128
        use_masked_gemm = True

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states_shape
    running_state["hidden_states_dtype"] = hidden_states_dtype
    running_state["hidden_states_device"] = hidden_states_device
    running_state["src2dst"] = src2dst

    return SparseGemmRunnerInput(
        hidden_states=packed_hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=use_masked_gemm,
        masked_m=masked_m,
        expected_m=expected_m,
        grouped_layout=grouped_layout,
        m_alignment=m_alignment,
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
