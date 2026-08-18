import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.sparse_gemm import (
    SparseGemmMoeQuantInfo,
    _payload_to_sparse_weight,
    load_sparse_gemm_moe_weight,
)
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod


class TestSparseGemmUnquantizedWeights(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_topk_one_masked_preprocess(self):
        from sglang.srt.layers.moe.ep_moe.kernels import (
            moe_ep_deepgemm_preprocess,
        )

        hidden_states = torch.arange(
            128, device="cuda", dtype=torch.bfloat16
        ).reshape(1, 128)
        topk_ids = torch.tensor([[3]], device="cuda", dtype=torch.int32)

        masked_m, expected_m, src2dst, packed, scale = moe_ep_deepgemm_preprocess(
            topk_ids,
            num_local_experts=4,
            hidden_states=hidden_states,
            top_k=1,
            block_shape=None,
            output_dtype=torch.bfloat16,
            m_alignment=64,
        )
        torch.cuda.synchronize()

        self.assertEqual(expected_m, 1)
        self.assertIsNone(scale)
        self.assertEqual(tuple(packed.shape), (4, 64, 128))
        self.assertEqual(masked_m.tolist(), [0, 0, 0, 1])
        torch.testing.assert_close(
            packed.view(-1, 128)[src2dst.long()], hidden_states
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_topk_many_masked_preprocess(self):
        from sglang.srt.layers.moe.ep_moe.kernels import (
            moe_ep_deepgemm_preprocess,
        )

        hidden_states = torch.arange(
            2 * 128, device="cuda", dtype=torch.bfloat16
        ).reshape(2, 128)
        topk_ids = torch.tensor(
            [list(range(8)), list(reversed(range(8)))],
            device="cuda",
            dtype=torch.int32,
        )

        masked_m, expected_m, src2dst, packed, scale = moe_ep_deepgemm_preprocess(
            topk_ids,
            num_local_experts=8,
            hidden_states=hidden_states,
            top_k=8,
            block_shape=None,
            output_dtype=torch.bfloat16,
            m_alignment=64,
        )
        torch.cuda.synchronize()

        self.assertEqual(expected_m, 2)
        self.assertIsNone(scale)
        self.assertEqual(tuple(packed.shape), (8, 64, 128))
        self.assertEqual(masked_m.tolist(), [2] * 8)
        torch.testing.assert_close(
            packed.view(-1, 128)[src2dst.long()],
            hidden_states.repeat_interleave(8, dim=0),
        )

    def test_create_weights_uses_zero_size_checkpoint_sinks(self):
        layer = torch.nn.Module()
        method = UnquantizedFusedMoEMethod()
        method.use_sparse_gemm = True

        method.create_weights(
            layer=layer,
            num_experts=128,
            hidden_size=7168,
            intermediate_size_per_partition=768,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )

        self.assertEqual(layer.w13_weight.numel(), 0)
        self.assertEqual(layer.w2_weight.numel(), 0)
        layer.w13_weight.weight_loader(
            layer.w13_weight,
            torch.ones(2, 2),
            "unused",
            shard_id="w1",
            expert_id=0,
        )
        self.assertEqual(layer.w13_weight.numel(), 0)

    def test_manifest_expert_count_must_match_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "weights"
            weights.mkdir()
            payload_path = weights / "layer_000_w13_weight.pt"
            torch.save({"original_shape": [60, 16, 8]}, payload_path)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "weights": [
                            {
                                "logical_name": "model.layers.0.mlp.experts.w13_weight",
                                "file": "weights/layer_000_w13_weight.pt",
                            }
                        ]
                    }
                )
            )

            with patch.dict(
                "os.environ", {"SGLANG_SPARSE_GEMM_MOE_PATH": str(root)}
            ):
                with self.assertRaisesRegex(ValueError, "contains 60 experts"):
                    load_sparse_gemm_moe_weight(
                        layer_id=0,
                        projection="w13_weight",
                        device=torch.device("cpu"),
                        num_global_experts=64,
                    )

    def test_tp2_down_shard_pads_odd_eleven_block_partition(self):
        class FakeLayout:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeWeight:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        sparse_module = ModuleType("sparse_gemm.hybrid_sparse")
        sparse_module.HybridBlockSparseLayout = FakeLayout
        sparse_module.HybridBlockSparseWeight = FakeWeight
        sparse_package = ModuleType("sparse_gemm")
        sparse_package.hybrid_sparse = sparse_module
        payload = {
            "original_shape": [1, 64, 1408],
            "layout": {"block_h": 64, "block_w": 64, "block_n": 1, "block_m": 2},
            "block_selector": torch.zeros(1, 1, 11, dtype=torch.int32),
            "dense_values": torch.ones(1, 1, 11, 2, 1),
            "sparse_values": torch.ones(1, 1, 11, 2, 1),
            "sparse_metadata": torch.zeros(1, 1, 11, 2, 1),
            "hardware_metadata": None,
        }

        with patch.dict(
            "sys.modules",
            {
                "sparse_gemm": sparse_package,
                "sparse_gemm.hybrid_sparse": sparse_module,
            },
        ):
            rank0 = _payload_to_sparse_weight(
                payload,
                torch.device("cpu"),
                projection="down_proj",
                moe_tp_rank=0,
                moe_tp_size=2,
            )
            rank1 = _payload_to_sparse_weight(
                payload,
                torch.device("cpu"),
                projection="down_proj",
                moe_tp_rank=1,
                moe_tp_size=2,
            )

        self.assertEqual(rank0.original_shape, (1, 64, 768))
        self.assertEqual(rank1.original_shape, (1, 64, 768))
        self.assertTrue(torch.equal(rank0.dense_values[0, 0, -1, 1], torch.zeros(1)))
        self.assertTrue(torch.equal(rank1.dense_values[0, 0, 0, 0], torch.zeros(1)))

        down = SimpleNamespace(
            original_shape=(1, 64, 768),
            layout=SimpleNamespace(block_w=64, block_m=2),
        )
        rank0_info = SparseGemmMoeQuantInfo(None, down, moe_tp_rank=0)
        rank1_info = SparseGemmMoeQuantInfo(None, down, moe_tp_rank=1)
        self.assertEqual(rank0_info.down_input_padding(704), (0, 64))
        self.assertEqual(rank1_info.down_input_padding(704), (64, 0))


if __name__ == "__main__":
    unittest.main()
