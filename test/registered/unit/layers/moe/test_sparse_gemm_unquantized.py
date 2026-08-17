import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.sparse_gemm import (
    load_sparse_gemm_moe_weight,
)
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod


class TestSparseGemmUnquantizedWeights(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
