import os
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models.llama4 import Llama4SharedExpert


class TestLlama4SparseSharedExpert(unittest.TestCase):
    def test_uses_sparse_gemm_weights_at_configured_threshold(self):
        expert = Llama4SharedExpert.__new__(Llama4SharedExpert)
        nn.Module.__init__(expert)
        expert.act_fn = nn.Identity()
        expert.set_sparse_gemm_weights("gate_up", "down")

        calls = []

        def sparse_gemm(x, weight):
            calls.append(weight)
            return x

        sparse_module = types.ModuleType("sparse_gemm")
        hybrid_module = types.ModuleType("sparse_gemm.hybrid_sparse")
        hybrid_module.hybrid_block_sparse_gemm_wgmma_tuned = sparse_gemm
        sparse_module.hybrid_sparse = hybrid_module

        with (
            patch.dict(
                sys.modules,
                {
                    "sparse_gemm": sparse_module,
                    "sparse_gemm.hybrid_sparse": hybrid_module,
                },
            ),
            patch.dict(os.environ, {"SGLANG_SPARSE_GEMM_SHARED_MIN_M": "0"}),
        ):
            output = expert(torch.ones(2, 4))

        self.assertEqual(calls, ["gate_up", "down"])
        self.assertEqual(output.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
