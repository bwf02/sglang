"""Run on the remote GPU with the SGLang Python package available."""

import unittest

import torch

from sglang.srt.layers.moe.ep_moe.kernels import moe_ep_deepgemm_preprocess


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class SlideSparseCapacityTest(unittest.TestCase):
    def test_capacity_preserves_all_routed_rows(self):
        for tokens, skewed in ((8, False), (1024, False), (1024, True)):
            with self.subTest(tokens=tokens, skewed=skewed):
                x = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
                ids = torch.arange(tokens * 2, device="cuda", dtype=torch.int32).reshape(tokens, 2) % 8
                if skewed:
                    ids[:, 0] = 0
                    ids[:, 1] = 1
                counts, _, offsets, packed, _ = moe_ep_deepgemm_preprocess(
                    ids, 8, x, 2, None, output_dtype=torch.bfloat16,
                    batched_capacity=True,
                )
                capacity = tokens * 2 if tokens <= 512 else int(counts.max())
                self.assertEqual(packed.shape[1], max(64, (capacity + 63) // 64 * 64))
                torch.testing.assert_close(
                    packed.flatten(0, 1)[offsets.long()], x.repeat_interleave(2, dim=0)
                )


if __name__ == "__main__":
    unittest.main()
