from pathlib import Path
import sys
import unittest
from unittest.mock import patch

try:
    import torch
    import torch.nn as nn
    import torch.nn.utils.parametrize as parametrize

    HAS_TORCH = True
except ModuleNotFoundError:
    torch = None
    nn = None
    parametrize = None
    HAS_TORCH = False


# Allow importing local src/ modules without packaging.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if HAS_TORCH:
    import quantizers  # noqa: E402
else:
    quantizers = None


if HAS_TORCH:
    class VectorModel(nn.Module):
        """One trainable vector parameter."""

        def __init__(self, values):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    class TwoLayerBlocksModel(nn.Module):
        """Two trainable parameters used to verify global motif selection."""

        def __init__(self, a_values, b_values):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(a_values, dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(b_values, dtype=torch.float32))

    class ConvBiasModel(nn.Module):
        """Trainable conv/linear weights+biases plus frozen/buffer state."""

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 2, kernel_size=3, bias=True)
            self.head = nn.Linear(8, 4, bias=True)
            self.frozen = nn.Parameter(torch.full((5,), 3.0), requires_grad=False)
            self.register_buffer("buf", torch.tensor([7.0, 8.0], dtype=torch.float32))
            with torch.no_grad():
                for p in self.parameters():
                    if p.requires_grad:
                        p.copy_(torch.linspace(0.1, 1.1, steps=p.numel()).view_as(p))

    class SmallQATModel(nn.Module):
        """Single linear layer with fixed weights for exact QAT checks."""

        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(6, 1, bias=False)
            with torch.no_grad():
                self.fc.weight.copy_(torch.tensor([[-2.0, -1.2, -0.2, 0.2, 1.2, 2.0]], dtype=torch.float32))

    class QATExcludeModel(nn.Module):
        """Model with one quantized layer and one excluded norm layer."""

        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4, bias=False)
            self.norm = nn.LayerNorm(4)
else:
    class VectorModel:
        pass

    class TwoLayerBlocksModel:
        pass

    class ConvBiasModel:
        pass

    class SmallQATModel:
        pass

    class QATExcludeModel:
        pass


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for MoMos/QAT tests")
class TestMoMosAndQAT(unittest.TestCase):
    def test_tensor_to_blocks_roundtrip_with_padding(self):
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        blocks, n_params, shape = quantizers.tensor_to_blocks(tensor, block_size=4)
        rebuilt = quantizers.blocks_to_tensor(blocks, n_params=n_params, shape=shape)

        expected_blocks = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(blocks, expected_blocks))
        self.assertEqual(n_params, 5)
        self.assertEqual(shape, torch.Size([5]))
        self.assertTrue(torch.equal(rebuilt, tensor))

    def test_count_total_blocks_ignores_frozen_parameters(self):
        class Mixed(nn.Module):
            def __init__(self):
                super().__init__()
                self.trainable = nn.Parameter(torch.arange(7.0), requires_grad=True)
                self.frozen = nn.Parameter(torch.ones(100), requires_grad=False)

        model = Mixed()
        self.assertEqual(quantizers.count_total_blocks(model, block_size=4), 2)

    def test_nearest_motifs_chunked_matches_torch_cdist_argmin(self):
        blocks = torch.tensor(
            [[1.0, 2.0], [1.2, 2.2], [8.8, 9.9], [9.1, 10.2]],
            dtype=torch.float32,
        )
        motifs = torch.tensor([[1.0, 2.0], [9.0, 10.0]], dtype=torch.float32)

        chunked = quantizers._nearest_motifs_chunked(blocks, motifs, chunk_size=1e-6)
        direct = torch.cdist(blocks, motifs).argmin(dim=1)
        self.assertTrue(torch.equal(chunked, direct))

    def test_momos_selects_motifs_globally_across_all_layers(self):
        # Global block order is [a0, a1, b0, b1]. We force motifs to [b0, b1].
        model = TwoLayerBlocksModel(
            a_values=[1.0, 1.0, 2.0, 2.0],
            b_values=[10.0, 10.0, 20.0, 20.0],
        )

        with patch("quantizers.torch.randperm", return_value=torch.tensor([2, 3, 0, 1], dtype=torch.long)) as mock_randperm:
            out = quantizers.momos(model, block_size=2, k=2, force_zero=False)

        self.assertEqual(mock_randperm.call_count, 1)
        self.assertEqual(int(mock_randperm.call_args.args[0]), 4)  # One global draw over all blocks.
        self.assertTrue(torch.equal(model.a.detach(), torch.tensor([10.0, 10.0, 10.0, 10.0], dtype=torch.float32)))
        self.assertTrue(torch.equal(model.b.detach(), torch.tensor([10.0, 10.0, 20.0, 20.0], dtype=torch.float32)))
        self.assertEqual(out["num_changed_weights"], 4)
        self.assertAlmostEqual(out["distortion"], 290.0, places=7)
        self.assertTrue(torch.equal(out["motif_counts"], torch.tensor([3, 1], dtype=torch.long)))

    def test_momos_metrics_count_only_non_padded_entries(self):
        # 5 params with block_size=4 => one real value in second block + 3 padded zeros.
        model = VectorModel(values=[1.0, 2.0, 3.0, 4.0, 0.0])

        with patch("quantizers.torch.randperm", return_value=torch.tensor([0, 1], dtype=torch.long)):
            out = quantizers.momos(model, block_size=4, k=1, force_zero=False)

        expected_after = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0], dtype=torch.float32)
        self.assertTrue(torch.equal(model.w.detach(), expected_after))
        self.assertEqual(out["num_changed_weights"], 1)
        self.assertAlmostEqual(out["distortion"], 1.0, places=7)

    def test_momos_replaces_blocks_then_trims_padding_on_writeback(self):
        # Blocks are [[1,2,3,4], [5,0,0,0]] with s=4.
        # With k=1 and motif picked as first block, second block is replaced by [1,2,3,4].
        # Only the first value of that block is real after trimming -> final tail becomes 1.
        model = VectorModel(values=[1.0, 2.0, 3.0, 4.0, 5.0])
        with patch("quantizers.torch.randperm", return_value=torch.tensor([0, 1], dtype=torch.long)):
            out = quantizers.momos(model, block_size=4, k=1, force_zero=False)

        self.assertTrue(torch.equal(model.w.detach(), torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0], dtype=torch.float32)))
        self.assertEqual(out["num_changed_weights"], 1)
        self.assertAlmostEqual(out["distortion"], 16.0, places=7)  # (5 - 1)^2

    def test_momos_padding_reconstruction_is_exact_for_uneven_two_layer_model(self):
        class UnevenTwoLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))
                self.b = nn.Parameter(torch.tensor([9.0, 9.0, 9.0, 9.0, 2.0], dtype=torch.float32))

        model = UnevenTwoLayer()
        # Global blocks (s=4): a0=[1,1,1,0], b0=[9,9,9,9], b1=[2,0,0,0]
        # Force motif to b0 by making randperm start with index 1.
        with patch("quantizers.torch.randperm", return_value=torch.tensor([1, 0, 2], dtype=torch.long)):
            out = quantizers.momos(model, block_size=4, k=1, force_zero=False)

        self.assertTrue(torch.equal(model.a.detach(), torch.tensor([9.0, 9.0, 9.0], dtype=torch.float32)))
        self.assertTrue(torch.equal(model.b.detach(), torch.tensor([9.0, 9.0, 9.0, 9.0, 9.0], dtype=torch.float32)))
        self.assertEqual(out["num_changed_weights"], 4)  # 3 in a, 1 in b tail
        self.assertAlmostEqual(out["distortion"], 241.0, places=7)  # 3*(9-1)^2 + (9-2)^2

    def test_momos_force_zero_with_padding_replaces_near_zero_tail_with_zero(self):
        # Blocks (s=2): [0.01,-0.01], [8,8], [0.02,0]
        # With force_zero and motif [8,8], near-zero blocks go to zero motif.
        model = VectorModel(values=[0.01, -0.01, 8.0, 8.0, 0.02])
        with patch("quantizers.torch.randperm", return_value=torch.tensor([1, 0, 2], dtype=torch.long)):
            out = quantizers.momos(model, block_size=2, k=2, force_zero=True)

        self.assertTrue(torch.equal(model.w.detach(), torch.tensor([0.0, 0.0, 8.0, 8.0, 0.0], dtype=torch.float32)))
        self.assertEqual(out["num_changed_weights"], 3)
        self.assertAlmostEqual(out["distortion"], 0.0006, places=9)
        self.assertTrue(torch.equal(out["motif_counts"], torch.tensor([2, 1], dtype=torch.long)))

    def test_momos_chunked_and_non_chunked_assignment_match(self):
        values = [1.0, 2.0, 9.0, 10.0, 1.1, 2.1, 8.9, 10.1]
        model_small_chunk = VectorModel(values=values)
        model_large_chunk = VectorModel(values=values)
        idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        with patch("quantizers.torch.randperm", return_value=idx):
            out_small = quantizers.momos(model_small_chunk, block_size=2, k=2, force_zero=False, chunk_size=1e-6)
        with patch("quantizers.torch.randperm", return_value=idx):
            out_large = quantizers.momos(
                model_large_chunk,
                block_size=2,
                k=2,
                force_zero=False,
                chunk_size=4096,
            )

        self.assertTrue(torch.equal(model_small_chunk.w.detach(), model_large_chunk.w.detach()))
        self.assertEqual(out_small["num_changed_weights"], out_large["num_changed_weights"])
        self.assertAlmostEqual(out_small["distortion"], out_large["distortion"], places=7)
        self.assertTrue(torch.equal(out_small["motif_counts"], out_large["motif_counts"]))

    def test_momos_updates_trainable_weights_and_biases_not_frozen_or_buffers(self):
        model = ConvBiasModel()
        before_frozen = model.frozen.detach().clone()
        before_buf = model.buf.detach().clone()

        _ = quantizers.momos(model, block_size=3, k=1, force_zero=True)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.assertEqual(torch.count_nonzero(param.detach()).item(), 0, msg=f"{name} should be zeroed")
        self.assertTrue(torch.equal(model.frozen.detach(), before_frozen))
        self.assertTrue(torch.equal(model.buf.detach(), before_buf))

    def test_qat_attaches_fake_quant_and_keeps_original_parameter_weights(self):
        model = SmallQATModel()
        out = quantizers.quantize_qat(model, {"method": "qat", "q": 2, "exclude_layers": ["bn", "norm"]})

        self.assertTrue(out["qat_enabled"])
        self.assertEqual(out["q_bits"], 2)
        self.assertTrue(parametrize.is_parametrized(model.fc, "weight"))

        expected_quantized_view = torch.tensor([[-2.0, -2.0, 0.0, 0.0, 2.0, 2.0]], dtype=torch.float32)
        expected_original = torch.tensor([[-2.0, -1.2, -0.2, 0.2, 1.2, 2.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(model.fc.weight.detach(), expected_quantized_view))
        self.assertTrue(torch.equal(model.fc.parametrizations.weight.original.detach(), expected_original))

    def test_qat_exclude_layers_uses_name_matching(self):
        model = QATExcludeModel()
        _ = quantizers.quantize_qat(model, {"method": "qat", "q": 4, "exclude_layers": ["norm"]})

        self.assertTrue(parametrize.is_parametrized(model.fc, "weight"))
        self.assertFalse(parametrize.is_parametrized(model.norm, "weight"))

    def test_prepare_qat_q32_disables_existing_fake_quant(self):
        model = SmallQATModel()
        _ = quantizers.prepare_qat(model, {"q": 2, "exclude_layers": []})
        out = quantizers.prepare_qat(model, {"q": 32})

        self.assertFalse(out["qat_enabled"])
        self.assertGreaterEqual(out["disabled_modules"], 1)
        fq = model.fc.parametrizations.weight[0]
        self.assertFalse(fq.enabled)
        self.assertTrue(torch.equal(model.fc.weight.detach(), model.fc.parametrizations.weight.original.detach()))

    def test_momos_plus_qat_attaches_qat_and_does_not_hard_project_weights(self):
        model = SmallQATModel()
        _ = quantizers.prepare_qat(model, {"q": 2, "exclude_layers": ["bn", "norm"]})

        with patch("quantizers.torch.randperm", return_value=torch.tensor([0, 1, 2], dtype=torch.long)):
            out = quantizers.quantize_momos(
                model,
                {
                    "method": "momos",
                    "s": 2,
                    "k": 2,
                    "q": 2,
                    "force_zero": False,
                },
            )

        self.assertEqual(out["num_changed_weights"], 2)
        self.assertTrue(torch.equal(out["motif_counts"], torch.tensor([1, 2], dtype=torch.long)))

        # Underlying trainable weights are MoMos-projected (not hard precision projected).
        expected_original_after_momos = torch.tensor(
            [[-2.0, -1.2, -0.2, 0.2, -0.2, 0.2]],
            dtype=torch.float32,
        )
        self.assertTrue(
            torch.equal(
                model.fc.parametrizations.weight.original.detach(),
                expected_original_after_momos,
            )
        )

        # Forward view still reflects QAT fake quantization.
        expected_qat_view = torch.tensor([[-2.0, -2.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(model.fc.weight.detach(), expected_qat_view))

    def test_quantize_dispatch_includes_method_and_elapsed_time(self):
        model = VectorModel(values=[1.0, 2.0, 3.0, 4.0])
        out = quantizers.quantize(
            model,
            {"method": "MoMoS", "s": 2, "k": 1, "q": 32, "force_zero": True},
        )

        self.assertEqual(out["method"], "momos")
        self.assertIn("q_time", out)
        self.assertGreaterEqual(out["q_time"], 0.0)
        self.assertIsNone(quantizers.quantize(model, None))


if __name__ == "__main__":
    unittest.main()
