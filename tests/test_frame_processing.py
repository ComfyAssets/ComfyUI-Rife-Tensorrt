import torch
import pytest

from vfi_utilities import preprocess_frames, postprocess_frames


class TestPreprocessFrames:
    def test_nhwc_to_nchw(self):
        frames = torch.rand(4, 64, 64, 3)
        result = preprocess_frames(frames)
        assert result.shape == (4, 3, 64, 64)

    def test_extra_channels_dropped(self):
        frames = torch.rand(2, 32, 32, 4)
        result = preprocess_frames(frames)
        assert result.shape == (2, 3, 32, 32)

    def test_single_frame(self):
        frames = torch.rand(1, 128, 256, 3)
        result = preprocess_frames(frames)
        assert result.shape == (1, 3, 128, 256)

    def test_values_preserved(self):
        frames = torch.zeros(1, 2, 2, 3)
        frames[0, 0, 0, 0] = 0.5  # R at (0,0)
        frames[0, 0, 0, 1] = 0.7  # G at (0,0)
        frames[0, 0, 0, 2] = 0.9  # B at (0,0)
        result = preprocess_frames(frames)
        assert result[0, 0, 0, 0].item() == pytest.approx(0.5)  # R channel
        assert result[0, 1, 0, 0].item() == pytest.approx(0.7)  # G channel
        assert result[0, 2, 0, 0].item() == pytest.approx(0.9)  # B channel


class TestPostprocessFrames:
    def test_nchw_to_nhwc(self):
        frames = torch.rand(4, 3, 64, 64)
        result = postprocess_frames(frames)
        assert result.shape == (4, 64, 64, 3)

    def test_output_on_cpu(self):
        frames = torch.rand(2, 3, 32, 32)
        result = postprocess_frames(frames)
        assert result.device == torch.device("cpu")

    def test_single_frame(self):
        frames = torch.rand(1, 3, 128, 256)
        result = postprocess_frames(frames)
        assert result.shape == (1, 128, 256, 3)


class TestRoundtrip:
    def test_preprocess_postprocess_roundtrip(self):
        original = torch.rand(3, 64, 64, 3)
        processed = preprocess_frames(original)
        restored = postprocess_frames(processed)
        assert restored.shape == original.shape
        assert torch.allclose(restored, original, atol=1e-6)
