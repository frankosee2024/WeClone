import os
import torch
import soundfile as sf

try:
    from gptsovits import Synthesizer
except ImportError as e:
    raise ImportError(
        "gptsovits library not installed. Please install with `pip install gptsovits4`"
    ) from e

class GPTSoVITS4:
    """Wrapper for GPT-SoVITS4 voice cloning model."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.device = device
        self.model = Synthesizer(model_dir, device=device)

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    @torch.no_grad()
    def inference(self, text: str, reference_audio: str):
        """Generate speech using GPT-SoVITS4."""
        if not os.path.isfile(reference_audio):
            raise FileNotFoundError(reference_audio)
        wav = self.model.tts(text=text, ref_path=reference_audio)
        return wav

    def save(self, wav, path: str):
        """Save audio to a file."""
        sf.write(path, wav, self.sample_rate)
