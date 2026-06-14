"""
Neural audio codec wrapper around Descript Audio Codec (DAC).

DAC encodes audio into continuous latents before quantization — these
are the "latent tokens" the DiT operates on. Working pre-quantization
gives the model a richer gradient signal than discrete codes.

44kHz model: latent_dim=64, hop_length=512 samples/token (~86 tokens/sec)
"""

import torch
import dac
from dac.utils import load_model

SAMPLE_RATE = 44100
LATENT_DIM  = 1024   # DAC 44kHz encoder output dimension


class AudioCodec:
    def __init__(self, model_type: str = "44khz", device: str = "cpu"):
        self.device = device
        model_path = dac.utils.download(model_type=model_type)
        self.model  = dac.DAC.load(model_path).to(device).eval()
        self.hop    = self.model.hop_length  # 512 samples → 1 token at 44kHz

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio : (B, 1, T)  float32  [-1, 1]  at SAMPLE_RATE
        returns (B, T_tok, LATENT_DIM)  continuous pre-quant latents
        """
        audio = audio.to(self.device)
        # DAC forward up to the encoder
        x     = self.model.preprocess(audio, SAMPLE_RATE)
        z     = self.model.encoder(x)          # (B, LATENT_DIM, T_tok)
        return z.permute(0, 2, 1).contiguous() # (B, T_tok, LATENT_DIM)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z      : (B, T_tok, LATENT_DIM)
        returns: (B, 1, T)  audio
        """
        z = z.permute(0, 2, 1).contiguous()  # (B, LATENT_DIM, T_tok)
        # Quantize then decode
        z_q, _, _, _, _ = self.model.quantizer(z, None)
        audio = self.model.decoder(z_q)
        return audio

    def secs_to_tokens(self, seconds: float) -> int:
        return int(seconds * SAMPLE_RATE / self.hop)

    def tokens_to_secs(self, tokens: int) -> float:
        return tokens * self.hop / SAMPLE_RATE
