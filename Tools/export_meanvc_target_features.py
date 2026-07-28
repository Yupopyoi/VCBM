#!/usr/bin/env python
"""
Export MeanVC target features from an arbitrary WAV file.

Outputs:
  speaker_embedding.pt : float32 CPU tensor, normally [1, 256]
  prompt_mel.pt        : float32 CPU tensor, [1, frames, 80]

This reproduces the target feature preparation used by
ASLP-lab/MeanVC src/runtime/run_rt.py and saves the tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn


SAMPLE_RATE = 16000


def amp_to_db(x: torch.Tensor, min_level_db: float) -> torch.Tensor:
    min_level = np.exp(min_level_db / 20.0 * np.log(10.0))
    floor = torch.ones_like(x) * min_level
    return 20.0 * torch.log10(torch.maximum(floor, x))


def normalize_mel(
    spectrogram: torch.Tensor,
    max_abs_value: float,
    min_db: float,
) -> torch.Tensor:
    value = (
        (2.0 * max_abs_value)
        * ((spectrogram - min_db) / (-min_db))
        - max_abs_value
    )
    return torch.clamp(value, -max_abs_value, max_abs_value)


class MeanVCMelSpectrogram(nn.Module):
    """Mel extraction equivalent to MeanVC's runtime implementation."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_size: int = 640,
        hop_length: int = 160,
        n_mels: int = 80,
        fmin: int = 0,
        fmax: int = 8000,
        center: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.center = center

        self._mel_basis: dict[str, torch.Tensor] = {}
        self._hann_window: dict[str, torch.Tensor] = {}

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        dtype_device = f"{waveform.dtype}_{waveform.device}"
        mel_key = f"{self.fmax}_{dtype_device}"
        window_key = f"{self.win_size}_{dtype_device}"

        if mel_key not in self._mel_basis:
            mel = librosa_mel_fn(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
            )
            self._mel_basis[mel_key] = torch.from_numpy(mel).to(
                dtype=waveform.dtype,
                device=waveform.device,
            )

        if window_key not in self._hann_window:
            self._hann_window[window_key] = torch.hann_window(
                self.win_size,
                dtype=waveform.dtype,
                device=waveform.device,
            )

        # MeanVC's source currently uses return_complex=False.
        # The complex form below computes the same magnitude while avoiding
        # the deprecated real/imag-last-dimension representation.
        stft = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_size,
            window=self._hann_window[window_key],
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        magnitude = torch.sqrt(
            stft.real.pow(2) + stft.imag.pow(2) + 1e-6
        )
        mel_spec = torch.matmul(self._mel_basis[mel_key], magnitude)
        mel_spec = amp_to_db(mel_spec, -115.0) - 20.0
        return normalize_mel(mel_spec, 1.0, -115.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create speaker_embedding.pt and prompt_mel.pt "
            "from a target WAV for MeanVC."
        )
    )
    parser.add_argument(
        "--wav",
        required=True,
        type=Path,
        help="Input WAV path.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory in which the two .pt files are written.",
    )
    parser.add_argument(
        "--meanvc-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "MeanVC repository root. The default assumes this script is "
            "placed directly in the MeanVC repository."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "wavlm_large_finetune.pth path. By default, use "
            "<MeanVC root>/src/runtime/speaker_verification/ckpt/"
            "wavlm_large_finetune.pth."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Feature extraction device. Default: auto.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="PyTorch CPU thread count. Default: 1, matching MeanVC runtime.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but CUDA is unavailable."
        )

    return torch.device(name)


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch versions.
        return torch.load(path, map_location="cpu")


def main() -> int:
    args = parse_args()

    wav_path = args.wav.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    meanvc_root = args.meanvc_root.expanduser().resolve()

    if not wav_path.is_file():
        raise FileNotFoundError(f"Input WAV was not found: {wav_path}")

    if not meanvc_root.is_dir():
        raise FileNotFoundError(
            f"MeanVC repository root was not found: {meanvc_root}"
        )

    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else meanvc_root
        / "src"
        / "runtime"
        / "speaker_verification"
        / "ckpt"
        / "wavlm_large_finetune.pth"
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            "WavLM speaker-verification checkpoint was not found:\n"
            f"  {checkpoint}"
        )

    speaker_path = out_dir / "speaker_embedding.pt"
    prompt_path = out_dir / "prompt_mel.pt"
    manifest_path = out_dir / "target_features.json"

    existing = [
        path
        for path in (speaker_path, prompt_path, manifest_path)
        if path.exists()
    ]
    if existing and not args.force:
        names = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Output already exists. Use --force to overwrite:\n" + names
        )

    # verification.py imports from the repository's src package.
    sys.path.insert(0, str(meanvc_root))
    os.chdir(meanvc_root)

    from src.runtime.speaker_verification.verification import (  # noqa: E402
        init_model,
    )

    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)

    print(f"Input WAV : {wav_path}")
    print(f"Output dir: {out_dir}")
    print(f"MeanVC    : {meanvc_root}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Device    : {device}")

    waveform_np, _ = librosa.load(
        wav_path,
        sr=SAMPLE_RATE,
        mono=True,
        dtype=np.float32,
    )
    waveform_np = np.asarray(waveform_np, dtype=np.float32)

    if waveform_np.size == 0:
        raise ValueError("The input WAV contains no samples.")

    if not np.isfinite(waveform_np).all():
        raise ValueError("The input WAV contains NaN or infinite samples.")

    duration = waveform_np.size / float(SAMPLE_RATE)
    waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(device)

    print(f"Duration  : {duration:.3f} s")
    print(f"Peak      : {float(np.max(np.abs(waveform_np))):.6f}")
    print("Loading WavLM speaker-verification model...")

    speaker_model = init_model("wavlm_large", str(checkpoint))
    speaker_model.eval()
    speaker_model.to(device)

    mel_extractor = MeanVCMelSpectrogram().to(device)

    print("Extracting target features...")
    with torch.inference_mode():
        speaker_embedding = speaker_model(waveform)
        prompt_mel = mel_extractor(waveform).transpose(1, 2)

    # Save simple CPU float32 tensors for later LibTorch loading.
    speaker_embedding = (
        speaker_embedding.detach().to("cpu", torch.float32).contiguous()
    )
    prompt_mel = prompt_mel.detach().to(
        "cpu",
        torch.float32,
    ).contiguous()

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(speaker_embedding, speaker_path)
    torch.save(prompt_mel, prompt_path)

    manifest = {
        "source_wav": str(wav_path),
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": duration,
        "speaker_embedding": {
            "file": speaker_path.name,
            "shape": list(speaker_embedding.shape),
            "dtype": str(speaker_embedding.dtype),
        },
        "prompt_mel": {
            "file": prompt_path.name,
            "shape": list(prompt_mel.shape),
            "dtype": str(prompt_mel.dtype),
        },
        "meanvc_checkpoint": str(checkpoint),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Read back immediately to catch damaged/incompatible output.
    loaded_speaker = safe_torch_load(speaker_path)
    loaded_prompt = safe_torch_load(prompt_path)

    if not isinstance(loaded_speaker, torch.Tensor):
        raise TypeError("speaker_embedding.pt did not reload as a tensor.")
    if not isinstance(loaded_prompt, torch.Tensor):
        raise TypeError("prompt_mel.pt did not reload as a tensor.")

    print()
    print("Export completed.")
    print(
        f"speaker_embedding.pt: "
        f"shape={tuple(speaker_embedding.shape)}, "
        f"dtype={speaker_embedding.dtype}"
    )
    print(
        f"prompt_mel.pt       : "
        f"shape={tuple(prompt_mel.shape)}, "
        f"dtype={prompt_mel.dtype}"
    )
    print(f"Saved to: {out_dir}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exception:
        print(f"\nERROR: {exception}", file=sys.stderr)
        raise SystemExit(1)
