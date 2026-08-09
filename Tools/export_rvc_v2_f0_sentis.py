from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
START_DIR = Path.cwd()

RVC_CANDIDATES = [
    SCRIPT_DIR / "Retrieval-based-Voice-Conversion-WebUI",
    SCRIPT_DIR.parent / "Retrieval-based-Voice-Conversion-WebUI",
    START_DIR,
]

RVC_DIR = next(
    (
        candidate.resolve()
        for candidate in RVC_CANDIDATES
        if (candidate / "webui.py").is_file()
        and (candidate / "infer" / "module" / "models.py").is_file()
    ),
    None,
)

if RVC_DIR is None:
    searched = "\n".join(f"  - {path}" for path in RVC_CANDIDATES)
    raise FileNotFoundError(
        "Retrieval-based-Voice-Conversion-WebUIが見つかりません。\n"
        "必要な構成: webui.py と infer/module/models.py\n"
        f"検索場所:\n{searched}"
    )

sys.path.insert(0, str(RVC_DIR))
os.chdir(RVC_DIR)

print(f"[INFO] RVC directory: {RVC_DIR}", flush=True)

from infer.module.models import SynthesizerTrnMs768NSFsid


class RvcV2F0OnnxWrapper(nn.Module):
    """
    RVC v2 F0 generator wrapper for ONNX.

    The normal infer() method creates latent noise internally.  This wrapper
    receives that noise as `rnd`, which makes the ONNX inputs explicit.
    """

    def __init__(self, generator: SynthesizerTrnMs768NSFsid) -> None:
        super().__init__()
        self.generator = generator

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        sid: torch.Tensor,
        rnd: torch.Tensor,
    ) -> torch.Tensor:
        g = self.generator.emb_g(sid).unsqueeze(-1)

        m_p, logs_p, x_mask = self.generator.enc_p(
            phone,
            pitch,
            phone_lengths,
        )

        z_p = (
            m_p
            + torch.exp(logs_p) * rnd * 0.66666
        ) * x_mask

        z = self.generator.flow(
            z_p,
            x_mask,
            g=g,
            reverse=True,
        )

        audio = self.generator.dec(
            z * x_mask,
            pitchf,
            g=g,
        )

        return audio


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    if "config" not in checkpoint or "weight" not in checkpoint:
        raise ValueError(
            "推論用RVCモデルではありません。"
            "configとweightを持つassets/weights内の.pthを指定してください。"
        )

    version = checkpoint.get("version", "v1")
    if version != "v2":
        raise ValueError(f"RVC v2モデルが必要です。実際: {version}")

    if_f0 = int(checkpoint.get("f0", 1))
    if if_f0 != 1:
        raise ValueError("F0ありモデルが必要です。")

    if "emb_g.weight" not in checkpoint["weight"]:
        raise ValueError("weight内にemb_g.weightがありません。")


def build_generator(
    checkpoint: dict[str, Any],
) -> SynthesizerTrnMs768NSFsid:
    config = list(checkpoint["config"])
    config[-3] = checkpoint["weight"]["emb_g.weight"].shape[0]

    generator = SynthesizerTrnMs768NSFsid(
        *config,
        is_half=False,
    )

    incompatible = generator.load_state_dict(
        checkpoint["weight"],
        strict=False,
    )

    if incompatible.missing_keys:
        print(
            f"[WARN] Missing keys: {len(incompatible.missing_keys)}",
            flush=True,
        )

    if incompatible.unexpected_keys:
        print(
            f"[WARN] Unexpected keys: {len(incompatible.unexpected_keys)}",
            flush=True,
        )

    # Training-only posterior encoder is not used by this wrapper.
    if hasattr(generator, "enc_q"):
        del generator.enc_q

    generator.eval()
    generator.float()
    return generator


def make_dummy_inputs(
    frames: int,
    speaker_id: int,
) -> tuple[torch.Tensor, ...]:
    if frames <= 0:
        raise ValueError("--framesは1以上にしてください。")

    if speaker_id < 0:
        raise ValueError("--speaker-idは0以上にしてください。")

    torch.manual_seed(1234)

    phone = torch.randn(
        1,
        frames,
        768,
        dtype=torch.float32,
    )

    phone_lengths = torch.tensor(
        [frames],
        dtype=torch.int64,
    )

    pitch = torch.randint(
        low=1,
        high=255,
        size=(1, frames),
        dtype=torch.int64,
    )

    pitchf = torch.full(
        (1, frames),
        220.0,
        dtype=torch.float32,
    )

    sid = torch.tensor(
        [speaker_id],
        dtype=torch.int64,
    )

    rnd = torch.randn(
        1,
        192,
        frames,
        dtype=torch.float32,
    )

    return phone, phone_lengths, pitch, pitchf, sid, rnd


def export_onnx(
    wrapper: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {
        "export_params": True,
        "opset_version": 17,
        "do_constant_folding": False,
        "input_names": [
            "phone",
            "phone_lengths",
            "pitch",
            "pitchf",
            "sid",
            "rnd",
        ],
        "output_names": ["audio"],
        "dynamic_axes": None,
        "verbose": False,
    }

    # Prefer the legacy exporter for this model.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            inputs,
            str(output_path),
            **kwargs,
        )


def validate_onnx(output_path: Path) -> None:
    import onnx

    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)

    print("[OK] ONNX checker passed", flush=True)
    print("[INFO] Inputs:", flush=True)

    for value in model.graph.input:
        dims = [
            dim.dim_value if dim.dim_value > 0 else dim.dim_param
            for dim in value.type.tensor_type.shape.dim
        ]
        print(f"  - {value.name}: {dims}", flush=True)

    print("[INFO] Outputs:", flush=True)

    for value in model.graph.output:
        dims = [
            dim.dim_value if dim.dim_value > 0 else dim.dim_param
            for dim in value.type.tensor_type.shape.dim
        ]
        print(f"  - {value.name}: {dims}", flush=True)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()

    return (START_DIR / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an RVC v2 F0 model for Unity Sentis."
    )

    parser.add_argument(
        "--model",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--frames",
        default=200,
        type=int,
    )

    parser.add_argument(
        "--speaker-id",
        default=0,
        type=int,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = resolve_path(args.model)
    output_path = resolve_path(args.output)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"RVCモデルが見つかりません: {model_path}"
        )

    print(f"[1/5] Loading checkpoint: {model_path}", flush=True)
    checkpoint = load_checkpoint(model_path)

    print("[2/5] Validating v2 / F0 model", flush=True)
    validate_checkpoint(checkpoint)

    print("[3/5] Building generator", flush=True)
    generator = build_generator(checkpoint)
    wrapper = RvcV2F0OnnxWrapper(generator).eval()

    print(
        f"[4/5] Exporting fixed {args.frames}-frame ONNX: "
        f"{output_path}",
        flush=True,
    )

    inputs = make_dummy_inputs(
        args.frames,
        args.speaker_id,
    )

    export_onnx(
        wrapper,
        inputs,
        output_path,
    )

    print("[5/5] Validating ONNX", flush=True)
    validate_onnx(output_path)

    if not output_path.is_file():
        raise RuntimeError(
            f"ONNXが作成されませんでした: {output_path}"
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print("", flush=True)
    print(f"[OK] Created: {output_path}", flush=True)
    print(f"[INFO] Size: {size_mb:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()