"""
Export the current RVC v2 HuBERT/ContentVec model to a fixed-input ONNX
for Unity Inference Engine / Sentis.

Current RVC model layout:
    assets/
      hubert_base/
        config.json
        preprocessor_config.json
        pytorch_model.bin

Interface:
    input:
        audio     float32 [1, 32000]   # mono 16 kHz, 2 seconds

    output:
        features  float32 [1, T, 768]  # RVC v2 HuBERT features

Usage:
    python export_hubert_rvc_v2_transformers.py ^
        assets/hubert_base ^
        hubert_rvc_v2_2s.onnx
"""

import argparse
import json
import os

import torch
from transformers import HubertModel

INPUT_SAMPLES = 32000


class RvcV2HubertWrapper(torch.nn.Module):
    def __init__(self, hubert: HubertModel, do_normalize: bool):
        super().__init__()
        self.hubert = hubert
        self.do_normalize = do_normalize

    def forward(self, audio):
        if self.do_normalize:
            mean = audio.mean(dim=-1, keepdim=True)
            var = ((audio - mean) ** 2).mean(dim=-1, keepdim=True)
            audio = (audio - mean) / torch.sqrt(var + 1e-7)

        outputs = self.hubert(
            input_values=audio,
            attention_mask=None,
            output_hidden_states=False,
            return_dict=True,
        )

        # Current RVC v2 uses the final 12th-layer 768-D representation directly.
        return outputs.last_hidden_state


def read_do_normalize(model_dir: str) -> bool:
    path = os.path.join(model_dir, "preprocessor_config.json")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return bool(cfg.get("do_normalize", True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hubert_dir", help="Path to assets/hubert_base")
    parser.add_argument("output_onnx", help="Output ONNX path")
    args = parser.parse_args()

    hubert_dir = os.path.abspath(args.hubert_dir)
    output_onnx = os.path.abspath(args.output_onnx)

    for name in (
        "config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",
    ):
        path = os.path.join(hubert_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    output_dir = os.path.dirname(output_onnx)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    do_normalize = read_do_normalize(hubert_dir)

    print("[INFO] HuBERT directory:", hubert_dir)
    print("[INFO] do_normalize:", do_normalize)
    print("[INFO] Loading HuBERT...")

    hubert = HubertModel.from_pretrained(
        hubert_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    hubert = hubert.float().eval()

    wrapper = RvcV2HubertWrapper(hubert, do_normalize).eval()

    dummy = torch.zeros(1, INPUT_SAMPLES, dtype=torch.float32)

    print("[INFO] Testing PyTorch forward...")
    with torch.no_grad():
        features = wrapper(dummy)

    print("[INFO] PyTorch output shape:", tuple(features.shape))

    if features.ndim != 3 or features.shape[0] != 1 or features.shape[2] != 768:
        raise RuntimeError(
            f"Expected [1,T,768], got {tuple(features.shape)}"
        )

    print("[INFO] Exporting ONNX...")

    torch.onnx.export(
        wrapper,
        dummy,
        output_onnx,
        input_names=["audio"],
        output_names=["features"],
        opset_version=17,
        do_constant_folding=True,
    )

    if not os.path.isfile(output_onnx):
        raise RuntimeError(
            f"ONNX export finished but file was not created: {output_onnx}"
        )

    print()
    print("[OK] HuBERT ONNX was created:")
    print(output_onnx)
    print("[INFO] Input : audio    [1, 32000]")
    print("[INFO] Output: features [1, T, 768]")


if __name__ == "__main__":
    main()