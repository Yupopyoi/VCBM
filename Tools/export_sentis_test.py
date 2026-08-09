from pathlib import Path

import torch
from torch import nn


class SentisTestModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 + 1.0


def main() -> None:
    folder_path = Path("../../Assets/Models")
    output_path = Path("../../Assets/Models/sentis_test.onnx").resolve()

    if not folder_path.exists():
        folder_path.mkdir(parents=True)

    model = SentisTestModel()
    model.eval()

    dummy_input = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]],
        dtype=torch.float32,
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes=None,
        do_constant_folding=True,
    )

    print(f"[OK] ONNX exported: {output_path}")


if __name__ == "__main__":
    main()