"""
Export RVC RMVPE to a Sentis-compatible fixed-length ONNX.

Why this exists:
- Upstream RMVPE contains a bidirectional torch.nn.GRU.
- Unity Sentis does not support the ONNX GRU operator.
- This exporter replaces torch.nn.GRU with an explicit GRU implementation
  built from MatMul/Add/Sigmoid/Tanh and unrolls the fixed 224-frame sequence.

Input:
    mel: float32 [1, 128, 224]

Output:
    hidden: float32 [1, 224, 360]

The mel spectrogram is intentionally kept OUTSIDE this ONNX.
Unity/C# will generate the RMVPE-compatible mel features later.
"""

import argparse
import copy
import glob
import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplicitGRU(nn.Module):
    """
    Drop-in inference-only replacement for nn.GRU.

    Supports the configuration used by RVC RMVPE:
      - batch_first=True
      - bidirectional=True
      - fixed sequence length during ONNX tracing

    The equations match PyTorch GRU gate order: reset, update, new.
    """

    def __init__(self, source: nn.GRU):
        super().__init__()

        if not source.batch_first:
            raise ValueError("This exporter expects batch_first=True.")
        if source.proj_size != 0:
            raise ValueError("GRU projection is not supported.")

        self.input_size = source.input_size
        self.hidden_size = source.hidden_size
        self.num_layers = source.num_layers
        self.bidirectional = source.bidirectional
        self.num_directions = 2 if source.bidirectional else 1
        self.batch_first = True

        # Store the original trained weights as buffers.
        # Names avoid dots because register_buffer does not allow them.
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                suffix = "_reverse" if direction == 1 else ""

                for base in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    name = f"{base}_l{layer}{suffix}"
                    value = getattr(source, name).detach().clone()
                    self.register_buffer(name, value)

    def _weights(self, layer: int, direction: int):
        suffix = "_reverse" if direction == 1 else ""
        w_ih = getattr(self, f"weight_ih_l{layer}{suffix}")
        w_hh = getattr(self, f"weight_hh_l{layer}{suffix}")
        b_ih = getattr(self, f"bias_ih_l{layer}{suffix}")
        b_hh = getattr(self, f"bias_hh_l{layer}{suffix}")
        return w_ih, w_hh, b_ih, b_hh

    def _cell(self, x_t, h, w_ih, w_hh, b_ih, b_hh):
        # PyTorch GRU:
        # r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
        # z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
        # n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
        # h' = (1-z)*n + z*h
        gi = F.linear(x_t, w_ih, b_ih)
        gh = F.linear(h, w_hh, b_hh)

        hs = self.hidden_size

        i_r = gi[:, 0:hs]
        i_z = gi[:, hs:2 * hs]
        i_n = gi[:, 2 * hs:3 * hs]

        h_r = gh[:, 0:hs]
        h_z = gh[:, hs:2 * hs]
        h_n = gh[:, 2 * hs:3 * hs]

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)

        return (1.0 - z) * n + z * h

    def _run_direction(self, x, layer: int, direction: int):
        batch = x.shape[0]
        seq_len = x.shape[1]

        # Fixed batch/sequence during tracing is intentional.
        h = torch.zeros(
            (batch, self.hidden_size),
            dtype=x.dtype,
            device=x.device,
        )

        w_ih, w_hh, b_ih, b_hh = self._weights(layer, direction)

        outputs = []

        if direction == 0:
            for t in range(seq_len):
                h = self._cell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)
                outputs.append(h)
        else:
            # Process backwards, then restore chronological output order.
            reverse_outputs = []
            for t in range(seq_len - 1, -1, -1):
                h = self._cell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)
                reverse_outputs.append(h)
            outputs = list(reversed(reverse_outputs))

        y = torch.stack(outputs, dim=1)
        return y, h

    def forward(self, x, hx=None):
        if hx is not None:
            raise ValueError("RMVPE does not use an explicit initial GRU state.")

        layer_input = x
        final_hidden = []

        for layer in range(self.num_layers):
            y_forward, h_forward = self._run_direction(
                layer_input, layer, 0
            )
            final_hidden.append(h_forward)

            if self.bidirectional:
                y_reverse, h_reverse = self._run_direction(
                    layer_input, layer, 1
                )
                final_hidden.append(h_reverse)
                layer_input = torch.cat(
                    (y_forward, y_reverse),
                    dim=2,
                )
            else:
                layer_input = y_forward

        h_n = torch.stack(final_hidden, dim=0)
        return layer_input, h_n


def replace_gru_modules(module: nn.Module):
    replaced = 0

    for name, child in list(module.named_children()):
        if isinstance(child, nn.GRU):
            setattr(module, name, ExplicitGRU(child))
            replaced += 1
        else:
            replaced += replace_gru_modules(child)

    return replaced


def find_rmvpe_python(rvc_dir: str) -> str:
    """
    Find the RMVPE implementation inside the actual RVC checkout.

    We intentionally do NOT import `infer.lib.rmvpe` here.
    Some Python environments can resolve another top-level package named
    `infer`, which makes `infer.lib` fail even when the RVC source file exists.
    """

    preferred = [
        os.path.join(rvc_dir, "infer", "lib", "rmvpe.py"),
        os.path.join(rvc_dir, "infer", "rmvpe.py"),
        os.path.join(rvc_dir, "rvc", "lib", "rmvpe.py"),
        os.path.join(rvc_dir, "lib", "rmvpe.py"),
    ]

    for path in preferred:
        if os.path.isfile(path):
            return os.path.abspath(path)

    candidates = []
    for path in glob.glob(
        os.path.join(rvc_dir, "**", "rmvpe.py"),
        recursive=True,
    ):
        normalized = path.replace("\\", "/").lower()

        # Never accidentally load a file from the virtual environment.
        if "/.venv/" in normalized or "/site-packages/" in normalized:
            continue

        candidates.append(os.path.abspath(path))

    if not candidates:
        raise FileNotFoundError(
            "RVC checkout内に rmvpe.py が見つかりませんでした。\n"
            f"検索開始場所: {rvc_dir}"
        )

    # Prefer the shallowest source path if the repository layout differs.
    candidates.sort(key=lambda p: (p.count(os.sep), len(p)))

    print("[INFO] RMVPE Python candidates:")
    for path in candidates:
        print("       " + path)

    return candidates[0]


def load_module_from_file(module_path: str, rvc_dir: str):
    """
    Load rmvpe.py by absolute file path while explicitly putting the
    actual RVC repository root on sys.path.

    Do NOT derive the repository root from the rmvpe.py depth:
    current RVC layouts may place rmvpe.py at either
      infer/rmvpe.py
    or
      infer/lib/rmvpe.py
    and those have different depths.
    """

    module_name = "_rvc_local_rmvpe_for_sentis_export"

    module_path = os.path.abspath(module_path)
    module_dir = os.path.dirname(module_path)
    rvc_dir = os.path.abspath(rvc_dir)
    infer_dir = os.path.join(rvc_dir, "infer")

    # rmvpe.py imports repository-local modules such as:
    #     from tools.cuda_graph import run_cuda_graph
    #
    # Therefore the repository ROOT itself must be importable.
    search_paths = [
        rvc_dir,
        infer_dir,
        module_dir,
    ]

    for path in reversed(search_paths):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    print("[DEBUG] Python import roots:")
    for path in search_paths:
        print("       " + path)

    tools_cuda_graph = os.path.join(
        rvc_dir,
        "tools",
        "cuda_graph.py",
    )

    if os.path.isfile(tools_cuda_graph):
        print("[DEBUG] Found tools.cuda_graph:")
        print("       " + tools_cuda_graph)
    else:
        print("[WARN] tools/cuda_graph.py was not found at:")
        print("       " + tools_cuda_graph)

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"rmvpe.py のロード仕様を作成できませんでした: {module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_rmvpe_model(rvc_dir: str, checkpoint_path: str):
    rmvpe_py = find_rmvpe_python(rvc_dir)

    print("[INFO] Loading RMVPE implementation directly:")
    print(rmvpe_py)

    module = load_module_from_file(rmvpe_py, rvc_dir)

    if not hasattr(module, "E2E"):
        classes = [
            name
            for name, value in vars(module).items()
            if isinstance(value, type)
        ]

        raise AttributeError(
            "見つけた rmvpe.py に E2E クラスがありません。\n"
            f"File: {rmvpe_py}\n"
            f"Classes: {classes}"
        )

    E2E = module.E2E

    model = E2E(4, 1, (2, 2))

    print("[INFO] Loading RMVPE checkpoint...")

    try:
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility fallback for older torch versions.
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    # Some checkpoints may wrap the state dict.
    if isinstance(ckpt, dict) and "model" in ckpt:
        maybe_model = ckpt["model"]
        if isinstance(maybe_model, dict):
            ckpt = maybe_model

    model.load_state_dict(ckpt, strict=True)
    model.float().eval()
    return model


def inspect_onnx(path: str):
    import onnx

    model = onnx.load(path, load_external_data=False)
    ops = sorted({node.op_type for node in model.graph.node})

    forbidden = {
        "GRU",
        "RNN",
        "LSTM",
        "Loop",
        "Scan",
        "If",
    }

    found_forbidden = sorted(set(ops) & forbidden)

    print("[INFO] ONNX operators:")
    print("       " + ", ".join(ops))

    if found_forbidden:
        raise RuntimeError(
            "Sentis-incompatible recurrent/control-flow operators remain: "
            + ", ".join(found_forbidden)
        )

    onnx.checker.check_model(model)
    print("[OK] ONNX checker passed.")
    print("[OK] No GRU/RNN/LSTM/Loop/Scan/If operators were found.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rvc-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=224)
    args = parser.parse_args()

    rvc_dir = os.path.abspath(args.rvc_dir)
    checkpoint = os.path.abspath(args.checkpoint)
    output = os.path.abspath(args.output)

    if not os.path.isdir(rvc_dir):
        raise FileNotFoundError(f"RVC directory not found: {rvc_dir}")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"RMVPE checkpoint not found: {checkpoint}")

    os.makedirs(os.path.dirname(output), exist_ok=True)

    print("[INFO] RVC directory:")
    print(rvc_dir)
    print("[INFO] RMVPE checkpoint:")
    print(checkpoint)
    print("[INFO] Output:")
    print(output)
    print()

    print("[INFO] Loading reference RMVPE...")
    reference = load_rmvpe_model(rvc_dir, checkpoint)

    print("[INFO] Creating Sentis-compatible copy...")
    sentis_model = copy.deepcopy(reference)

    count = replace_gru_modules(sentis_model)
    if count == 0:
        raise RuntimeError("No torch.nn.GRU module was found.")

    print(f"[INFO] Replaced GRU modules: {count}")

    frames = args.frames

    # RMVPE DeepUnet has five 2x down/up-sampling stages.
    # The upstream RMVPE implementation pads the mel time axis to a
    # multiple of 32 before E2E.forward(). Feeding 200 directly causes
    # decoder skip-connection shapes such as 24 vs 25.
    if frames % 32 != 0:
        raise ValueError(
            f"RMVPE network input frames must be a multiple of 32, got {frames}. "
            "Use 224 for a 2-second / 16 kHz chunk."
        )

    dummy = torch.randn(
        1,
        128,
        frames,
        dtype=torch.float32,
    ) * 0.1

    print("[INFO] Checking numerical parity...")
    with torch.no_grad():
        expected = reference(dummy)
        actual = sentis_model(dummy)

    diff = (expected - actual).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    print(f"[INFO] Reference output: {tuple(expected.shape)}")
    print(f"[INFO] Sentis output   : {tuple(actual.shape)}")
    print(f"[INFO] parity max_abs  : {max_abs:.8g}")
    print(f"[INFO] parity mean_abs : {mean_abs:.8g}")

    if max_abs > 1e-4:
        raise RuntimeError(
            f"Explicit GRU parity failed: max_abs={max_abs}"
        )

    print("[OK] Explicit GRU matches torch.nn.GRU.")

    print()
    print("[INFO] Exporting fixed-length ONNX...")

    with torch.no_grad():
        torch.onnx.export(
            sentis_model,
            dummy,
            output,
            input_names=["mel"],
            output_names=["hidden"],
            opset_version=15,
            do_constant_folding=True,
            dynamo=False,
        )

    if not os.path.isfile(output):
        raise RuntimeError("ONNX file was not created.")

    print("[OK] ONNX created.")
    print(f"[INFO] File size: {os.path.getsize(output):,} bytes")

    inspect_onnx(output)

    print()
    print("[SUCCESS] Sentis-compatible RMVPE model is ready.")
    print(f"[INFO] Input : mel    [1, 128, {frames}]")
    print(f"[INFO] Output: hidden [1, {frames}, 360]")
    print("[INFO] For a 2-second 16 kHz chunk: mel is ~201 frames, pad to 224,")
    print("[INFO] run this model, then trim the hidden output back to the original mel length.")


if __name__ == "__main__":
    main()