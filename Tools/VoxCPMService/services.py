from __future__ import annotations

import os
import random
import threading
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from voxcpm import VoxCPM


# ============================================================
# 設定
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_ID = os.getenv(
    "VOXCPM_MODEL_ID",
    "openbmb/VoxCPM2",
)

DEVICE = os.getenv(
    "VOXCPM_DEVICE",
    "auto",
)


# ============================================================
# モデル状態
# ============================================================

ModelState = Literal[
    "not_loaded",
    "loading",
    "ready",
    "error",
]

CloneMode = Literal[
    "controllable",
    "ultimate",
]

_model: VoxCPM | None = None

_model_state: ModelState = "not_loaded"
_model_error = ""
_model_devices: list[str] = []

_model_lock = threading.Lock()


class GenerationBusyError(RuntimeError):
    """
    別の音声生成処理が実行中の場合に発生する。
    """


# ============================================================
# 共通処理
# ============================================================

def set_seed(seed: int) -> None:
    """
    音声生成の乱数シードを設定する。
    """

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_module_devices(module: object) -> list[str]:
    """
    PyTorchモジュールが配置されているデバイスを取得する。
    """

    if not isinstance(module, torch.nn.Module):
        return []

    devices: set[str] = set()

    for parameter in module.parameters():
        devices.add(str(parameter.device))

    for buffer in module.buffers():
        devices.add(str(buffer.device))

    return sorted(devices)


def build_voice_design_text(
    voice_prompt: str,
    speech_text: str,
) -> str:
    """
    VoxCPM2 Voice Design用の入力文章を作成する。

    例:
        (A youthful Japanese girl...)こんにちは。
    """

    prompt = voice_prompt.strip()
    text = speech_text.strip()

    if prompt.startswith("(") and prompt.endswith(")"):
        prompt = prompt[1:-1].strip()

    return f"({prompt}){text}"


def build_clone_text(
    style_prompt: str,
    speech_text: str,
) -> str:
    """
    Controllable Cloning用の入力文章を作成する。

    style_promptが空の場合は、
    speech_textをそのまま返す。
    """

    prompt = style_prompt.strip()
    text = speech_text.strip()

    if not prompt:
        return text

    if prompt.startswith("(") and prompt.endswith(")"):
        prompt = prompt[1:-1].strip()

    return f"({prompt}){text}"


# ============================================================
# モデル管理
# ============================================================

def get_model() -> VoxCPM:
    """
    VoxCPM2モデルを取得する。

    初回呼び出し時のみモデルを読み込み、
    2回目以降は読み込み済みモデルを返す。
    """

    global _model
    global _model_state
    global _model_error
    global _model_devices

    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        _model_state = "loading"
        _model_error = ""

        try:
            print(
                f"Loading VoxCPM2: "
                f"{MODEL_ID} / device={DEVICE}"
            )

            model = VoxCPM.from_pretrained(
                MODEL_ID,
                load_denoiser=False,
                device=DEVICE,
            )

            _model = model
            _model_devices = get_module_devices(
                model.tts_model
            )
            _model_state = "ready"

            print("VoxCPM2 model ready.")
            print(f"Configured device: {DEVICE}")
            print(
                f"Actual model devices: "
                f"{_model_devices}"
            )

            return model

        except Exception as exception:
            _model = None
            _model_devices = []
            _model_state = "error"
            _model_error = str(exception)

            raise


def get_service_status() -> dict:
    """
    モデルとCUDAの状態を返す。
    """

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        cuda_device_count = torch.cuda.device_count()
        cuda_device_name = torch.cuda.get_device_name(0)

        allocated_memory = (
            torch.cuda.memory_allocated(0) / 1024**2
        )
        reserved_memory = (
            torch.cuda.memory_reserved(0) / 1024**2
        )
    else:
        cuda_device_count = 0
        cuda_device_name = None
        allocated_memory = 0.0
        reserved_memory = 0.0

    return {
        "model_state": _model_state,
        "model_id": MODEL_ID,

        # 指定されたデバイス
        "configured_device": DEVICE,

        # 実際にモデルが配置されたデバイス
        "model_devices": _model_devices,

        # PyTorchおよびCUDA
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,

        # GPUメモリ
        "cuda_memory_allocated_mb": round(
            allocated_memory,
            2,
        ),
        "cuda_memory_reserved_mb": round(
            reserved_memory,
            2,
        ),

        "error": _model_error,
    }

# ============================================================
# 音声ファイル取得
# ============================================================

def get_audio_path(file_name: str) -> Path:
    """
    指定された生成済み音声のパスを取得する。

    ディレクトリトラバーサルを防ぐため、
    ファイル名部分だけであることを確認する。
    """

    safe_name = Path(file_name).name

    if safe_name != file_name:
        raise ValueError(
            "Invalid file name."
        )

    audio_path = OUTPUT_DIR / safe_name

    if not audio_path.is_file():
        raise FileNotFoundError(
            "Audio file not found."
        )

    return audio_path
