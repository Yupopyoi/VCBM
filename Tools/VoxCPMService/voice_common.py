from __future__ import annotations

import os
import random
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import torch
from voxcpm import VoxCPM


ROOT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLONE_DIR = ROOT_DIR / "clone"
CLONE_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_DIR = ROOT_DIR / "references"
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.getenv(
    "VOXCPM_MODEL_ID",
    "openbmb/VoxCPM2",
)

DEVICE = os.getenv(
    "VOXCPM_DEVICE",
    "auto",
)

ModelState = Literal[
    "not_loaded",
    "loading",
    "ready",
    "error",
]

_model: VoxCPM | None = None
_model_state: ModelState = "not_loaded"
_model_error = ""
_model_devices: list[str] = []

_model_lock = threading.Lock()
_generation_lock = threading.Lock()


class GenerationBusyError(RuntimeError):
    """別の音声生成処理が実行中の場合に発生する。"""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_module_devices(module: object) -> list[str]:
    if not isinstance(module, torch.nn.Module):
        return []

    devices: set[str] = set()

    for parameter in module.parameters():
        devices.add(str(parameter.device))

    for buffer in module.buffers():
        devices.add(str(buffer.device))

    return sorted(devices)


def get_model() -> VoxCPM:
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


def get_audio_path(file_path: str) -> Path:
    """
    outputフォルダ内の音声を安全に取得する。

    例:
        voiceclone_xxx.wav
        20260801_215500/part_001_seed_1234.wav
    """

    output_root = OUTPUT_DIR.resolve()
    audio_path = (OUTPUT_DIR / file_path).resolve()

    # outputフォルダの外を指定できないようにする
    if output_root not in audio_path.parents:
        raise ValueError(
            "Invalid audio file path."
        )

    if not audio_path.is_file():
        raise FileNotFoundError(
            "Audio file not found."
        )

    return audio_path


@contextmanager
def generation_session() -> Iterator[None]:
    if not _generation_lock.acquire(blocking=False):
        raise GenerationBusyError(
            "Another generation request is already running."
        )

    try:
        yield
    finally:
        _generation_lock.release()


def get_service_status() -> dict:
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
        "configured_device": DEVICE,
        "model_devices": _model_devices,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
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


def _get_safe_audio_path(
    directory: Path,
    file_name: str,
    not_found_message: str,
) -> Path:
    safe_name = Path(file_name).name

    if safe_name != file_name:
        raise ValueError("Invalid file name.")

    audio_path = directory / safe_name

    if not audio_path.is_file():
        raise FileNotFoundError(not_found_message)

    return audio_path


def get_audio_path(file_path: str) -> Path:
    """
    音声ファイルを安全に取得する。

    検索順序:
        1. outputフォルダ
        2. cloneフォルダ

    使用例:
        voicegen_xxx.wav
        20260801_224103/part_005_seed_46.wav
    """

    if not file_path:
        raise ValueError(
            "Audio file path is empty."
        )

    relative_path = Path(file_path)

    if relative_path.is_absolute():
        raise ValueError(
            "Absolute paths are not allowed."
        )

    if relative_path.suffix.lower() != ".wav":
        raise ValueError(
            "Only WAV files can be accessed."
        )

    search_directories = [
        OUTPUT_DIR,
        CLONE_DIR,
    ]

    for directory in search_directories:
        directory_root = directory.resolve()
        audio_path = (
            directory / relative_path
        ).resolve()

        try:
            # 指定パスが検索対象フォルダ内にあることを確認
            audio_path.relative_to(
                directory_root
            )
        except ValueError:
            continue

        if audio_path.is_file():
            return audio_path

    raise FileNotFoundError(
        "Audio file was not found in "
        f"output or clone: {file_path}"
    )
    

def get_reference_audio_path(file_name: str) -> Path:
    return _get_safe_audio_path(
        REFERENCE_DIR,
        file_name,
        "Reference audio file not found.",
    )
