from __future__ import annotations

import os
import random
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from voxcpm import VoxCPM


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.getenv("VOXCPM_MODEL_ID", "openbmb/VoxCPM2")
DEVICE = os.getenv("VOXCPM_DEVICE", "auto")

app = FastAPI(
    title="VCBM VoxCPM2 Local Service",
    version="0.1.0",
)

_model: VoxCPM | None = None
_model_state: Literal["not_loaded", "loading", "ready", "error"] = "not_loaded"
_model_error = ""
_model_lock = threading.Lock()
_generation_lock = threading.Lock()


class GenerateRequest(BaseModel):
    voice_prompt: str = Field(min_length=1, max_length=4000)
    speech_text: str = Field(min_length=1, max_length=4000)
    cfg_value: float = Field(default=1.5, ge=1.0, le=3.0)
    inference_timesteps: int = Field(default=20, ge=4, le=30)
    seed: int = 1234
    candidate_count: int = Field(default=1, ge=1, le=3)
    normalize: bool = True


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

    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        _model_state = "loading"
        _model_error = ""

        try:
            print(f"Loading VoxCPM2: {MODEL_ID} / device={DEVICE}")
            _model = VoxCPM.from_pretrained(
                MODEL_ID,
                load_denoiser=False,
                device=DEVICE,
            )
            _model_state = "ready"
            print("VoxCPM2 model ready.")
            print(f"Configured device: {DEVICE}")
            print(f"Actual model devices: {get_module_devices(_model.tts_model)}")
            return _model
        except Exception as exception:
            _model_state = "error"
            _model_error = str(exception)
            raise


def build_voice_design_text(
    voice_prompt: str,
    speech_text: str,
) -> str:
    prompt = voice_prompt.strip()
    text = speech_text.strip()

    if prompt.startswith("(") and prompt.endswith(")"):
        prompt = prompt[1:-1].strip()

    return f"({prompt}){text}"


@app.get("/health")
def health() -> dict:
    return {
        "success": True,
        "service": "VCBM VoxCPM2 Local Service",
        "model_state": _model_state,
        "model_id": MODEL_ID,
        "device": DEVICE,
        "error": _model_error,
    }


@app.get("/model/status")
def model_status() -> dict:
    return {
        "success": _model_state != "error",
        "model_state": _model_state,
        "model_id": MODEL_ID,
        "device": DEVICE,
        "error": _model_error,
    }


@app.post("/model/load")
def load_model() -> dict:
    try:
        model = get_model()
        return {
            "success": True,
            "model_state": _model_state,
            "sample_rate": int(model.tts_model.sample_rate),
        }
    except Exception as exception:
        raise HTTPException(
            status_code=500,
            detail=f"Model loading failed: {exception}",
        ) from exception


@app.post("/voice/generate")
def generate_voice(request: GenerateRequest) -> dict:
    if not _generation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Another generation request is already running.",
        )

    try:
        model = get_model()
        sample_rate = int(model.tts_model.sample_rate)
        request_id = "voicegen_" + uuid.uuid4().hex[:12]
        design_text = build_voice_design_text(
            request.voice_prompt,
            request.speech_text,
        )

        candidates: list[dict] = []

        for index in range(request.candidate_count):
            candidate_seed = request.seed + index
            set_seed(candidate_seed)

            print(
                f"[{request_id}] generating "
                f"{index + 1}/{request.candidate_count}, "
                f"seed={candidate_seed}, "
                f"cfg={request.cfg_value}, "
                f"steps={request.inference_timesteps}"
            )

            started_at = time.perf_counter()

            waveform = model.generate(
                text=design_text,
                cfg_value=request.cfg_value,
                inference_timesteps=request.inference_timesteps,
                normalize=request.normalize,
                retry_badcase=True,
                retry_badcase_max_times=3,
                retry_badcase_ratio_threshold=6.0,
            )

            elapsed = time.perf_counter() - started_at
            waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)

            if waveform.size == 0:
                raise RuntimeError("VoxCPM2 returned an empty waveform.")

            file_name = (
                f"{request_id}_candidate_{index + 1:02d}"
                f"_seed_{candidate_seed}.wav"
            )
            output_path = OUTPUT_DIR / file_name

            sf.write(
                output_path,
                waveform,
                sample_rate,
                subtype="PCM_16",
            )

            duration_seconds = waveform.size / float(sample_rate)

            candidates.append(
                {
                    "candidate_id": f"candidate_{index + 1:02d}",
                    "audio_url": f"/audio/{file_name}",
                    "file_name": file_name,
                    "sample_rate": sample_rate,
                    "seed": candidate_seed,
                    "duration_seconds": round(duration_seconds, 4),
                    "generation_seconds": round(elapsed, 4),
                }
            )

        return {
            "success": True,
            "request_id": request_id,
            "candidates": candidates,
            "error": "",
        }
    except HTTPException:
        raise
    except Exception as exception:
        print(f"Generation error: {exception}")
        raise HTTPException(
            status_code=500,
            detail=f"Voice generation failed: {exception}",
        ) from exception
    finally:
        _generation_lock.release()


@app.get("/audio/{file_name}")
def get_audio(file_name: str) -> FileResponse:
    safe_name = Path(file_name).name

    if safe_name != file_name:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    path = OUTPUT_DIR / safe_name

    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found.")

    return FileResponse(
        path,
        media_type="audio/wav",
        filename=safe_name,
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8765,
        log_level="info",
    )
