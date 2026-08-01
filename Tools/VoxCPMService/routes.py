from __future__ import annotations

from typing import Literal

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from clone_voice import (
    clone_voice,
    save_reference_audio,
)
from generate_voice import generate_voice
from voice_common import (
    GenerationBusyError,
    get_audio_path,
    get_model,
    get_service_status,
)


app = FastAPI(
    title="VCBM VoxCPM2 Local Service",
    version="0.4.0",
)


class GenerateRequest(BaseModel):
    voice_prompt: str = Field(
        min_length=1,
        max_length=4000,
    )
    speech_text: str = Field(
        min_length=1,
        max_length=4000,
    )
    cfg_value: float = Field(
        default=1.5,
        ge=1.0,
        le=3.0,
    )
    inference_timesteps: int = Field(
        default=20,
        ge=4,
        le=30,
    )
    seed: int = 42
    candidate_count: int = Field(
        default=1,
        ge=1,
        le=10,
    )
    normalize: bool = True


class CloneRequest(BaseModel):
    reference_file_name: str = Field(
        min_length=1,
        max_length=255,
    )

    speech_text: str = Field(
        default="",
        max_length=4000,
    )

    mode: Literal[
        "controllable",
        "ultimate",
    ] = "controllable"

    style_prompt: str = Field(
        default="",
        max_length=4000,
    )

    reference_text: str = Field(
        default="",
        max_length=4000,
    )

    # 追加
    five_minute_mode: bool = False

    cfg_value: float = Field(
        default=1.5,
        ge=1.0,
        le=3.0,
    )

    inference_timesteps: int = Field(
        default=20,
        ge=4,
        le=30,
    )

    seed: int = 42

    candidate_count: int = Field(
        default=1,
        ge=1,
        le=10,
    )

    normalize: bool = True


@app.get("/health")
def health() -> dict:
    return {
        "success": True,
        "service": "VCBM VoxCPM2 Local Service",
        **get_service_status(),
    }


@app.get("/model/status")
def model_status() -> dict:
    status = get_service_status()

    return {
        "success": status["model_state"] != "error",
        **status,
    }


@app.post("/model/load")
def load_model() -> dict:
    try:
        model = get_model()

        return {
            "success": True,
            "model_state": (
                get_service_status()["model_state"]
            ),
            "sample_rate": int(
                model.tts_model.sample_rate
            ),
            "error": "",
        }

    except Exception as exception:
        raise HTTPException(
            status_code=500,
            detail=f"Model loading failed: {exception}",
        ) from exception


@app.post("/voice/generate")
def voice_generate(request: GenerateRequest) -> dict:
    try:
        return generate_voice(
            voice_prompt=request.voice_prompt,
            speech_text=request.speech_text,
            cfg_value=request.cfg_value,
            inference_timesteps=(
                request.inference_timesteps
            ),
            seed=request.seed,
            candidate_count=request.candidate_count,
            normalize=request.normalize,
        )

    except GenerationBusyError as exception:
        raise HTTPException(
            status_code=409,
            detail=str(exception),
        ) from exception

    except Exception as exception:
        print(f"Generation error: {exception}")
        raise HTTPException(
            status_code=500,
            detail=f"Voice generation failed: {exception}",
        ) from exception


@app.post("/voice/reference/upload")
async def upload_voice_reference(
    file: UploadFile = File(...),
) -> dict:
    try:
        original_file_name = (
            file.filename or "reference.wav"
        )
        file_data = await file.read()

        return save_reference_audio(
            original_file_name=original_file_name,
            file_data=file_data,
        )

    except ValueError as exception:
        raise HTTPException(
            status_code=400,
            detail=str(exception),
        ) from exception

    except Exception as exception:
        print(f"Reference upload error: {exception}")
        raise HTTPException(
            status_code=500,
            detail=(
                "Reference audio upload failed: "
                f"{exception}"
            ),
        ) from exception

    finally:
        await file.close()


@app.post("/voice/clone")
def voice_clone(request: CloneRequest) -> dict:
    try:
        return clone_voice(
            reference_file_name=request.reference_file_name,
            speech_text=request.speech_text,
            mode=request.mode,
            style_prompt=request.style_prompt,
            reference_text=request.reference_text,
            five_minute_mode=request.five_minute_mode,
            cfg_value=request.cfg_value,
            inference_timesteps=request.inference_timesteps,
            seed=request.seed,
            candidate_count=request.candidate_count,
            normalize=request.normalize,
        )

    except GenerationBusyError as exception:
        raise HTTPException(
            status_code=409,
            detail=str(exception),
        ) from exception

    except ValueError as exception:
        raise HTTPException(
            status_code=400,
            detail=str(exception),
        ) from exception

    except FileNotFoundError as exception:
        raise HTTPException(
            status_code=404,
            detail=str(exception),
        ) from exception

    except Exception as exception:
        print(f"Clone error: {exception}")
        raise HTTPException(
            status_code=500,
            detail=f"Voice cloning failed: {exception}",
        ) from exception


@app.get("/audio/{file_path:path}")
def get_audio(
    file_path: str,
) -> FileResponse:
    try:
        audio_path = get_audio_path(
            file_path
        )

    except ValueError as exception:
        raise HTTPException(
            status_code=400,
            detail=str(exception),
        ) from exception

    except FileNotFoundError as exception:
        raise HTTPException(
            status_code=404,
            detail=str(exception),
        ) from exception

    return FileResponse(
        path=audio_path,
        media_type="audio/wav",
        filename=audio_path.name,
    )
    
    
@app.post("/voice/reference/upload")
async def upload_voice_reference(
    file: UploadFile = File(...),
) -> dict:
    """
    UnityからVoice Clone用の参照WAVを受信する。
    """

    try:
        original_file_name = (
            file.filename or "reference.wav"
        )

        file_data = await file.read()

        return save_reference_audio(
            original_file_name=original_file_name,
            file_data=file_data,
        )

    except ValueError as exception:
        raise HTTPException(
            status_code=400,
            detail=str(exception),
        ) from exception

    except Exception as exception:
        print(f"Reference upload error: {exception}")

        raise HTTPException(
            status_code=500,
            detail=(
                "Reference audio upload failed: "
                f"{exception}"
            ),
        ) from exception

    finally:
        await file.close()
