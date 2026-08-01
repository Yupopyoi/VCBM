from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Literal
import json
from datetime import datetime
from itertools import cycle

import numpy as np
import soundfile as sf

from voice_common import (
    OUTPUT_DIR,
    CLONE_DIR,
    REFERENCE_DIR,
    generation_session,
    get_model,
    get_reference_audio_path,
    set_seed,
)

from five_minute_texts import FIVE_MINUTE_TEXTS


CloneMode = Literal[
    "controllable",
    "ultimate",
]

MAX_REFERENCE_FILE_SIZE = 50 * 1024 * 1024


def save_reference_audio(
    *,
    original_file_name: str,
    file_data: bytes,
) -> dict:
    """Unityから受け取ったWAVをreferencesへ保存する。"""

    if not file_data:
        raise ValueError("The uploaded WAV file is empty.")

    if len(file_data) > MAX_REFERENCE_FILE_SIZE:
        raise ValueError(
            "The reference WAV file exceeds 50 MB."
        )

    if Path(original_file_name).suffix.lower() != ".wav":
        raise ValueError(
            "Reference audio must be a WAV file."
        )

    reference_file_name = (
        "reference_"
        + uuid.uuid4().hex[:16]
        + ".wav"
    )
    reference_path = REFERENCE_DIR / reference_file_name

    try:
        reference_path.write_bytes(file_data)
        audio_info = sf.info(reference_path)

        if audio_info.frames <= 0:
            raise ValueError(
                "The WAV file contains no audio frames."
            )

        duration_seconds = (
            audio_info.frames
            / float(audio_info.samplerate)
        )

        return {
            "success": True,
            "reference_file_name": reference_file_name,
            "original_file_name": original_file_name,
            "sample_rate": int(audio_info.samplerate),
            "channels": int(audio_info.channels),
            "duration_seconds": round(
                duration_seconds,
                4,
            ),
            "error": "",
        }

    except Exception:
        reference_path.unlink(missing_ok=True)
        raise


def build_clone_text(
    style_prompt: str,
    speech_text: str,
) -> str:
    prompt = style_prompt.strip()
    text = speech_text.strip()

    if not prompt:
        return text

    if prompt.startswith("(") and prompt.endswith(")"):
        prompt = prompt[1:-1].strip()

    return f"({prompt}){text}"


def clone_voice(
    *,
    reference_file_name: str,
    speech_text: str,
    mode: CloneMode,
    style_prompt: str,
    reference_text: str,
    five_minute_mode: bool,
    cfg_value: float,
    inference_timesteps: int,
    seed: int,
    candidate_count: int,
    normalize: bool,
) -> dict:
    """アップロード済み参照音声を使ってVoice Cloneする。"""

    with generation_session():
        reference_path = get_reference_audio_path(
            reference_file_name
        )

        if mode not in {
            "controllable",
            "ultimate",
        }:
            raise ValueError("Invalid clone mode.")

        cleaned_reference_text = reference_text.strip()

        if mode == "ultimate" and not cleaned_reference_text:
            raise ValueError(
                "reference_text is required "
                "when mode is 'ultimate'."
            )

        model = get_model()

        if five_minute_mode:
            return clone_five_minutes(
                model=model,
                reference_path=reference_path,
                mode=mode,
                style_prompt=style_prompt,
                reference_text=reference_text,
                cfg_value=cfg_value,
                inference_timesteps=(
                    inference_timesteps
                ),
                seed=seed,
                normalize=normalize,
            )
            
        sample_rate = int(
            model.tts_model.sample_rate
        )

        request_id = (
            "voiceclone_"
            + uuid.uuid4().hex[:12]
        )

        generation_text = (
            build_clone_text(
                style_prompt=style_prompt,
                speech_text=speech_text,
            )
            if mode == "controllable"
            else speech_text.strip()
        )

        candidates: list[dict] = []

        for index in range(candidate_count):
            candidate_number = index + 1
            candidate_seed = seed + index
            set_seed(candidate_seed)

            generate_arguments: dict = {
                "text": generation_text,
                "reference_wav_path": str(
                    reference_path
                ),
                "cfg_value": cfg_value,
                "inference_timesteps": (
                    inference_timesteps
                ),
                "normalize": normalize,
                "retry_badcase": True,
                "retry_badcase_max_times": 3,
                "retry_badcase_ratio_threshold": 6.0,
            }

            if mode == "ultimate":
                generate_arguments[
                    "prompt_wav_path"
                ] = str(reference_path)
                generate_arguments[
                    "prompt_text"
                ] = cleaned_reference_text

            print(
                f"[{request_id}] cloning "
                f"{candidate_number}/{candidate_count}, "
                f"mode={mode}, "
                f"reference={reference_path.name}, "
                f"seed={candidate_seed}"
            )

            started_at = time.perf_counter()
            waveform = model.generate(
                **generate_arguments
            )
            generation_seconds = (
                time.perf_counter() - started_at
            )

            waveform = np.asarray(
                waveform,
                dtype=np.float32,
            ).reshape(-1)

            if waveform.size == 0:
                raise RuntimeError(
                    "VoxCPM2 returned an empty waveform."
                )

            file_name = (
                f"{request_id}"
                f"_candidate_{candidate_number:02d}"
                f"_seed_{candidate_seed}.wav"
            )
            output_path = CLONE_DIR / file_name

            sf.write(
                file=output_path,
                data=waveform,
                samplerate=sample_rate,
                subtype="PCM_16",
            )

            duration_seconds = (
                waveform.size / float(sample_rate)
            )

            candidates.append(
                {
                    "candidate_id": (
                        f"candidate_{candidate_number:02d}"
                    ),
                    "audio_url": f"/audio/{file_name}",
                    "file_name": file_name,
                    "sample_rate": sample_rate,
                    "seed": candidate_seed,
                    "duration_seconds": round(
                        duration_seconds,
                        4,
                    ),
                    "generation_seconds": round(
                        generation_seconds,
                        4,
                    ),
                }
            )

        return {
            "success": True,
            "request_id": request_id,
            "mode": mode,
            "reference_file_name": reference_path.name,
            "candidates": candidates,
            "error": "",
        }
        
        
TARGET_DURATION_SECONDS = 300.0


def generate_clone_waveform(
    *,
    model,
    reference_path: Path,
    speech_text: str,
    mode: CloneMode,
    style_prompt: str,
    reference_text: str,
    cfg_value: float,
    inference_timesteps: int,
    normalize: bool,
) -> np.ndarray:
    """
    1文章分のClone音声を生成する。
    """

    if mode == "controllable":
        generation_text = build_clone_text(
            style_prompt=style_prompt,
            speech_text=speech_text,
        )
    else:
        generation_text = speech_text.strip()

    generate_arguments: dict = {
        "text": generation_text,
        "reference_wav_path": str(
            reference_path
        ),
        "cfg_value": cfg_value,
        "inference_timesteps": (
            inference_timesteps
        ),
        "normalize": normalize,
        "retry_badcase": True,
        "retry_badcase_max_times": 3,
        "retry_badcase_ratio_threshold": 6.0,
    }

    if mode == "ultimate":
        generate_arguments[
            "prompt_wav_path"
        ] = str(reference_path)

        generate_arguments[
            "prompt_text"
        ] = reference_text.strip()

    waveform = model.generate(
        **generate_arguments
    )

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    ).reshape(-1)

    if waveform.size == 0:
        raise RuntimeError(
            "VoxCPM2 returned an empty waveform."
        )

    return waveform    

        
def clone_five_minutes(
    *,
    model,
    reference_path: Path,
    mode: CloneMode,
    style_prompt: str,
    reference_text: str,
    cfg_value: float,
    inference_timesteps: int,
    seed: int,
    normalize: bool,
) -> dict:
    """
    Python側に用意した文章を順番に生成し、
    合計300秒以上になるまで続ける。
    """

    sample_rate = int(
        model.tts_model.sample_rate
    )

    folder_name = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_folder = (
        CLONE_DIR / folder_name
    )

    # 同じ秒に複数回開始した場合
    if output_folder.exists():
        folder_name += (
            "_" + uuid.uuid4().hex[:4]
        )
        output_folder = (
            CLONE_DIR / folder_name
        )

    output_folder.mkdir(
        parents=True,
        exist_ok=False,
    )

    candidates: list[dict] = []
    manifest_items: list[dict] = []

    total_duration_seconds = 0.0
    segment_number = 0

    # 文章が尽きた場合は先頭に戻る
    for speech_text in cycle(
        FIVE_MINUTE_TEXTS
    ):
        if total_duration_seconds >= (
            TARGET_DURATION_SECONDS
        ):
            break

        segment_number += 1
        candidate_seed = (
            seed + segment_number - 1
        )

        set_seed(candidate_seed)

        print(
            f"[{folder_name}] "
            f"segment={segment_number}, "
            f"seed={candidate_seed}, "
            f"current={total_duration_seconds:.2f}s"
        )

        started_at = time.perf_counter()

        waveform = generate_clone_waveform(
            model=model,
            reference_path=reference_path,
            speech_text=speech_text,
            mode=mode,
            style_prompt=style_prompt,
            reference_text=reference_text,
            cfg_value=cfg_value,
            inference_timesteps=(
                inference_timesteps
            ),
            normalize=normalize,
        )

        generation_seconds = (
            time.perf_counter() - started_at
        )

        duration_seconds = (
            waveform.size / float(sample_rate)
        )

        file_name = (
            f"part_{segment_number:03d}"
            f"_seed_{candidate_seed}.wav"
        )

        output_path = (
            output_folder / file_name
        )

        sf.write(
            file=output_path,
            data=waveform,
            samplerate=sample_rate,
            subtype="PCM_16",
        )

        total_duration_seconds += (
            duration_seconds
        )

        relative_path = (
            Path(folder_name) / file_name
        ).as_posix()

        candidate = {
            "candidate_id": (
                f"segment_{segment_number:03d}"
            ),
            "audio_url": (
                f"/audio/{relative_path}"
            ),
            "file_name": relative_path,
            "sample_rate": sample_rate,
            "seed": candidate_seed,
            "duration_seconds": round(
                duration_seconds,
                4,
            ),
            "generation_seconds": round(
                generation_seconds,
                4,
            ),
        }

        candidates.append(candidate)

        manifest_items.append(
            {
                **candidate,
                "speech_text": speech_text,
            }
        )

    manifest = {
        "folder_name": folder_name,
        "reference_file_name": (
            reference_path.name
        ),
        "target_duration_seconds": (
            TARGET_DURATION_SECONDS
        ),
        "total_duration_seconds": round(
            total_duration_seconds,
            4,
        ),
        "segment_count": len(candidates),
        "segments": manifest_items,
    }

    manifest_path = (
        output_folder / "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "success": True,
        "request_id": folder_name,
        "mode": mode,
        "five_minute_mode": True,
        "folder_name": folder_name,
        "reference_file_name": (
            reference_path.name
        ),
        "target_duration_seconds": (
            TARGET_DURATION_SECONDS
        ),
        "total_duration_seconds": round(
            total_duration_seconds,
            4,
        ),
        "candidates": candidates,
        "error": "",
    }
