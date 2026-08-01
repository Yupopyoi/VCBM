from __future__ import annotations

import time
import uuid

import numpy as np
import soundfile as sf

from voice_common import (
    OUTPUT_DIR,
    generation_session,
    get_model,
    set_seed,
)


def build_voice_design_text(
    voice_prompt: str,
    speech_text: str,
) -> str:
    """
    Voice Design用の入力文字列を作成する。
    """

    prompt = voice_prompt.strip()
    text = speech_text.strip()

    if prompt.startswith("(") and prompt.endswith(")"):
        prompt = prompt[1:-1].strip()

    return f"({prompt}){text}"


def generate_voice(
    *,
    voice_prompt: str,
    speech_text: str,
    cfg_value: float,
    inference_timesteps: int,
    seed: int,
    candidate_count: int,
    normalize: bool,
) -> dict:
    """
    プロンプトから音声を生成する。
    """

    with generation_session():
        model = get_model()

        sample_rate = int(
            model.tts_model.sample_rate
        )

        request_id = (
            "voicegen_"
            + uuid.uuid4().hex[:12]
        )

        design_text = build_voice_design_text(
            voice_prompt=voice_prompt,
            speech_text=speech_text,
        )

        candidates: list[dict] = []

        for index in range(candidate_count):
            candidate_number = index + 1
            candidate_seed = seed + index

            set_seed(candidate_seed)

            print(
                f"[{request_id}] generating "
                f"{candidate_number}/{candidate_count}, "
                f"seed={candidate_seed}, "
                f"cfg={cfg_value}, "
                f"steps={inference_timesteps}"
            )

            started_at = time.perf_counter()

            waveform = model.generate(
                text=design_text,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                retry_badcase=True,
                retry_badcase_max_times=3,
                retry_badcase_ratio_threshold=6.0,
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

            output_path = OUTPUT_DIR / file_name

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
                    "audio_url": (
                        f"/audio/{file_name}"
                    ),
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
            "candidates": candidates,
            "error": "",
        }
        