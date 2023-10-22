import numpy as np
from faster_whisper import WhisperModel
from typing import Optional
import time
import zlib
from typing import TypedDict

class WhisperOutput(TypedDict):
    text: str 
    dur: float
    compression_ratio: float
    avg_logprob: float
    no_speech: float
    language: str
    language_prob: float
    snr: float

def _get_compression_ratio(text: str) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def _signaltonoise(a: np.ndarray, axis: int = 0, ddof: int = 0) -> np.ndarray:
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)

def stt_result(
    y: np.ndarray,
    model: WhisperModel,
    brand_lang: Optional[str] = None,
    compression_ratio_threshold: float = 2.4,
    logprob_threshold: float = -1.0,
    no_speech_threshold: float = 0.6,
    task: str = "translate",
    prompt: str = "",
    prefix: str = "",
) -> WhisperOutput:
    t1 = time.time()
    if brand_lang is not None:
        result = model.transcribe(
            y,
            task=task,
            language=brand_lang,
            beam_size=5,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=logprob_threshold,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            prefix=prefix,
        )
    else:    
        result = model.transcribe(
            y,
            task=task,
            beam_size=5,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=logprob_threshold,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            prefix=prefix,
        )
    final_text = []
    try:
        for segment in result[0]:
            text = segment.text

            needs_fallback = False

            if _get_compression_ratio(text) > compression_ratio_threshold:
                needs_fallback = True

            if segment.avg_logprob < logprob_threshold:
                needs_fallback = True

            if segment.no_speech_prob > no_speech_threshold:
                needs_fallback = True

            if needs_fallback:
                text = ""

            final_text.append(text)

        text = " ".join(final_text)
        compression_ratio = _get_compression_ratio(text)

        if compression_ratio > compression_ratio_threshold:
            text = ""

        snr = _signaltonoise(y)
        t2 = time.time()

        output: WhisperOutput = {
            "text": text.strip(),
            "dur": t2 - t1,
            "compression_ratio": compression_ratio,
            "avg_logprob": segment.avg_logprob,
            "no_speech": segment.no_speech_prob,
            "language": result[1].language,
            "language_prob": result[1].language_probability,
            "snr": snr,
        }

        return output
    
    except Exception:
        output: WhisperOutput =  {
            "text": "",
            "dur": 0,
            "compression_ratio": 0,
            "avg_logprob": 0,
            "no_speech": 0,
            "language": "en",
            "language_prob": 0,
            "snr": 0,
        }

        return output
