#!/usr/bin/env python3
"""Small stdin/stdout bridge for local faster-whisper transcription.

Input: newline-delimited JSON objects with base64 PCM16 mono 16 kHz audio.
Output: newline-delimited JSON result/error objects.
"""
import argparse, base64, json, re, sys
from pathlib import Path
import numpy as np


def clean_transcript(text, convert_to_simplified):
    """Normalize Whisper text before it reaches conversation matching/AI."""
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text).strip()
    if convert_to_simplified:
        try:
            text = convert_to_simplified.convert(text)
        except Exception:
            pass
    # Whisper often adds a leading filler when it starts recording mid-sentence.
    # Remove it only when there is meaningful content after it.
    text = re.sub(r"^(?:嗯+|呃+|额+|啊+|那个)[，,、：:。\s]+(?=.{3,})", "", text)
    return text.strip()


def is_speech_like(audio):
    """Cheap guard for silence/noise before Whisper sees a short live chunk."""
    if not audio.size:
        return False
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    # Whisper is prone to inventing repeated text for quiet 3.5s windows.
    # Keep a little headroom for a quiet microphone, while rejecting the usual
    # digital silence and room-noise floor.
    return rms >= 0.004 or peak >= 0.025


def is_hallucination_like(text):
    compact = re.sub(r"\s+", "", text)
    if len(compact) > 120:
        unique_ratio = len(set(compact)) / max(1, len(compact))
        if unique_ratio < 0.45:
            return True
    # Repeated characters/short tokens are a common Whisper silence failure.
    if re.search(r"(.{1,12})\1{3,}", compact):
        return True
    if re.search(r"(.)\1{5,}", compact):
        return True
    return False


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--chunk-ms", type=int, default=3500)
    args = ap.parse_args()
    try:
        # Keep the conversion data in the app so the packaged worker does not
        # depend on whatever Python packages happen to be installed globally.
        sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
        from opencc import OpenCC
        convert_to_simplified = OpenCC("t2s")
    except Exception as exc:
        emit({"type": "warning", "message": f"繁简转换不可用，将使用原始识别文本：{exc}"})
        convert_to_simplified = None
    try:
        # On Apple Silicon, prefer the same MLX Whisper backend as Jarvis.
        # It uses the local large-v3-turbo model and is much better at mixed
        # Chinese/English technical terms such as HashMap and TreeMap.
        try:
            import mlx_whisper
        except Exception:
            mlx_whisper = None
        if mlx_whisper is not None:
            model = None
            backend = "mlx"
        else:
            from faster_whisper import WhisperModel
            model = WhisperModel(args.model, device="cpu", compute_type="int8")
            backend = "faster-whisper"
        emit({"type": "ready", "backend": backend})
    except Exception as exc:
        emit({"type": "error", "message": f"本地 Whisper 加载失败：{exc}"})
        return 2

    # Segment on local speech activity instead of blindly transcribing every
    # fixed window.  This is the important part of Jarvis's pipeline: Whisper
    # sees a complete phrase with its context, not half a sentence plus
    # silence.  20 ms frames are enough for a lightweight VAD on the worker.
    frame_samples = 320  # 20 ms at 16 kHz
    pre_roll: list[np.ndarray] = []
    speech: list[np.ndarray] = []
    in_speech = False
    silence_ms = 0
    last_partial_ms = 0
    max_speech_ms = 12000
    end_silence_ms = 650

    def transcribe_segment(audio, final=True):
        if len(audio) < int(16000 * 0.25) or not is_speech_like(audio):
            return
        if backend == "mlx":
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=args.model,
                fp16=True,
                language="zh",
                condition_on_previous_text=False,
                no_speech_threshold=0.7,
                compression_ratio_threshold=2.0,
                hallucination_silence_threshold=1,
                initial_prompt="Java、HashMap、TreeMap、ConcurrentHashMap、并发、线程、线程安全、线程池、锁、数据库、接口、API。",
            )
            text = str(result.get("text", "")).strip()
        else:
            segments, _ = model.transcribe(
                audio,
                language="zh",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
                initial_prompt="Java、HashMap、TreeMap、ConcurrentHashMap、并发、线程、线程安全、线程池、锁、数据库、接口、API。",
                hotwords="Java, HashMap, TreeMap, ConcurrentHashMap, synchronized, volatile, 并发, 线程, 线程安全, 线程池, 锁",
            )
            text = "".join(seg.text.strip() for seg in segments).strip()
        text = clean_transcript(text, convert_to_simplified)
        if text and not is_hallucination_like(text):
            emit({"type": "result", "text": text, "final": final})

    for line in sys.stdin:
        try:
            msg = json.loads(line)
            if msg.get("type") == "stop":
                break
            if msg.get("type") != "audio":
                continue
            samples = np.frombuffer(base64.b64decode(msg["pcm16"]), dtype=np.int16).astype(np.float32) / 32768.0
            for start in range(0, len(samples) - frame_samples + 1, frame_samples):
                frame = samples[start : start + frame_samples]
                active = is_speech_like(frame)
                if active:
                    if not in_speech:
                        speech = list(pre_roll)
                        in_speech = True
                    speech.append(frame)
                    silence_ms = 0
                    speech_ms = len(speech) * 1000 / 16000
                    if speech_ms >= 800 and speech_ms - last_partial_ms >= 1000:
                        transcribe_segment(np.concatenate(speech), final=False)
                        last_partial_ms = speech_ms
                elif in_speech:
                    speech.append(frame)
                    silence_ms += 20
                    if silence_ms >= end_silence_ms or len(speech) * 1000 / 16000 >= max_speech_ms:
                        transcribe_segment(np.concatenate(speech))
                        speech = []
                        in_speech = False
                        silence_ms = 0
                        last_partial_ms = 0
                pre_roll.append(frame)
                if len(pre_roll) > 10:
                    pre_roll.pop(0)
        except Exception as exc:
            emit({"type": "error", "message": f"本地 Whisper 转写失败：{exc}"})

    if speech:
        try:
            transcribe_segment(np.concatenate(speech))
        except Exception as exc:
            emit({"type": "error", "message": f"本地 Whisper 转写失败：{exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
