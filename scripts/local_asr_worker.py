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

    buffer = bytearray()
    chunk_bytes = int(16000 * 2 * args.chunk_ms / 1000)
    for line in sys.stdin:
            try:
                msg = json.loads(line)
                if msg.get("type") == "stop":
                    break
                if msg.get("type") != "audio":
                    continue
                buffer.extend(base64.b64decode(msg["pcm16"]))
                while len(buffer) >= chunk_bytes:
                    chunk = bytes(buffer[:chunk_bytes])
                    del buffer[:chunk_bytes]
                    audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    if not is_speech_like(audio):
                        continue
                    if backend == "mlx":
                        result = mlx_whisper.transcribe(
                            audio,
                            path_or_hf_repo=args.model,
                            fp16=True,
                            language="zh",
                            condition_on_previous_text=False,
                            hallucination_silence_threshold=1,
                            initial_prompt="Java、HashMap、TreeMap、ConcurrentHashMap、并发、线程、锁、数据库、接口、API。",
                        )
                        text = str(result.get("text", "")).strip()
                    else:
                        segments, _ = model.transcribe(
                            audio,
                            language="zh",
                            beam_size=5,
                            vad_filter=True,
                            condition_on_previous_text=False,
                            initial_prompt="Java、HashMap、TreeMap、ConcurrentHashMap、并发、线程、锁、数据库、接口、API。",
                            hotwords="Java, HashMap, TreeMap, ConcurrentHashMap, synchronized, volatile, 并发, 线程, 锁",
                        )
                        text = "".join(seg.text.strip() for seg in segments).strip()
                    text = clean_transcript(text, convert_to_simplified)
                    if text and not is_hallucination_like(text):
                        emit({"type": "result", "text": text, "final": True})
            except Exception as exc:
                emit({"type": "error", "message": f"本地 Whisper 转写失败：{exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
