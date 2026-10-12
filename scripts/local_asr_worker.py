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
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device="cpu", compute_type="int8")
        emit({"type": "ready"})
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
                    segments, _ = model.transcribe(audio, language="zh", beam_size=1, vad_filter=True, condition_on_previous_text=False)
                    text = "".join(seg.text.strip() for seg in segments).strip()
                    text = clean_transcript(text, convert_to_simplified)
                    if text:
                        emit({"type": "result", "text": text, "final": True})
            except Exception as exc:
                emit({"type": "error", "message": f"本地 Whisper 转写失败：{exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
