"""Gemini TTS helpers tuned for steadier long-form narration."""

import os
import re
import time

import numpy as np
import soundfile as sf

from . import config

_genai_client = None


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        from google import genai
        if not config.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY not set in .env")
        _genai_client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _genai_client


def _trim_silence(audio, sr, thresh=0.015, head_ms=30, tail_ms=120):
    if audio.size == 0:
        return audio
    above = np.where(np.abs(audio) > thresh)[0]
    if above.size == 0:
        return audio
    start = max(0, above[0] - int(head_ms / 1000 * sr))
    end = min(len(audio), above[-1] + int(tail_ms / 1000 * sr))
    return audio[start:end]


def _estimate_chunk_word_limit():
    max_seconds = (
        config.TTS_MAX_OUTPUT_TOKENS / config.TTS_OUTPUT_TOKENS_PER_SECOND
    ) * config.TTS_REQUEST_MARGIN
    estimated = max(1, int(max_seconds * config.WORDS_PER_SECOND))
    return min(estimated, config.TTS_MAX_CHUNK_WORDS)


def _safe_peak_normalize(audio, peak_limit):
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio
    if peak <= peak_limit:
        return audio
    return audio * (peak_limit / peak)


def _speech_rms(audio, gate=0.02):
    if audio.size == 0:
        return 0.0
    active = audio[np.abs(audio) >= gate]
    if active.size == 0:
        active = audio
    return float(np.sqrt(np.mean(np.square(active), dtype=np.float64)))


def _match_chunk_loudness(audio, target_rms, peak_limit):
    if audio.size == 0:
        return audio
    rms = _speech_rms(audio)
    if rms <= 1e-6:
        return _safe_peak_normalize(audio, peak_limit)
    gain = target_rms / rms
    matched = audio * gain
    return _safe_peak_normalize(matched, peak_limit)


def _crossfade_join(parts, sr, crossfade_ms):
    if not parts:
        return np.array([], dtype=np.float32)
    if len(parts) == 1:
        return parts[0]

    crossfade_samples = max(0, int(sr * crossfade_ms / 1000))
    output = parts[0]
    for nxt in parts[1:]:
        overlap = min(crossfade_samples, len(output), len(nxt))
        if overlap <= 0:
            output = np.concatenate([output, nxt])
            continue

        fade_out = np.linspace(1.0, 0.0, overlap, endpoint=False, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
        mixed = output[-overlap:] * fade_out + nxt[:overlap] * fade_in
        output = np.concatenate([output[:-overlap], mixed, nxt[overlap:]])
    return output


def _silence(sr, duration_ms):
    samples = max(0, int(sr * duration_ms / 1000))
    return np.zeros(samples, dtype=np.float32)


def _normalize_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    paragraphs = [line for line in lines if line]
    return "\n\n".join(paragraphs)


def _split_sentences(paragraph):
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [part.strip() for part in parts if part.strip()]


def _chunk_pause_ms(chunk_text, paragraph_break=False):
    stripped = chunk_text.rstrip()
    if paragraph_break:
        return config.TTS_LONG_PAUSE_MS
    if stripped.endswith(("?", "!")):
        return config.TTS_MEDIUM_PAUSE_MS
    if stripped.endswith((".", ":", ";")):
        return config.TTS_MEDIUM_PAUSE_MS
    if stripped.endswith(","):
        return config.TTS_SHORT_PAUSE_MS
    return config.TTS_SHORT_PAUSE_MS


def _build_chunk_prompt(chunk_text, index, total, prev_chunk=None, next_chunk=None):
    context_lines = [
        config.GEMINI_TTS_STYLE or "",
        f"This is part {index} of {total} from one continuous narration recording.",
        "This is not a separate read — it is a seamless continuation of the same recording session.",
        "Keep the exact same narrator: same voice, same accent, same microphone distance, same energy level.",
        "Maintain the exact same speaking rate throughout — do not speed up or slow down.",
        "Do not add extra breath pauses, preambles, or trailing silence.",
        "Read only the EXCERPT text below aloud, nothing else.",
    ]
    if prev_chunk:
        context_lines.append(f"Preceding text (for continuity, do not re-read): {prev_chunk[-220:]}")
    if next_chunk:
        context_lines.append(f"Following text (for continuity, do not pre-read): {next_chunk[:220]}")
    context_lines.append("EXCERPT:")
    context_lines.append(chunk_text)
    return "\n".join(context_lines)


def _build_minimal_prompt(chunk_text):
    return "\n".join(
        [
            config.GEMINI_TTS_STYLE or "",
            "Keep the same narrator identity and steady pacing.",
            "Read only the text below aloud.",
            chunk_text,
        ]
    )


def _build_full_script_prompt(text):
    return "\n".join(
        [
            config.GEMINI_TTS_STYLE or "",
            "This is one continuous full-length narration.",
            "Keep the exact same narrator identity, tone, pacing, and recording character throughout.",
            "Read the entire script continuously in one pass.",
            text,
        ]
    )


def _determine_chunk_count(word_count: int) -> int:
    if word_count <= 400:
        return 1
    if word_count <= 800:
        return 2
    if word_count <= 1200:
        return 3
    return 4


def _split_into_n_chunks(text: str, n: int) -> list[str]:
    """Split text into exactly n chunks at sentence boundaries, as evenly as possible by word count."""
    normalized = _normalize_text(text)
    if n == 1:
        return [normalized]

    sentences: list[str] = []
    for para in normalized.split("\n\n"):
        if para.strip():
            sentences.extend(_split_sentences(para))
    sentences = [s for s in sentences if s.strip()]

    if not sentences:
        return [normalized]
    if len(sentences) <= n:
        return sentences

    total_words = sum(len(s.split()) for s in sentences)
    target = total_words / n

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for i, sentence in enumerate(sentences):
        current.append(sentence)
        current_words += len(sentence.split())
        sentences_left = len(sentences) - i - 1
        chunks_left = n - len(chunks) - 1
        if current_words >= target and sentences_left >= chunks_left and chunks_left > 0:
            chunks.append(" ".join(current).strip())
            current = []
            current_words = 0

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if c.strip()]


def _is_retryable_tts_error(exc):
    try:
        from google.genai import errors

        return isinstance(exc, errors.ServerError)
    except Exception:  # noqa: BLE001
        return False


def _gemini_pcm(prompt_text):
    from google.genai import types

    client = _get_genai_client()
    response = client.models.generate_content(
        model=config.GEMINI_TTS_MODEL,
        contents=prompt_text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=config.GEMINI_TTS_VOICE
                    )
                )
            ),
        ),
    )
    pcm = response.candidates[0].content.parts[0].inline_data.data
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _request_chunk_audio(prompt_text):
    delay = config.TTS_RETRY_BASE_DELAY_SECONDS
    last_exc = None
    for attempt in range(1, config.TTS_API_RETRIES + 1):
        try:
            return _gemini_pcm(prompt_text)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_retryable_tts_error(exc) or attempt == config.TTS_API_RETRIES:
                raise
            print(
                f"[tts] Gemini server error, retrying in {delay:.1f}s "
                f"(attempt {attempt}/{config.TTS_API_RETRIES})..."
            )
            time.sleep(delay)
            delay *= 2
    raise last_exc


def _synthesize_chunk_audio(chunk_text, index, total, prev_chunk=None, next_chunk=None):
    prompt_variants = [
        _build_chunk_prompt(
            chunk_text,
            index,
            total,
            prev_chunk=prev_chunk,
            next_chunk=next_chunk,
        ),
        _build_minimal_prompt(chunk_text),
    ]
    last_exc = None
    for variant_index, prompt_text in enumerate(prompt_variants, 1):
        try:
            if variant_index > 1:
                print(f"[tts] falling back to simpler prompt for chunk {index}/{total}...")
            return _request_chunk_audio(prompt_text)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_retryable_tts_error(exc):
                raise
    split_chunks = _split_chunk_for_retry(chunk_text)
    if split_chunks:
        print(f"[tts] splitting chunk {index}/{total} after repeated server errors...")
        parts = []
        for split_index, split_text in enumerate(split_chunks, 1):
            split_prev = prev_chunk if split_index == 1 else split_chunks[0]
            split_next = next_chunk if split_index == len(split_chunks) else split_chunks[1]
            split_audio = _synthesize_chunk_audio(
                split_text,
                index,
                total,
                prev_chunk=split_prev,
                next_chunk=split_next,
            )
            parts.append(_trim_silence(split_audio, config.TTS_SAMPLE_RATE))
            if split_index < len(split_chunks):
                parts.append(_silence(config.TTS_SAMPLE_RATE, config.TTS_SHORT_PAUSE_MS))
        return _crossfade_join(parts, config.TTS_SAMPLE_RATE, 0)
    raise last_exc


def _flush_chunk(chunks, current_lines):
    if current_lines:
        chunks.append("\n\n".join(current_lines).strip())


def _split_long_paragraph(paragraph, max_chars, max_words):
    sentences = _split_sentences(paragraph) or [paragraph.strip()]
    chunks = []
    current_lines = []
    current_chars = 0
    current_words = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        separator_chars = 1 if current_lines else 0
        exceeds_chars = (
            current_lines
            and current_chars + separator_chars + len(sentence) > max_chars
        )
        exceeds_words = current_lines and current_words + sentence_words > max_words
        if exceeds_chars or exceeds_words:
            chunks.append(" ".join(current_lines).strip())
            current_lines = [sentence]
            current_chars = len(sentence)
            current_words = sentence_words
            continue

        current_lines.append(sentence)
        current_chars += separator_chars + len(sentence)
        current_words += sentence_words

    if current_lines:
        chunks.append(" ".join(current_lines).strip())
    return chunks


def _split_chunk_for_retry(chunk_text):
    sentences = _split_sentences(chunk_text)
    if len(sentences) < 2:
        return None
    midpoint = max(1, len(sentences) // 2)
    left = " ".join(sentences[:midpoint]).strip()
    right = " ".join(sentences[midpoint:]).strip()
    if not left or not right:
        return None
    return [left, right]


def _split_text(text, max_chars, max_words):
    text = _normalize_text(text)
    paragraphs = [paragraph for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks = []
    current_lines = []
    current_chars = 0
    current_words = 0
    for paragraph in paragraphs:
        paragraph_words = len(paragraph.split())
        paragraph_chunks = (
            [paragraph]
            if len(paragraph) <= max_chars and paragraph_words <= max_words
            else _split_long_paragraph(paragraph, max_chars, max_words)
        )
        for paragraph_chunk in paragraph_chunks:
            paragraph_chunk_words = len(paragraph_chunk.split())
            separator_chars = 2 if current_lines else 0
            exceeds_chars = (
                current_lines
                and current_chars + separator_chars + len(paragraph_chunk) > max_chars
            )
            exceeds_words = (
                current_lines and current_words + paragraph_chunk_words > max_words
            )
            if exceeds_chars or exceeds_words:
                _flush_chunk(chunks, current_lines)
                current_lines = [paragraph_chunk]
                current_chars = len(paragraph_chunk)
                current_words = paragraph_chunk_words
                continue

            current_lines.append(paragraph_chunk)
            current_chars += separator_chars + len(paragraph_chunk)
            current_words += paragraph_chunk_words

    _flush_chunk(chunks, current_lines)
    return chunks or [text]


def _apply_final_atempo(audio, sr, atempo):
    if audio.size == 0 or abs(atempo - 1.0) < 1e-3:
        return audio

    ffmpeg = config.find_ffmpeg()
    if not ffmpeg:
        return audio

    temp_input = os.path.join(config.TEMP_DIR, "tts_atempo_in.wav")
    temp_output = os.path.join(config.TEMP_DIR, "tts_atempo_out.wav")
    sf.write(temp_input, audio, sr)

    try:
        import subprocess

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                temp_input,
                "-filter:a",
                f"atempo={atempo}",
                temp_output,
            ],
            capture_output=True,
            check=True,
        )
        slowed, slowed_sr = sf.read(temp_output, dtype="float32")
        if slowed_sr != sr:
            return audio
        if slowed.ndim > 1:
            slowed = slowed[:, 0]
        return slowed.astype(np.float32)
    except Exception:  # noqa: BLE001
        return audio
    finally:
        for path in (temp_input, temp_output):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


def _iter_chunk_plan(chunks):
    total = len(chunks)
    for index, chunk in enumerate(chunks, 1):
        yield {
            "index": index,
            "total": total,
            "text": chunk,
            "prev_chunk": chunks[index - 2] if index > 1 else None,
            "next_chunk": chunks[index] if index < total else None,
        }


def _synthesize_chunked(text):
    chunk_word_limit = _estimate_chunk_word_limit()
    chunk_char_limit = min(config.TTS_MAX_INPUT_CHARS, config.TTS_MAX_CHUNK_CHARS)
    chunks = _split_text(text, chunk_char_limit, chunk_word_limit)
    print(f"[tts] generating narration with {len(chunks)} request(s)...")
    parts = []
    for item in _iter_chunk_plan(chunks):
        started = time.time()
        chunk_audio = _trim_silence(
            _synthesize_chunk_audio(
                item["text"],
                item["index"],
                item["total"],
                prev_chunk=item["prev_chunk"],
                next_chunk=item["next_chunk"],
            ),
            config.TTS_SAMPLE_RATE,
        )
        chunk_audio = _match_chunk_loudness(
            chunk_audio,
            config.TTS_TARGET_RMS,
            config.TTS_PEAK_LIMIT,
        )
        parts.append(chunk_audio)
        if item["index"] < len(chunks):
            parts.append(
                _silence(
                    config.TTS_SAMPLE_RATE,
                    _chunk_pause_ms(item["text"]),
                )
            )
        print(
            f"[tts] chunk {item['index']}/{len(chunks)} finished "
            f"in {time.time() - started:.0f}s"
        )
    return _crossfade_join(parts, config.TTS_SAMPLE_RATE, config.TTS_JOIN_CROSSFADE_MS)


def _synthesize_n_chunks(text: str) -> np.ndarray:
    normalized = _normalize_text(text)
    word_count = len(normalized.split())
    n = _determine_chunk_count(word_count)
    chunks = _split_into_n_chunks(normalized, n)
    actual = len(chunks)
    print(f"[tts] {word_count} words → {actual} request(s)...")

    parts: list[np.ndarray] = []
    for index, chunk_text in enumerate(chunks, 1):
        prev_chunk = chunks[index - 2] if index > 1 else None
        next_chunk = chunks[index] if index < actual else None
        started = time.time()
        chunk_audio = _trim_silence(
            _synthesize_chunk_audio(chunk_text, index, actual, prev_chunk=prev_chunk, next_chunk=next_chunk),
            config.TTS_SAMPLE_RATE,
        )
        chunk_audio = _match_chunk_loudness(chunk_audio, config.TTS_TARGET_RMS, config.TTS_PEAK_LIMIT)
        parts.append(chunk_audio)
        if index < actual:
            parts.append(_silence(config.TTS_SAMPLE_RATE, _chunk_pause_ms(chunk_text)))
        print(f"[tts] part {index}/{actual} done in {time.time() - started:.0f}s")

    joined = _crossfade_join(parts, config.TTS_SAMPLE_RATE, config.TTS_JOIN_CROSSFADE_MS)
    # Global loudness pass so the full joined audio lands at the target level
    return _match_chunk_loudness(joined, config.TTS_TARGET_RMS, config.TTS_PEAK_LIMIT)


def synthesize_full(text, out_path=None, final_atempo=None):
    output_path = out_path or config.FULL_AUDIO
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    audio = _synthesize_n_chunks(text)
    target_atempo = config.TTS_FINAL_ATEMPO if final_atempo is None else final_atempo
    audio = _apply_final_atempo(audio, config.TTS_SAMPLE_RATE, target_atempo)
    audio = _trim_silence(audio, config.TTS_SAMPLE_RATE)
    audio = _safe_peak_normalize(audio, config.TTS_PEAK_LIMIT)
    sf.write(output_path, audio, config.TTS_SAMPLE_RATE)
    return output_path, len(audio) / config.TTS_SAMPLE_RATE
