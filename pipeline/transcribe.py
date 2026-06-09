"""faster-whisper transcription helpers with finer scene-oriented splitting."""

from . import config

_model = None


def _load_model(device, compute_type):
    from faster_whisper import WhisperModel
    return WhisperModel(config.WHISPER_MODEL, device=device, compute_type=compute_type)


def _get_model():
    global _model
    if _model is None:
        try:
            _model = _load_model(config.WHISPER_DEVICE, config.WHISPER_COMPUTE)
            print(f"[whisper] {config.WHISPER_MODEL} on {config.WHISPER_DEVICE}")
        except Exception as exc:  # noqa: BLE001
            print(f"[whisper] {config.WHISPER_DEVICE} failed ({exc}); falling back to cpu")
            _model = _load_model("cpu", "int8")
    return _model


def _word_text(word):
    return getattr(word, "word", "") or ""


def _word_start(word, fallback):
    value = getattr(word, "start", None)
    return fallback if value is None else float(value)


def _word_end(word, fallback):
    value = getattr(word, "end", None)
    return fallback if value is None else float(value)


def _chunk_from_words(words, fallback_start, fallback_end):
    text = "".join(_word_text(word) for word in words).strip()
    if not text:
        return None
    start = _word_start(words[0], fallback_start)
    end = _word_end(words[-1], fallback_end)
    return {
        "start": float(start),
        "end": float(max(start, end)),
        "text": text,
    }


def _split_segment(segment):
    words = list(getattr(segment, "words", None) or [])
    seg_start = float(segment.start)
    seg_end = float(segment.end)
    seg_text = (segment.text or "").strip()
    if not words:
        return [{"start": seg_start, "end": seg_end, "text": seg_text}] if seg_text else []

    min_seconds = config.TRANSCRIPT_MIN_SEGMENT_SECONDS
    target_seconds = config.TRANSCRIPT_TARGET_SEGMENT_SECONDS
    max_seconds = config.TRANSCRIPT_MAX_SEGMENT_SECONDS

    chunks = []
    current = []
    current_start = seg_start

    for word in words:
        if not current:
            current_start = _word_start(word, seg_start)
        current.append(word)
        current_end = _word_end(word, seg_end)
        duration = current_end - current_start
        token = _word_text(word).strip()
        hard_boundary = any(mark in token for mark in ".!?")
        soft_boundary = any(mark in token for mark in ",;:")

        should_split = False
        if hard_boundary and duration >= min_seconds:
            should_split = True
        elif soft_boundary and duration >= target_seconds:
            should_split = True
        elif duration >= max_seconds:
            should_split = True

        if should_split:
            chunk = _chunk_from_words(current, current_start, current_end)
            if chunk:
                chunks.append(chunk)
            current = []

    if current:
        chunk = _chunk_from_words(current, current_start, seg_end)
        if chunk:
            chunks.append(chunk)

    return chunks


def _merge_tiny_chunks(chunks):
    min_seconds = config.TRANSCRIPT_MIN_SEGMENT_SECONDS
    output = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        duration = float(current["end"]) - float(current["start"])
        if duration < min_seconds and i + 1 < len(chunks):
            nxt = chunks[i + 1]
            output.append(
                {
                    "start": current["start"],
                    "end": nxt["end"],
                    "text": f"{current['text']} {nxt['text']}".strip(),
                }
            )
            i += 2
            continue
        output.append(current)
        i += 1
    return output


def transcribe(audio_path):
    global _model
    model = _get_model()
    try:
        segments, info = model.transcribe(
            audio_path,
            vad_filter=True,
            beam_size=5,
            word_timestamps=True,
        )
    except Exception as exc:  # noqa: BLE001
        if config.WHISPER_DEVICE == "cpu":
            raise
        print(f"[whisper] runtime {config.WHISPER_DEVICE} failure ({exc}); retrying on cpu")
        _model = _load_model("cpu", "int8")
        model = _model
        segments, info = model.transcribe(
            audio_path,
            vad_filter=True,
            beam_size=5,
            word_timestamps=True,
        )
    rows = []
    for segment in segments:
        rows.extend(_split_segment(segment))
    rows = _merge_tiny_chunks(rows)
    for index, row in enumerate(rows, 1):
        row["index"] = index
    print(f"[whisper] transcribed {len(rows)} segments from {info.duration:.1f}s of audio")
    return rows
