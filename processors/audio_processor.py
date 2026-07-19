"""
processors/audio_processor.py
Handles: .mp3, .wav, .m4a, .ogg, .flac, .aac, .wma

Pipeline:
  1. Transcribe audio with faster-whisper (runs fully locally)
  2. Send transcription to local LLM → concise audio summary
"""

from pathlib import Path
from utils.logger import get_logger
from config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_LANGUAGE,
    AUDIO_SUMMARY_PROMPT,
    GPU_INFO,
)

log = get_logger("AudioProcessor")

# Lazy-loaded Whisper model (shared across calls in the same process)
_whisper_model = None


def _load_whisper(device: str, compute_type: str):
    from faster_whisper import WhisperModel
    return WhisperModel(
        WHISPER_MODEL_SIZE,
        device=device,
        compute_type=compute_type,
        cpu_threads=0 if device == "cuda" else 4,   # all available threads when on CPU
        num_workers=1,                              # keep model resident between calls
    )


def _get_whisper_model():
    """
    Load the faster-whisper model once and cache it for the process lifetime.

    GPU behaviour:
      - CUDA + float16  → fastest, best quality  (NVIDIA GPU with fp16 kernel support)
      - CUDA + int8/float32 → fallback for GPUs whose CTranslate2 build doesn't
        support efficient float16 (e.g. older architectures dropped from
        newer wheels — config.WHISPER_COMPUTE_TYPE can guess wrong here,
        this actually verifies it and downgrades instead of crashing)
      - CPU  + int8     → no GPU available
    """
    global _whisper_model
    if _whisper_model is None:
        device, compute_type = WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

        log.info(f"Loading Whisper '{WHISPER_MODEL_SIZE}' | device={device} | compute={compute_type}")
        if GPU_INFO["available"]:
            log.info(
                f"  GPU: {GPU_INFO['name']} | "
                f"VRAM: {GPU_INFO['vram_mb']:,} MB | "
                f"Driver: {GPU_INFO['driver_version']} | "
                f"CUDA: {GPU_INFO['cuda_version']}"
            )

        try:
            _whisper_model = _load_whisper(device, compute_type)
        except ValueError as e:
            if device != "cuda":
                raise
            log.warning(
                f"Whisper failed to load with compute_type='{compute_type}' on CUDA ({e}) "
                f"— retrying with 'int8_float32'"
            )
            try:
                _whisper_model = _load_whisper("cuda", "int8_float32")
                compute_type = "int8_float32"
            except Exception as e2:
                log.warning(f"CUDA still unusable ({e2}) — falling back to CPU")
                _whisper_model = _load_whisper("cpu", "int8")
                device, compute_type = "cpu", "int8"

        log.info(f"  ✓ Whisper model loaded and ready (device={device}, compute={compute_type})")
    return _whisper_model


def transcribe(filepath: Path) -> tuple[str, str]:
    """
    Transcribe an audio file.

    Returns:
        (full_transcription: str, detected_language: str)
    """
    model = _get_whisper_model()
    log.info(f"  Transcribing: {filepath.name} ...")

    # On GPU: larger beam_size = better accuracy with minimal latency cost
    beam_size = 5 if WHISPER_DEVICE == "cuda" else 3

    segments, info = model.transcribe(
        str(filepath),
        language=WHISPER_LANGUAGE,
        beam_size=beam_size,
        best_of=5,                # Candidate sequences — GPU handles this fast
        vad_filter=True,          # Voice Activity Detection — skips silence
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=400,
        ),
        condition_on_previous_text=True,   # Better coherence across segments
        temperature=0.0,                   # Greedy decoding — deterministic
    )

    text_parts = []
    for seg in segments:
        text_parts.append(seg.text.strip())

    transcription = " ".join(text_parts)
    lang = info.language
    log.info(f"  ✓ Transcription complete ({len(transcription):,} chars, lang={lang})")
    return transcription, lang


def summarize_transcription(transcription: str, llm_client) -> str:
    """
    Send the transcription to the local LLM and get a concise summary.

    Args:
        transcription: Full transcribed text
        llm_client: Instance of core.llm_client.LLMClient

    Returns:
        Summary string
    """
    if not transcription.strip():
        return "No speech detected in this audio file."

    prompt = AUDIO_SUMMARY_PROMPT.format(transcription=transcription[:6000])
    return llm_client.generate(prompt)


def process(filepath: str | Path, llm_client=None) -> dict:
    """
    Full audio processing pipeline: transcribe → summarize.

    Args:
        filepath:   Path to the audio file
        llm_client: LLMClient instance (optional — if None, only transcription is returned)

    Returns:
        {
            "source": str,
            "type": "audio",
            "transcription": str,       ← full raw transcription
            "summary": str,             ← LLM-generated summary
            "language": str,
            "char_count": int,
            "error": str | None
        }
    """
    filepath = Path(filepath)
    log.info(f"Processing audio: {filepath.name}")

    result = {
        "source": filepath.name,
        "type": "audio",
        "transcription": "",
        "summary": "",
        "language": "unknown",
        "char_count": 0,
        "error": None,
    }

    try:
        transcription, lang = transcribe(filepath)
        result["transcription"] = transcription
        result["language"]      = lang
        result["char_count"]    = len(transcription)

        if llm_client:
            log.info(f"  Generating audio summary via LLM ...")
            result["summary"] = summarize_transcription(transcription, llm_client)
            log.info(f"  ✓ Summary generated")
        else:
            result["summary"] = transcription[:500] + "..." if len(transcription) > 500 else transcription

    except Exception as e:
        result["error"] = str(e)
        log.error(f"  ✗ Audio processing failed: {filepath.name} — {e}")

    return result
