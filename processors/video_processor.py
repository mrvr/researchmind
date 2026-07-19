"""
processors/video_processor.py
Handles: .mp4, .mkv, .avi, .mov, .webm, .wmv, .flv

Pipeline:
  1. Extract audio track from video using FFmpeg
  2. Transcribe extracted audio with faster-whisper
  3. Send transcription to local LLM → concise video gist
"""

import subprocess
import shutil
from pathlib import Path
from utils.logger import get_logger
from config import TEMP_DIR, VIDEO_GIST_PROMPT
from processors.audio_processor import transcribe

log = get_logger("VideoProcessor")


def extract_audio(video_path: Path, output_dir: Path = TEMP_DIR) -> Path:
    """
    Extract the audio track from a video file using FFmpeg.

    Args:
        video_path:  Path to the input video file
        output_dir:  Directory to save the extracted .wav file

    Returns:
        Path to the extracted .wav audio file

    Raises:
        RuntimeError: if FFmpeg is not installed or extraction fails
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg is not installed or not in PATH.\n"
            "Install it with: sudo apt install ffmpeg"
        )

    audio_out = output_dir / f"{video_path.stem}_audio.wav"

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",                     # no video
        "-acodec", "pcm_s16le",    # WAV format
        "-ar", "16000",            # 16kHz — optimal for Whisper
        "-ac", "1",                # mono
        "-y",                      # overwrite if exists
        str(audio_out),
    ]

    log.info(f"  Extracting audio from {video_path.name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{result.stderr}")

    log.info(f"  ✓ Audio extracted → {audio_out.name}")
    return audio_out


def get_video_metadata(video_path: Path) -> dict:
    """
    Get basic video metadata using ffprobe.

    Returns:
        {"duration_sec": float, "width": int, "height": int, "fps": float}
    """
    if not shutil.which("ffprobe"):
        return {}

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    meta = {}
    if result.returncode == 0 and result.stdout.strip():
        parts = result.stdout.strip().split(",")
        try:
            meta["width"]    = int(parts[0]) if parts[0] else 0
            meta["height"]   = int(parts[1]) if parts[1] else 0
            if parts[2] and "/" in parts[2]:
                num, den = parts[2].split("/")
                meta["fps"] = round(int(num) / int(den), 2)
            meta["duration_sec"] = float(parts[3]) if len(parts) > 3 and parts[3] else 0
        except (ValueError, IndexError):
            pass
    return meta


def generate_gist(transcription: str, llm_client) -> str:
    """Generate a concise gist from a video transcription using the LLM."""
    if not transcription.strip():
        return "No speech detected in this video."
    prompt = VIDEO_GIST_PROMPT.format(transcription=transcription[:6000])
    return llm_client.generate(prompt)


def process(filepath: str | Path, llm_client=None) -> dict:
    """
    Full video processing pipeline: extract audio → transcribe → gist.

    Args:
        filepath:    Path to the video file
        llm_client:  LLMClient instance (optional)

    Returns:
        {
            "source": str,
            "type": "video",
            "transcription": str,   ← full raw transcription
            "gist": str,            ← LLM-generated gist / key points
            "language": str,
            "metadata": dict,       ← resolution, fps, duration
            "char_count": int,
            "error": str | None
        }
    """
    filepath = Path(filepath)
    log.info(f"Processing video: {filepath.name}")

    result = {
        "source": filepath.name,
        "type": "video",
        "transcription": "",
        "gist": "",
        "language": "unknown",
        "metadata": {},
        "char_count": 0,
        "error": None,
    }

    extracted_audio = None

    try:
        # Step 1: Get metadata
        result["metadata"] = get_video_metadata(filepath)
        if result["metadata"].get("duration_sec"):
            mins = int(result["metadata"]["duration_sec"] // 60)
            secs = int(result["metadata"]["duration_sec"] % 60)
            log.info(f"  Video duration: {mins}m {secs}s")

        # Step 2: Extract audio
        extracted_audio = extract_audio(filepath)

        # Step 3: Transcribe
        transcription, lang = transcribe(extracted_audio)
        result["transcription"] = transcription
        result["language"]      = lang
        result["char_count"]    = len(transcription)

        # Step 4: Generate gist via LLM
        if llm_client and transcription.strip():
            log.info(f"  Generating video gist via LLM ...")
            result["gist"] = generate_gist(transcription, llm_client)
            log.info(f"  ✓ Gist generated")
        else:
            result["gist"] = transcription[:500] + "..." if len(transcription) > 500 else transcription

    except Exception as e:
        result["error"] = str(e)
        log.error(f"  ✗ Video processing failed: {filepath.name} — {e}")

    finally:
        # Cleanup extracted audio temp file
        if extracted_audio and extracted_audio.exists():
            extracted_audio.unlink()
            log.info(f"  Cleaned up temp audio file")

    return result
