# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Install system dependency (Ubuntu/Debian)
sudo apt install portaudio19-dev

# Development mode
uv sync
uv run simple-asr

# Install as tool (with cache bypass for updates)
uv tool install --force --no-cache /path/to/simple-asr
```

## Architecture

This is a real-time audio transcription CLI using faster-whisper (CTranslate2-based Whisper).

**Components:**
- `cli.py` - Entry point, main application loop with keyboard controls (Enter=start/stop, Esc=exit)
- `recorder.py` - Audio capture using sounddevice.InputStream (16kHz mono float32)
- `transcriber.py` - faster-whisper integration with auto GPU/CPU detection via ctranslate2
- `utils.py` - File naming (`{prefix}-v{NNN}.txt` pattern), non-blocking keyboard input (termios)

**Key Technical Details:**
- CUDA detection uses `ctranslate2.get_supported_compute_types("cuda")` (not torch)
- GPU uses `small.en` model with float16; CPU uses `tiny.en` with int8 quantization
- VAD (silero-vad) is bundled with faster-whisper
- Models download automatically on first run (~75MB tiny, ~500MB small)

## Git Policy

Never add "Co-Authored-By: Claude" to commits in this repository.
