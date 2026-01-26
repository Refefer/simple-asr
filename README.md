# simple-asr

Real-time audio transcription CLI using a local Whisper model.

## Features

- Records audio from system microphone
- Real-time speech-to-text using faster-whisper
- Auto-detects GPU (CUDA) or CPU and selects optimal model
- Voice Activity Detection (VAD) filters silence
- Auto-incrementing output filenames

## Installation

### System Dependencies (Ubuntu/Debian)

```bash
sudo apt install portaudio19-dev
```

### Install as a Tool (recommended)

```bash
# Install globally with uv
uv tool install /path/to/simple-asr

# Then run from anywhere
simple-asr --name meeting --output ./transcripts
```

### Install with pipx

```bash
pipx install /path/to/simple-asr

# Then run from anywhere
simple-asr --name meeting --output ./transcripts
```

### Development Mode

```bash
cd simple-asr
uv sync
uv run simple-asr --name meeting --output ./transcripts
```

## Usage

```bash
simple-asr [--output <directory>] [--name <prefix>]
```

### Examples

```bash
# Saves transcript-v001.txt to current directory
simple-asr

# Save to specific directory
simple-asr --output ./transcripts

# Custom prefix -> meeting-v001.txt
simple-asr --name meeting --output ./transcripts
```

The output directory is created automatically if it doesn't exist.

### Controls

- **Enter** - Start/stop recording
- **Esc** - Exit application
- **Ctrl-C** - Exit application

## Output

Transcriptions are saved as plain text files with auto-incrementing versions:
- `meeting-v001.txt`
- `meeting-v002.txt`
- etc.

## Hardware Support

- **GPU (CUDA)**: Uses `small.en` model with float16 for better accuracy
- **CPU**: Uses `tiny.en` model with int8 quantization for speed

Models are downloaded automatically on first run (~75MB for tiny, ~500MB for small).
