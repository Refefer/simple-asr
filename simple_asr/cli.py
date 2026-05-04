"""Command-line interface for simple-asr."""

import argparse
import shlex
import signal
import subprocess
import sys
from pathlib import Path

from .recorder import AudioRecorder
from .transcriber import Transcriber
from .utils import KeyboardHandler, get_next_filename, print_status, clear_line


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Formatter that shows defaults and preserves epilog formatting."""
    pass


def extract_exec(argv: list[str]) -> tuple[list[str], list[str] | None]:
    """Pull a find-style ``--exec CMD ... ;`` clause out of argv.

    Returns the remaining argv (with the clause removed) and the command
    template, or ``None`` if ``--exec`` was not present.
    """
    try:
        i = argv.index("--exec")
    except ValueError:
        return argv, None

    try:
        end = argv.index(";", i + 1)
    except ValueError:
        raise SystemExit("error: --exec must be terminated with ';' (escape as \\; in the shell)")

    cmd = argv[i + 1:end]
    if not cmd:
        raise SystemExit("error: --exec requires a command before ';'")

    return argv[:i] + argv[end + 1:], cmd


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    argv, exec_cmd = extract_exec(sys.argv[1:])

    parser = argparse.ArgumentParser(
        description="Real-time audio transcription using local Whisper model",
        formatter_class=HelpFormatter,
        epilog="""
Examples:
  simple-asr                                    # saves to current directory
  simple-asr --output ./transcripts
  simple-asr --name meeting --output ./transcripts
  simple-asr --file recording.wav
  simple-asr --file recording.mp3 --output ./transcripts
  simple-asr --exec wc -l {} \\;                 # run wc on each saved transcript

--exec CMD ... ;
  Run CMD on each saved transcript file. '{}' is replaced with the file
  path. The clause must be terminated with ';' (use '\\;' in the shell).
        """
    )
    parser.add_argument(
        "--name", "-n",
        default="transcript",
        help="Prefix for output filenames (e.g., 'meeting' -> meeting-v001.txt)"
    )
    parser.add_argument(
        "--output", "-o",
        default=".",
        type=Path,
        help="Output directory for transcription files"
    )
    parser.add_argument(
        "--file", "-f",
        type=Path,
        help="Audio file to transcribe (skips interactive recording)"
    )
    args = parser.parse_args(argv)
    args.exec_cmd = exec_cmd
    return args


def run_exec(cmd_template: list[str], file_path: Path) -> None:
    """Run an exec command template, substituting ``{}`` with the file path."""
    cmd = [arg.replace("{}", str(file_path)) for arg in cmd_template]
    print(f"Running --exec: {shlex.join(cmd)}", flush=True)
    try:
        result = subprocess.run(cmd, check=False)
        print(f"--exec finished: exit code {result.returncode}")
    except FileNotFoundError:
        print(f"\nError: --exec command not found: {cmd[0]}", file=sys.stderr)
    except OSError as e:
        print(f"\nError running --exec: {e}", file=sys.stderr)


class Application:
    """Main application controller."""

    def __init__(self, name: str, output_dir: Path, exec_cmd: list[str] | None = None):
        self.name = name
        self.output_dir = output_dir
        self.exec_cmd = exec_cmd
        self.recorder = AudioRecorder()
        self.transcriber = Transcriber()
        self.running = True
        self.recording = False

    def setup(self):
        """Initialize the application."""
        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load transcription model
        self.transcriber.load_model()

    def handle_signal(self, signum, frame):
        """Handle interrupt signals."""
        self.running = False
        if self.recording:
            self.recorder.stop()
        print("\nExiting...")

    def save_transcription(self, text: str) -> Path:
        """Save transcription to file and return the path."""
        filepath = get_next_filename(self.output_dir, self.name)
        filepath.write_text(text.strip() + "\n")
        return filepath

    def run(self):
        """Main application loop."""
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        print("\n" + "=" * 50)
        print("Simple ASR - Real-time Audio Transcription")
        print("=" * 50)
        print("\nControls:")
        print("  Enter  - Start/Stop recording")
        print("  Esc    - Exit application")
        print("  Ctrl-C - Exit application")
        print("\n" + "-" * 50)

        with KeyboardHandler() as keyboard:
            while self.running:
                if not self.recording:
                    print_status("\nPress Enter to start recording...")

                    # Wait for Enter or exit
                    while self.running:
                        key = keyboard.get_key()
                        if key == 'ENTER':
                            break
                        elif key in ('ESC', 'CTRL-C'):
                            self.running = False
                            break

                    if not self.running:
                        break

                    # Start recording
                    self.recording = True
                    self.recorder.start()
                    print_status("Recording... (Press Enter to stop)")

                else:
                    # Check for stop signal
                    key = keyboard.get_key()
                    if key == 'ENTER':
                        # Stop recording
                        print_status("\nProcessing...")
                        audio = self.recorder.stop()
                        self.recording = False

                        if len(audio) > 0:
                            # Transcribe
                            print_status("Transcribing...")
                            text = self.transcriber.transcribe(audio)

                            if text.strip():
                                # Save to file
                                filepath = self.save_transcription(text)
                                print(f"\n\nTranscription:")
                                print("-" * 40)
                                print(text)
                                print("-" * 40)
                                print(f"Saved to: {filepath}")
                                if self.exec_cmd:
                                    run_exec(self.exec_cmd, filepath)
                            else:
                                print("\nNo speech detected.")
                        else:
                            print("\nNo audio captured.")

                    elif key in ('ESC', 'CTRL-C'):
                        self.recorder.stop()
                        self.recording = False
                        self.running = False

        print("\nGoodbye!")


def transcribe_file(file_path: Path, name: str, output_dir: Path, exec_cmd: list[str] | None = None):
    """Transcribe an audio file and save the result."""
    if not file_path.is_file():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    transcriber = Transcriber()
    transcriber.load_model()

    print(f"\nTranscribing: {file_path}")
    text = transcriber.transcribe(str(file_path), show_progress=True)

    if text.strip():
        filepath = get_next_filename(output_dir, name)
        filepath.write_text(text.strip() + "\n")
        print(f"\n\nTranscription:")
        print("-" * 40)
        print(text)
        print("-" * 40)
        print(f"Saved to: {filepath}")
        if exec_cmd:
            run_exec(exec_cmd, filepath)
    else:
        print("\nNo speech detected.")


def main():
    """Entry point."""
    args = parse_args()

    try:
        if args.file:
            transcribe_file(args.file, args.name, args.output, args.exec_cmd)
        else:
            app = Application(args.name, args.output, args.exec_cmd)
            app.setup()
            app.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
