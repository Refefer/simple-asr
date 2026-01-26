"""Command-line interface for simple-asr."""

import argparse
import signal
import sys
from pathlib import Path

from .recorder import AudioRecorder
from .transcriber import Transcriber
from .utils import KeyboardHandler, get_next_filename, print_status, clear_line


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Formatter that shows defaults and preserves epilog formatting."""
    pass


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time audio transcription using local Whisper model",
        formatter_class=HelpFormatter,
        epilog="""
Examples:
  simple-asr                                    # saves to current directory
  simple-asr --output ./transcripts
  simple-asr --name meeting --output ./transcripts
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
    return parser.parse_args()


class Application:
    """Main application controller."""

    def __init__(self, name: str, output_dir: Path):
        self.name = name
        self.output_dir = output_dir
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
                            else:
                                print("\nNo speech detected.")
                        else:
                            print("\nNo audio captured.")

                    elif key in ('ESC', 'CTRL-C'):
                        self.recorder.stop()
                        self.recording = False
                        self.running = False

        print("\nGoodbye!")


def main():
    """Entry point."""
    args = parse_args()

    app = Application(args.name, args.output)

    try:
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
