#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Finder - local offline OCR file matcher.
Finds local images containing specified text using RapidOCR.
CPU-only version with multi-keyword support.
"""

import argparse
import hashlib
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Optional

try:
    import psutil
except ImportError:
    psutil = None

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}
COMMON_LANGUAGE_OPTIONS = ("ch", "en")
MATCH_MODES = ("and", "or", "not")
OUTPUT_MODES = ("hardlink", "symlink", "copy")

EventCallback = Callable[[dict], None]


def is_packaged_runtime() -> bool:
    """Check if running as packaged executable."""
    return bool(getattr(sys, "frozen", False) or "__compiled__" in globals())


def get_runtime_resource_dir() -> Path:
    """Get the directory containing bundled resources."""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        executable_path = getattr(sys, "executable", None)
        if executable_path:
            return Path(executable_path).resolve().parent
    if "__compiled__" in globals():
        executable_path = getattr(sys, "argv", [None])[0] or getattr(sys, "executable", None)
        if executable_path:
            return Path(executable_path).resolve().parent
    return Path(__file__).resolve().parent


def parse_keywords(target_text: str) -> list[str]:
    """Parse target text into list of keywords (split by comma or semicolon)."""
    if not target_text:
        return []
    # Split by comma, semicolon, or Chinese comma
    keywords = []
    for sep in [",", ";", "，", "；", "|"]:
        if sep in target_text:
            keywords = [k.strip() for k in target_text.split(sep) if k.strip()]
            break
    if not keywords:
        keywords = [target_text.strip()]
    return keywords


class OCRFinder:
    """Main class for local OCR file matching using RapidOCR."""

    def __init__(
        self,
        target_chars: str,
        source_dir: str,
        output_dir: str,
        verbose: bool = False,
        language: str = "ch",
        match_mode: str = "or",
        cpu_limit: int = 50,
        output_mode: str = "hardlink",
        event_callback: Optional[EventCallback] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ):
        self.target_text = target_chars
        self.source_dir = Path(source_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.verbose = verbose
        self.language = language
        self.match_mode = match_mode.lower() if match_mode else "or"
        self.cpu_limit = max(1, min(100, cpu_limit))
        self.output_mode = output_mode.lower() if output_mode else "hardlink"
        self.event_callback = event_callback
        self.should_stop = should_stop or (lambda: False)

        # Parse keywords
        self.keywords = parse_keywords(target_chars)
        self.normalized_keywords = [self.normalize_match_text(k) for k in self.keywords]

        self.seen_hashes: set[str] = set()
        self.matched_files: list[Path] = []
        self.processed_files = 0
        self.total_files = 0
        self.skipped_files = 0
        self.error_files = 0
        self.output_hashes_loaded = False

        self.ocr_backend: Optional[RapidOCR] = None
        self.start_time: Optional[float] = None
        self.original_process_priority = None

        # CPU limiting through process priority
        self._apply_cpu_limit_priority()

    def emit_event(self, kind: str, message: str, **kwargs) -> None:
        """Emit an event to the callback if available."""
        if self.event_callback:
            event = {"kind": kind, "message": message, **kwargs}
            self.event_callback(event)

    def log(self, message: str) -> None:
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[LOG] {message}")
        self.emit_event("log", message)

    def verbose_log(self, message: str) -> None:
        """Log only in verbose mode (detailed processing info)."""
        if self.verbose:
            print(f"[VERBOSE] {message}")
            self.emit_event("log", message)

    def status(self, message: str) -> None:
        """Print status message."""
        print(message)
        self.emit_event("status", message)

    def error(self, message: str) -> None:
        """Print error message."""
        print(f"错误: {message}" if any('\u4e00' <= c <= '\u9fff' for c in message) else f"ERROR: {message}")
        self.emit_event("error", message)

    def is_stop_requested(self) -> bool:
        """Check if stop has been requested."""
        return self.should_stop()

    def normalize_match_text(self, text: Optional[str]) -> str:
        """Normalize text for matching (case-insensitive, whitespace-normalized)."""
        if not text:
            return ""
        return " ".join(text.lower().split())

    def _apply_cpu_limit_priority(self) -> None:
        """Apply CPU limit by lowering process priority."""
        if psutil is None:
            return
        try:
            process = psutil.Process()
            self.original_process_priority = process.nice()

            # Map CPU limit to priority level
            # Lower CPU limit = higher priority value (lower priority)
            if self.cpu_limit <= 30:
                # Very low CPU: lowest priority (idle)
                if os.name == "nt":
                    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                else:
                    process.nice(19)  # Max nice value
            elif self.cpu_limit <= 50:
                # Low CPU: below normal priority
                if os.name == "nt":
                    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                else:
                    process.nice(10)
            elif self.cpu_limit <= 70:
                # Medium CPU: slightly reduced priority
                if os.name == "nt":
                    process.nice(psutil.NORMAL_PRIORITY_CLASS)
                else:
                    process.nice(5)
            # else: keep normal priority for high CPU limit

            self.log(f"Applied CPU limit priority (limit={self.cpu_limit}%)")
        except Exception as exc:
            self.log(f"Could not set process priority: {exc}")

    def _restore_process_priority(self) -> None:
        """Restore original process priority."""
        if psutil is None or self.original_process_priority is None:
            return
        try:
            process = psutil.Process()
            process.nice(self.original_process_priority)
        except Exception:
            pass

    def should_scan_file(self, file_path: Path) -> bool:
        """Check if file should be scanned (image only, no PDF)."""
        return file_path.suffix.lower() in SUPPORTED_EXTENSIONS

    def get_input_files(self, directory: Path) -> list[Path]:
        """Recursively scan directory for supported files."""
        input_files: list[Path] = []
        try:
            for root, dirs, files in os.walk(directory):
                root_path = Path(root).resolve()
                if root_path == self.output_dir:
                    dirs[:] = []
                    continue
                dirs[:] = [d for d in dirs if (root_path / d).resolve() != self.output_dir]
                for file_name in files:
                    file_path = root_path / file_name
                    if self.should_scan_file(file_path):
                        input_files.append(file_path)
        except Exception as exc:
            self.error(f"Error scanning directory: {exc}")
        return input_files

    def load_output_hashes(self) -> None:
        """Load hashes of existing output files to avoid duplicates."""
        if self.output_hashes_loaded:
            return
        if not self.output_dir.exists():
            self.output_hashes_loaded = True
            return
        try:
            for existing_file in self.output_dir.iterdir():
                if not existing_file.is_file():
                    continue
                try:
                    self.seen_hashes.add(self.calculate_file_hash(existing_file))
                except Exception:
                    pass
        except Exception:
            pass
        self.output_hashes_loaded = True

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for deduplication."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def initialize_backend(self) -> bool:
        """Initialize RapidOCR backend."""
        global RapidOCR

        if RapidOCR is None:
            try:
                from rapidocr_onnxruntime import RapidOCR as _RapidOCR
                RapidOCR = _RapidOCR
            except ImportError as exc:
                self.error(f"RapidOCR not available: {exc}")
                return False

        try:
            self.ocr_backend = RapidOCR()
            self.status("[INFO] RapidOCR backend initialized.")
            return True
        except Exception as exc:
            self.error(f"RapidOCR initialization failed: {exc}")
            return False

    def _run_ocr_backend(self, file_path: Path) -> Optional[str]:
        """Run OCR on image file using RapidOCR."""
        if self.ocr_backend is None:
            return None

        try:
            # RapidOCR accepts file path directly
            result, elapse = self.ocr_backend(str(file_path))
            if result is None or len(result) == 0:
                return ""

            texts = []
            for item in result:
                # RapidOCR returns: [box, text, score] or [box, (text, score)]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    # item[0] is box (coordinates), item[1] is text or (text, score)
                    text_part = item[1]
                    if isinstance(text_part, (list, tuple)):
                        # (text, score) format
                        text = str(text_part[0]) if text_part else ""
                    else:
                        # text directly
                        text = str(text_part)
                    if text:
                        texts.append(text)

            return " ".join(texts)
        except Exception as exc:
            self.log(f"OCR error: {exc}")
            return None

    def contains_target_text(self, extracted_text: Optional[str]) -> bool:
        """Check if extracted text matches based on match mode (and/or/not)."""
        normalized_extracted_text = self.normalize_match_text(extracted_text)
        if not normalized_extracted_text:
            return False

        if not self.normalized_keywords:
            return False

        if self.match_mode == "and":
            # All keywords must be present
            return all(kw in normalized_extracted_text for kw in self.normalized_keywords)
        elif self.match_mode == "not":
            # None of the keywords should be present
            return not any(kw in normalized_extracted_text for kw in self.normalized_keywords)
        else:  # "or" - default
            # At least one keyword must be present
            return any(kw in normalized_extracted_text for kw in self.normalized_keywords)

    def copy_to_output(self, file_path: Path) -> str:
        """Copy matching file to output directory using hardlink."""
        self.load_output_hashes()

        # Ensure output directory exists
        if not self.output_dir.exists():
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self.error(f"Failed to create output directory: {exc}")
                return "error"

        file_hash = self.calculate_file_hash(file_path)
        if file_hash in self.seen_hashes:
            return "skipped_duplicate"

        dest_path = self.output_dir / file_path.name
        counter = 1
        while dest_path.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            dest_path = self.output_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        try:
            if self.output_mode == "symlink":
                # Use symbolic link (works across devices)
                os.symlink(str(file_path), str(dest_path))
                self.seen_hashes.add(file_hash)
                return "symlinked"
            elif self.output_mode == "copy":
                # Direct copy
                shutil.copy2(file_path, dest_path)
                self.seen_hashes.add(file_hash)
                return "copied"
            else:
                # Default: hardlink (saves space, same device only)
                os.link(str(file_path), str(dest_path))
                self.seen_hashes.add(file_hash)
                return "linked"
        except OSError as e:
            # Hardlink/symlink failed, fallback to copy
            if self.output_mode != "copy":
                try:
                    shutil.copy2(file_path, dest_path)
                    self.seen_hashes.add(file_hash)
                    return "copied"
                except Exception as exc:
                    self.error(f"Failed to copy {file_path.name}: {exc}")
                    return "error"
            self.error(f"Failed to copy {file_path.name}: {e}")
            return "error"
        except Exception as exc:
            self.error(f"Failed to output {file_path.name}: {exc}")
            return "error"

    def process_file(self, file_path: Path) -> bool:
        """Process a single file and return True if it matched."""
        self.processed_files += 1
        self.emit_event("file_start", f"Processing {file_path.name}",
                       current=self.processed_files, total=self.total_files)

        try:
            # Run OCR directly on file path
            extracted_text = self._run_ocr_backend(file_path)
            if extracted_text is None:
                self.error_files += 1
                self.verbose_log(f"OCR failed: {file_path.name}")
                return False

            if extracted_text == "":
                self.verbose_log(f"No text found: {file_path.name}")
                return False

            # Show OCR result in verbose mode
            self.verbose_log(f"OCR {file_path.name}: {extracted_text[:100]}{'...' if len(extracted_text) > 100 else ''}")

            if self.contains_target_text(extracted_text):
                copy_result = self.copy_to_output(file_path)
                if copy_result in ("linked", "symlinked", "copied"):
                    self.matched_files.append(file_path)
                    matched_keywords = [kw for kw in self.keywords if self.normalize_match_text(kw) in self.normalize_match_text(extracted_text)]
                    # Use symbol format for keywords: [kw1|kw2|kw3]
                    keyword_symbols = "[" + "|".join(matched_keywords) + "]"
                    self.emit_event("match", f"✓ {file_path.name} {keyword_symbols}")
                    self.log(f"Found match: {file_path.name}")
                elif copy_result == "skipped_duplicate":
                    self.skipped_files += 1
                    self.log(f"Skipped duplicate: {file_path.name}")
                elif copy_result == "error":
                    self.error_files += 1
                return True
            else:
                self.verbose_log(f"No match: {file_path.name}")
            return False
        except Exception as exc:
            self.error_files += 1
            self.error(f"Failed to process {file_path.name}: {exc}")
            return False

    def get_summary_data(self, total_files: int = None, elapsed_time: float = None) -> dict:
        """Get summary data for reporting."""
        if total_files is not None:
            self.total_files = total_files
        if elapsed_time is None and self.start_time:
            elapsed_time = time.time() - self.start_time

        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "matches_found": len(self.matched_files),
            "skipped_files": self.skipped_files,
            "error_files": self.error_files,
            "elapsed_time": elapsed_time,
            "output_dir": str(self.output_dir),
            "matched_files": [str(path) for path in self.matched_files],
        }

    def run(self) -> int:
        """Main entry point for OCR matching."""
        print("=" * 60)
        print("OCR Finder - Local Offline OCR Matcher (RapidOCR)")
        print("=" * 60)
        self.emit_event("status", "Starting scan")

        if not self.source_dir.exists():
            self.error(f"Source directory does not exist: {self.source_dir}")
            return 1

        if not self.source_dir.is_dir():
            self.error(f"Source path is not a directory: {self.source_dir}")
            return 1

        mode_desc = {"and": "ALL keywords must match", "or": "ANY keyword matches", "not": "NO keyword should match"}

        print("\nConfiguration:")
        print(f"  Keywords: {', '.join(self.keywords)}")
        print(f"  Match mode: {self.match_mode.upper()} ({mode_desc.get(self.match_mode, '')})")
        print(f"  Output mode: {self.output_mode}")
        print(f"  Source directory: {self.source_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  OCR language: {self.language}")
        print(f"  CPU limit: {self.cpu_limit}%")
        print(f"  Verbose mode: {'On' if self.verbose else 'Off'}")

        print("\nScanning for images...")
        self.emit_event("status", "Scanning for images...")
        input_files = self.get_input_files(self.source_dir)

        if not input_files:
            print("No supported image files found in source directory.")
            self.emit_event("finished", "No supported files found.",
                          summary=self.get_summary_data(total_files=0, elapsed_time=0.0))
            return 0

        if not self.initialize_backend():
            return 1

        self.total_files = len(input_files)
        print(f"Found {self.total_files} image file(s)")
        print("\nProcessing files...")
        print("-" * 60)
        self.emit_event("scan_complete", f"Found {self.total_files} image file(s)",
                       total=self.total_files)

        self.start_time = time.time()

        for index, file_path in enumerate(input_files, 1):
            if self.is_stop_requested():
                elapsed_time = time.time() - self.start_time
                cancel_message = "Scan cancelled by user."
                print("-" * 60)
                print(cancel_message)
                self.emit_event(
                    "cancelled",
                    cancel_message,
                    summary=self.get_summary_data(elapsed_time=elapsed_time),
                )
                return 2

            self.process_file(file_path)

        elapsed_time = time.time() - self.start_time
        self._restore_process_priority()  # Restore priority after scan
        summary = self.get_summary_data(elapsed_time=elapsed_time)

        print("-" * 60)
        print("\nSummary:")
        print(f"  Scanned: {summary['total_files']} files")
        print(f"  Processed: {summary['processed_files']} files")
        print(f"  Matches: {summary['matches_found']} files")
        print(f"  Skipped: {summary['skipped_files']} files")
        print(f"  Errors: {summary['error_files']} files")
        print(f"  Elapsed: {elapsed_time:.2f} seconds")
        print(f"  Output: {self.output_dir}")

        if self.matched_files:
            print("\nMatched files:")
            for matched_path in self.matched_files:
                print(f"  - {matched_path.name}")

        self.emit_event("finished", "Scan completed.", summary=summary)
        return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OCR Finder - Find local images containing specific text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chinese_ocr_finder.py -t 你好 -s ./images
  python chinese_ocr_finder.py -t "hello,world" -s ./documents -m or
  python chinese_ocr_finder.py -t "A,B,C" -s ./docs -m and
  python chinese_ocr_finder.py -t "spam" -s ./photos -m not
        """,
    )

    parser.add_argument(
        "-t", "--target",
        required=True,
        help="Target keywords to search for (separate multiple keywords with comma)",
    )

    parser.add_argument(
        "-s", "--source",
        required=True,
        help="Source directory to scan for images",
    )

    parser.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory for matching files (default: ./output)",
    )

    parser.add_argument(
        "-l", "--language",
        default="ch",
        choices=COMMON_LANGUAGE_OPTIONS,
        help="OCR language (default: ch for Chinese)",
    )

    parser.add_argument(
        "-m", "--mode",
        default="or",
        choices=MATCH_MODES,
        help="Match mode: 'and' (all keywords), 'or' (any keyword), 'not' (no keyword) (default: or)",
    )

    parser.add_argument(
        "-c", "--cpu",
        type=int,
        default=50,
        help="CPU usage limit percentage (1-100, default: 50)",
    )

    parser.add_argument(
        "--output-mode",
        default="hardlink",
        choices=OUTPUT_MODES,
        help="Output mode: 'hardlink' (default, saves space), 'symlink' (cross-device), 'copy' (full copy)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    finder = OCRFinder(
        target_chars=args.target,
        source_dir=args.source,
        output_dir=args.output,
        verbose=args.verbose,
        language=args.language,
        match_mode=args.mode,
        cpu_limit=args.cpu,
        output_mode=args.output_mode,
    )

    return finder.run()


if __name__ == "__main__":
    sys.exit(main())