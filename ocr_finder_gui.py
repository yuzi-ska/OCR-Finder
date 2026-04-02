#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from ocr_finder import OCRFinder, MATCH_MODES


DEFAULT_UI_LANGUAGE = "zh"

UI_STRINGS = {
    "zh": {
        "window_title": "OCR Finder",
        "ready": "就绪",
        "ui_language_label": "界面语言",
        "ui_language_zh": "中文",
        "ui_language_en": "English",
        "target_label": "目标文本 [关键词1|关键词2|...]",
        "source_label": "源文件夹",
        "output_label": "输出文件夹",
        "browse_button": "浏览",
        "match_mode_label": "匹配模式",
        "mode_and": "与 (全部匹配)",
        "mode_or": "或 (任一匹配)",
        "mode_not": "非 (都不匹配)",
        "cpu_limit_label": "进程优先级",
        "cpu_low": "低",
        "cpu_medium": "中",
        "cpu_high": "高",
        "output_mode_label": "输出方式",
        "output_hardlink": "硬链接",
        "output_symlink": "软链接",
        "output_copy": "复制",
        "verbose_log": "详细日志",
        "start_button": "开始",
        "stop_button": "停止",
        "open_output_button": "打开输出目录",
        "clear_log_button": "清空日志",
        "missing_target_title": "缺少目标文本",
        "missing_target_message": "请输入要搜索的目标文本。",
        "missing_source_title": "缺少源文件夹",
        "missing_source_message": "请选择源文件夹。",
        "missing_output_title": "缺少输出文件夹",
        "missing_output_message": "请选择输出文件夹。",
        "starting_scan": "开始扫描...",
        "log_target": "搜索关键词: [{target}]",
        "log_source": "源文件夹: {source}",
        "log_output": "输出文件夹: {output}",
        "log_mode": "匹配模式: {mode}",
        "log_cpu": "CPU限制: {cpu}%",
        "stopping": "将在当前文件处理完后停止...",
        "stop_requested": "已请求停止。",
        "unexpected_error": "发生意外错误: {exc}",
        "error_prefix": "错误: {message}",
        "scan_completed": "扫描完成。",
        "scan_cancelled": "扫描已取消。",
        "scan_failed": "扫描失败。",
        "summary_scanned": "扫描: {value}",
        "summary_processed": "处理: {value}",
        "summary_matches": "匹配: {value}",
        "summary_skipped": "跳过: {value}",
        "summary_errors": "错误: {value}",
        "summary_elapsed": "耗时: {value:.2f}秒",
        "open_output_failed_title": "打开输出目录失败",
        "exit_title": "退出",
        "exit_message": "扫描仍在进行中。要停止并退出吗？",
    },
    "en": {
        "window_title": "OCR Finder",
        "ready": "Ready",
        "ui_language_label": "UI Language",
        "ui_language_zh": "中文",
        "ui_language_en": "English",
        "target_label": "Target text [kw1|kw2|...]",
        "source_label": "Source folder",
        "output_label": "Output folder",
        "browse_button": "Browse",
        "match_mode_label": "Match mode",
        "mode_and": "AND (all match)",
        "mode_or": "OR (any match)",
        "mode_not": "NOT (none match)",
        "cpu_limit_label": "Process priority",
        "cpu_low": "Low",
        "cpu_medium": "Medium",
        "cpu_high": "High",
        "output_mode_label": "Output mode",
        "output_hardlink": "Hard link",
        "output_symlink": "Symbolic link",
        "output_copy": "Copy",
        "verbose_log": "Verbose log",
        "start_button": "Start",
        "stop_button": "Stop",
        "open_output_button": "Open output",
        "clear_log_button": "Clear log",
        "missing_target_title": "Missing target",
        "missing_target_message": "Please enter the target text to search for.",
        "missing_source_title": "Missing source",
        "missing_source_message": "Please choose a source folder.",
        "missing_output_title": "Missing output",
        "missing_output_message": "Please choose an output folder.",
        "starting_scan": "Starting scan...",
        "log_target": "Keywords: [{target}]",
        "log_source": "Source: {source}",
        "log_output": "Output: {output}",
        "log_mode": "Mode: {mode}",
        "log_cpu": "CPU limit: {cpu}%",
        "stopping": "Stopping after current file...",
        "stop_requested": "Stop requested.",
        "unexpected_error": "Unexpected error: {exc}",
        "error_prefix": "ERROR: {message}",
        "scan_completed": "Scan completed.",
        "scan_cancelled": "Scan cancelled.",
        "scan_failed": "Scan failed.",
        "summary_scanned": "Scanned: {value}",
        "summary_processed": "Processed: {value}",
        "summary_matches": "Matches: {value}",
        "summary_skipped": "Skipped: {value}",
        "summary_errors": "Errors: {value}",
        "summary_elapsed": "Elapsed: {value:.2f}s",
        "open_output_failed_title": "Open output failed",
        "exit_title": "Exit",
        "exit_message": "A scan is still running. Stop and exit?",
    },
}


def resolve_ui_language(language: str | None) -> str:
    normalized = (language or "").strip().lower()
    if normalized.startswith("en"):
        return "en"
    return DEFAULT_UI_LANGUAGE


class OCRFinderGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.ui_language = resolve_ui_language(os.environ.get("OCR_FINDER_UI_LANGUAGE"))
        self.title(self.tr("window_title"))
        self.geometry("780x560")
        self.minsize(680, 460)

        # Store values as codes (not display text)
        self._cpu_code = "medium"  # low/medium/high
        self._output_mode_code = "hardlink"  # hardlink/symlink/copy

        cwd = Path.cwd()
        self.target_var = tk.StringVar()
        self.source_var = tk.StringVar(value=str(cwd))
        self.output_var = tk.StringVar(value=str(cwd / "output"))
        self.mode_var = tk.StringVar(value="or")
        self.cpu_var = tk.StringVar()  # Will be set in _build_ui
        self.output_mode_var = tk.StringVar()  # Will be set in _build_ui
        self.ui_lang_var = tk.StringVar()
        self.verbose_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value=self.tr("ready"))

        self.event_queue: queue.Queue[dict] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None

        self._build_ui()
        self.after(100, self.process_events)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def tr(self, key: str, **kwargs) -> str:
        raw_ui_language = self.__dict__.get("ui_language")
        if raw_ui_language is None:
            raw_ui_language = os.environ.get("OCR_FINDER_UI_LANGUAGE") or DEFAULT_UI_LANGUAGE
        ui_language = resolve_ui_language(raw_ui_language)
        translations = UI_STRINGS.get(ui_language, UI_STRINGS[DEFAULT_UI_LANGUAGE])
        template = translations.get(key, UI_STRINGS[DEFAULT_UI_LANGUAGE].get(key, key))
        return template.format(**kwargs)

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        form = ttk.Frame(self, padding=12)
        form.grid(row=0, column=0, sticky="nsew")
        form.columnconfigure(1, weight=1)

        # UI Language selection
        ui_lang_frame = ttk.Frame(form)
        ui_lang_frame.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(ui_lang_frame, text=self.tr("ui_language_label")).pack(side="left")
        self.ui_lang_var.set("中文" if self.ui_language == "zh" else "English")
        ui_lang_combo = ttk.Combobox(ui_lang_frame, textvariable=self.ui_lang_var, state="readonly", width=10)
        ui_lang_combo["values"] = (self.tr("ui_language_zh"), self.tr("ui_language_en"))
        ui_lang_combo.current(0 if self.ui_language == "zh" else 1)
        ui_lang_combo.pack(side="left", padx=(8, 0))
        ui_lang_combo.bind("<<ComboboxSelected>>", self._on_ui_language_change)

        ttk.Label(form, text=self.tr("target_label")).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.target_var).grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(form, text=self.tr("source_label")).grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.source_var).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Button(form, text=self.tr("browse_button"), command=self.choose_source).grid(row=2, column=2, padx=(8, 0), pady=4)

        ttk.Label(form, text=self.tr("output_label")).grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.output_var).grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Button(form, text=self.tr("browse_button"), command=self.choose_output).grid(row=3, column=2, padx=(8, 0), pady=4)

        # Match mode frame
        mode_frame = ttk.LabelFrame(form, text=self.tr("match_mode_label"), padding=8)
        mode_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        ttk.Radiobutton(mode_frame, text=self.tr("mode_or"), variable=self.mode_var, value="or").pack(side="left", padx=(0, 20))
        ttk.Radiobutton(mode_frame, text=self.tr("mode_and"), variable=self.mode_var, value="and").pack(side="left", padx=(0, 20))
        ttk.Radiobutton(mode_frame, text=self.tr("mode_not"), variable=self.mode_var, value="not").pack(side="left")

        # CPU limit frame
        cpu_frame = ttk.Frame(form)
        cpu_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        ttk.Label(cpu_frame, text=self.tr("cpu_limit_label")).pack(side="left")
        # Set display text based on stored code
        cpu_display_map = {
            "low": self.tr("cpu_low"),
            "medium": self.tr("cpu_medium"),
            "high": self.tr("cpu_high")
        }
        self.cpu_var.set(cpu_display_map.get(self._cpu_code, self.tr("cpu_medium")))
        cpu_combo = ttk.Combobox(cpu_frame, textvariable=self.cpu_var, state="readonly", width=15)
        cpu_combo["values"] = (self.tr("cpu_low"), self.tr("cpu_medium"), self.tr("cpu_high"))
        cpu_combo.current(["low", "medium", "high"].index(self._cpu_code))
        cpu_combo.pack(side="left", padx=(8, 0))

        # Output mode frame
        output_mode_frame = ttk.Frame(form)
        output_mode_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        ttk.Label(output_mode_frame, text=self.tr("output_mode_label")).pack(side="left")
        # Set display text based on stored code
        output_mode_display_map = {
            "hardlink": self.tr("output_hardlink"),
            "symlink": self.tr("output_symlink"),
            "copy": self.tr("output_copy")
        }
        self.output_mode_var.set(output_mode_display_map.get(self._output_mode_code, self.tr("output_hardlink")))
        output_mode_combo = ttk.Combobox(output_mode_frame, textvariable=self.output_mode_var, state="readonly", width=18)
        output_mode_combo["values"] = (self.tr("output_hardlink"), self.tr("output_symlink"), self.tr("output_copy"))
        output_mode_combo.current(["hardlink", "symlink", "copy"].index(self._output_mode_code))
        output_mode_combo.pack(side="left", padx=(8, 0))

        # Options frame
        options = ttk.Frame(form)
        options.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        options.columnconfigure(4, weight=1)

        ttk.Checkbutton(options, text=self.tr("verbose_log"), variable=self.verbose_var).grid(row=0, column=0, sticky="w")
        self.start_button = ttk.Button(options, text=self.tr("start_button"), command=self.start_scan)
        self.start_button.grid(row=0, column=1, padx=(12, 0))
        self.stop_button = ttk.Button(options, text=self.tr("stop_button"), command=self.stop_scan, state="disabled")
        self.stop_button.grid(row=0, column=2, padx=(8, 0))
        ttk.Button(options, text=self.tr("open_output_button"), command=self.open_output_folder).grid(row=0, column=3, padx=(8, 0))
        ttk.Button(options, text=self.tr("clear_log_button"), command=self.clear_log).grid(row=0, column=4, padx=(8, 0))

        # Progress frame
        progress_frame = ttk.Frame(self, padding=(12, 0, 12, 12))
        progress_frame.grid(row=1, column=0, sticky="ew")
        progress_frame.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(progress_frame, mode="determinate", maximum=1, value=0)
        self.progress.grid(row=0, column=0, sticky="ew")
        ttk.Label(progress_frame, textvariable=self.status_var).grid(row=1, column=0, sticky="w", pady=(6, 0))

        # Log frame
        log_frame = ttk.Frame(self, padding=(12, 0, 12, 12))
        log_frame.grid(row=2, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = ScrolledText(log_frame, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")

    def choose_source(self):
        path = filedialog.askdirectory(initialdir=self.source_var.get() or str(Path.cwd()))
        if path:
            self.source_var.set(path)

    def choose_output(self):
        path = filedialog.askdirectory(initialdir=self.output_var.get() or str(Path.cwd()))
        if path:
            self.output_var.set(path)

    def _on_ui_language_change(self, event=None):
        """Handle UI language change."""
        new_lang_display = self.ui_lang_var.get()
        if new_lang_display == "中文":
            self.ui_language = "zh"
        else:
            self.ui_language = "en"
        self._rebuild_ui()

    def _rebuild_ui(self):
        """Rebuild the UI with the new language."""
        # Save current values as codes
        target = self.target_var.get()
        source = self.source_var.get()
        output = self.output_var.get()
        mode = self.mode_var.get()
        verbose = self.verbose_var.get()

        # Convert display text to codes before rebuild
        cpu_display = self.cpu_var.get()
        if cpu_display in (self.tr("cpu_low"), "低", "Low") or "低" in cpu_display or "low" in cpu_display.lower():
            self._cpu_code = "low"
        elif cpu_display in (self.tr("cpu_high"), "高", "High") or "高" in cpu_display or "high" in cpu_display.lower():
            self._cpu_code = "high"
        else:
            self._cpu_code = "medium"

        output_mode_display = self.output_mode_var.get()
        if output_mode_display in (self.tr("output_hardlink"), "硬链接", "Hard link") or "硬链接" in output_mode_display or "hardlink" in output_mode_display.lower():
            self._output_mode_code = "hardlink"
        elif output_mode_display in (self.tr("output_symlink"), "软链接", "Symbolic link") or "软链接" in output_mode_display or "symlink" in output_mode_display.lower():
            self._output_mode_code = "symlink"
        else:
            self._output_mode_code = "copy"

        # Clear all widgets
        for widget in self.winfo_children():
            widget.destroy()

        # Update status
        self.status_var.set(self.tr("ready"))

        # Rebuild UI
        self._build_ui()

        # Restore values
        self.target_var.set(target)
        self.source_var.set(source)
        self.output_var.set(output)
        self.mode_var.set(mode)
        self.verbose_var.set(verbose)

    def append_log(self, message: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def start_scan(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return

        target = self.target_var.get().strip()
        source = self.source_var.get().strip()
        output = self.output_var.get().strip()

        # Use default language (ch) - RapidOCR supports Chinese and English
        language = "ch"

        mode = self.mode_var.get()
        cpu_setting = self.cpu_var.get()
        output_mode_setting = self.output_mode_var.get()

        # Convert CPU setting to percentage
        if "低" in cpu_setting or "low" in cpu_setting.lower():
            cpu = 30
        elif "高" in cpu_setting or "high" in cpu_setting.lower():
            cpu = 90
        else:
            cpu = 50

        # Convert output mode setting
        if "硬链接" in output_mode_setting or "hardlink" in output_mode_setting.lower():
            output_mode = "hardlink"
        elif "软链接" in output_mode_setting or "symlink" in output_mode_setting.lower():
            output_mode = "symlink"
        else:
            output_mode = "copy"

        if not target:
            messagebox.showerror(self.tr("missing_target_title"), self.tr("missing_target_message"))
            return
        if not source:
            messagebox.showerror(self.tr("missing_source_title"), self.tr("missing_source_message"))
            return
        if not output:
            messagebox.showerror(self.tr("missing_output_title"), self.tr("missing_output_message"))
            return

        self.stop_event.clear()
        self.progress.configure(maximum=1, value=0)
        self.status_var.set(self.tr("starting_scan"))
        self.append_log(self.tr("log_target", target=target))
        self.append_log(self.tr("log_source", source=source))
        self.append_log(self.tr("log_output", output=output))
        self.append_log(self.tr("log_mode", mode=mode.upper()))
        self.append_log(self.tr("log_cpu", cpu=cpu))

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        self.worker_thread = threading.Thread(
            target=self.run_scan,
            args=(
                target,
                source,
                output,
                self.verbose_var.get(),
                language,
                mode,
                cpu,
                output_mode,
            ),
            daemon=True,
        )
        self.worker_thread.start()

    def stop_scan(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set()
            self.status_var.set(self.tr("stopping"))
            self.append_log(self.tr("stop_requested"))

    def run_scan(
        self,
        target: str,
        source: str,
        output: str,
        verbose: bool,
        language: str,
        mode: str,
        cpu: int,
        output_mode: str,
    ):
        try:
            finder = OCRFinder(
                target_chars=target,
                source_dir=source,
                output_dir=output,
                verbose=verbose,
                language=language,
                match_mode=mode,
                cpu_limit=cpu,
                output_mode=output_mode,
                event_callback=self.event_queue.put,
                should_stop=self.stop_event.is_set,
            )
            exit_code = finder.run()
            self.event_queue.put(
                {
                    "kind": "worker_finished",
                    "message": "Worker finished",
                    "exit_code": exit_code,
                    "summary": finder.get_summary_data(),
                }
            )
        except Exception as exc:
            self.event_queue.put({"kind": "error", "message": self.tr("unexpected_error", exc=exc)})
            self.event_queue.put(
                {
                    "kind": "worker_finished",
                    "message": "Worker finished",
                    "exit_code": 1,
                    "summary": None,
                }
            )

    def process_events(self):
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break
            self.handle_event(event)

        self.after(100, self.process_events)

    def handle_event(self, event: dict):
        kind = event.get("kind")
        message = event.get("message", "")

        if kind == "scan_complete":
            total = max(int(event.get("total", 0)), 1)
            self.progress.configure(maximum=total, value=0)
            self.status_var.set(message)
            self.append_log(message)
            return

        if kind == "file_start":
            current = int(event.get("current", 0))
            total = max(int(event.get("total", 0)), 1)
            self.progress.configure(maximum=total, value=current)
            self.status_var.set(message)
            return

        if kind == "match":
            self.append_log(message)
            self.status_var.set(message)
            return

        if kind == "error":
            self.append_log(self.tr("error_prefix", message=message))
            self.status_var.set(message)
            return

        if kind == "log":
            self.append_log(message)
            return

        if kind == "progress":
            self.status_var.set(message)
            return

        if kind == "status":
            self.status_var.set(message)
            self.append_log(message)
            return

        if kind == "cancelled":
            summary = event.get("summary") or {}
            self.append_log(message)
            self.append_log(self.format_summary(summary))
            self.status_var.set(message)
            return

        if kind == "finished":
            summary = event.get("summary") or {}
            self.append_log(self.format_summary(summary))
            self.status_var.set(message)
            return

        if kind == "worker_finished":
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            exit_code = int(event.get("exit_code", 1))
            if exit_code == 0:
                self.status_var.set(self.tr("scan_completed"))
            elif exit_code == 2:
                self.status_var.set(self.tr("scan_cancelled"))
            else:
                self.status_var.set(self.tr("scan_failed"))
            return

    def format_summary(self, summary: dict) -> str:
        total_files = summary.get("total_files", 0)
        processed_files = summary.get("processed_files", 0)
        matches_found = summary.get("matches_found", 0)
        skipped_files = summary.get("skipped_files", 0)
        error_files = summary.get("error_files", 0)
        elapsed_time = summary.get("elapsed_time")

        parts = [
            self.tr("summary_scanned", value=total_files),
            self.tr("summary_processed", value=processed_files),
            self.tr("summary_matches", value=matches_found),
            self.tr("summary_skipped", value=skipped_files),
            self.tr("summary_errors", value=error_files),
        ]
        if isinstance(elapsed_time, (int, float)):
            parts.append(self.tr("summary_elapsed", value=elapsed_time))
        return " | ".join(parts)

    def open_output_folder(self):
        output_dir = Path(self.output_var.get().strip() or ".").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if os.name == "nt":
                os.startfile(output_dir)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(output_dir)])
            else:
                subprocess.Popen(["xdg-open", str(output_dir)])
        except Exception as exc:
            messagebox.showerror(self.tr("open_output_failed_title"), str(exc))

    def on_close(self):
        if self.worker_thread and self.worker_thread.is_alive():
            if not messagebox.askyesno(self.tr("exit_title"), self.tr("exit_message")):
                return
            self.stop_event.set()
        self.destroy()


def main():
    app = OCRFinderGUI()
    app.mainloop()


if __name__ == "__main__":
    main()