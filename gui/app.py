#!/usr/bin/env python3
"""
Flash-MoE Universal GUI
Cross-platform AI model manager and inference runner
Works on: macOS (Intel + Apple Silicon), Linux, Windows
"""

import sys
import os
import json
import subprocess
import threading
import platform
import shutil
from pathlib import Path

try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
except ImportError:
    print("PyQt5 not found. Install with: pip install PyQt5")
    sys.exit(1)

# ─── Platform detection ───────────────────────────────────────────────────────
PLATFORM = platform.system()   # "Darwin", "Linux", "Windows"
ARCH     = platform.machine()  # "x86_64", "arm64", "AMD64", etc.
IS_MACOS   = PLATFORM == "Darwin"
IS_LINUX   = PLATFORM == "Linux"
IS_WINDOWS = PLATFORM == "Windows"
IS_APPLE_SILICON = IS_MACOS and ARCH == "arm64"

# Find flash_moe binary
def find_binary():
    candidates = [
        Path(__file__).parent.parent / "build" / "flash_moe",
        Path(__file__).parent.parent / "build" / "flash_moe.exe",
        Path("/usr/local/bin/flash_moe"),
        shutil.which("flash_moe"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None

FLASH_MOE_BIN = find_binary()

# ─── Color palette ───────────────────────────────────────────────────────────
COLORS = {
    "bg":       "#1a1b26",
    "bg2":      "#24253a",
    "bg3":      "#2a2b40",
    "accent":   "#7aa2f7",
    "accent2":  "#bb9af7",
    "green":    "#9ece6a",
    "yellow":   "#e0af68",
    "red":      "#f7768e",
    "text":     "#c0caf5",
    "subtext":  "#565f89",
    "border":   "#3b3f5c",
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
    font-family: "SF Pro Text", "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}}
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    background: {COLORS['bg2']};
    border-radius: 8px;
}}
QTabBar::tab {{
    background: {COLORS['bg3']};
    color: {COLORS['subtext']};
    padding: 8px 20px;
    border: none;
    border-radius: 4px;
    margin: 2px;
}}
QTabBar::tab:selected {{
    background: {COLORS['accent']};
    color: white;
    font-weight: bold;
}}
QPushButton {{
    background: {COLORS['accent']};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: 600;
}}
QPushButton:hover   {{ background: {COLORS['accent2']}; }}
QPushButton:pressed {{ background: {COLORS['bg3']}; }}
QPushButton:disabled {{ background: {COLORS['subtext']}; opacity: 0.6; }}
QPushButton.danger  {{ background: {COLORS['red']}; }}
QPushButton.success {{ background: {COLORS['green']}; color: #1a1b26; }}
QLineEdit, QTextEdit, QPlainTextEdit {{
    background: {COLORS['bg3']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 6px 10px;
    color: {COLORS['text']};
    selection-background-color: {COLORS['accent']};
}}
QLineEdit:focus, QTextEdit:focus {{ border-color: {COLORS['accent']}; }}
QComboBox {{
    background: {COLORS['bg3']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 6px 10px;
    color: {COLORS['text']};
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background: {COLORS['bg2']};
    border: 1px solid {COLORS['border']};
    color: {COLORS['text']};
    selection-background-color: {COLORS['accent']};
}}
QScrollBar:vertical {{
    background: {COLORS['bg3']};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['border']};
    border-radius: 4px;
    min-height: 20px;
}}
QLabel {{ color: {COLORS['text']}; }}
QLabel.title  {{ font-size: 18px; font-weight: bold; color: {COLORS['accent']}; }}
QLabel.sub    {{ color: {COLORS['subtext']}; font-size: 11px; }}
QProgressBar {{
    background: {COLORS['bg3']};
    border-radius: 4px;
    border: 1px solid {COLORS['border']};
    text-align: center;
    color: white;
}}
QProgressBar::chunk {{
    background: {COLORS['accent']};
    border-radius: 4px;
}}
QListWidget {{
    background: {COLORS['bg2']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    outline: none;
}}
QListWidget::item {{
    padding: 10px 14px;
    border-bottom: 1px solid {COLORS['border']};
}}
QListWidget::item:selected {{
    background: {COLORS['accent']};
    color: white;
    border-radius: 4px;
}}
QSlider::groove:horizontal {{
    background: {COLORS['bg3']};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {COLORS['accent']};
    width: 16px; height: 16px;
    border-radius: 8px;
    margin: -6px 0;
}}
QGroupBox {{
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    color: {COLORS['subtext']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    color: {COLORS['accent']};
    font-weight: 600;
}}
QCheckBox {{ color: {COLORS['text']}; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 2px solid {COLORS['border']};
    border-radius: 3px;
    background: {COLORS['bg3']};
}}
QCheckBox::indicator:checked {{
    background: {COLORS['accent']};
    border-color: {COLORS['accent']};
    image: url(:/icons/check.png);
}}
QSpinBox, QDoubleSpinBox {{
    background: {COLORS['bg3']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 4px 8px;
    color: {COLORS['text']};
}}
"""

# ─── Model registry (predefined models) ──────────────────────────────────────
KNOWN_MODELS = [
    {"name": "Qwen3.5-397B-A17B", "size": "~209GB (4-bit)", "params": "397B", "arch": "MoE", "active": "17B/tok", "ram": "48GB", "hf": "mlx-community/Qwen3.5-397B-A17B-4bit"},
    {"name": "Qwen2.5-72B",       "size": "~40GB",          "params": "72B",  "arch": "Dense", "active": "72B/tok", "ram": "24GB+", "hf": "Qwen/Qwen2.5-72B-Instruct"},
    {"name": "Qwen2.5-32B",       "size": "~19GB",          "params": "32B",  "arch": "Dense", "active": "32B/tok", "ram": "16GB+", "hf": "Qwen/Qwen2.5-32B-Instruct"},
    {"name": "Qwen2.5-14B",       "size": "~9GB",           "params": "14B",  "arch": "Dense", "active": "14B/tok", "ram": "12GB",  "hf": "Qwen/Qwen2.5-14B-Instruct"},
    {"name": "Qwen2.5-7B",        "size": "~4.5GB",         "params": "7B",   "arch": "Dense", "active": "7B/tok",  "ram": "8GB",   "hf": "Qwen/Qwen2.5-7B-Instruct"},
    {"name": "Qwen2.5-3B",        "size": "~2GB",           "params": "3B",   "arch": "Dense", "active": "3B/tok",  "ram": "4GB",   "hf": "Qwen/Qwen2.5-3B-Instruct"},
    {"name": "Qwen2.5-1.5B",      "size": "~1GB",           "params": "1.5B", "arch": "Dense", "active": "1.5B/tok","ram": "2GB",   "hf": "Qwen/Qwen2.5-1.5B-Instruct"},
    {"name": "Qwen2.5-0.5B",      "size": "~0.5GB",         "params": "0.5B", "arch": "Dense", "active": "0.5B/tok","ram": "1GB",   "hf": "Qwen/Qwen2.5-0.5B-Instruct"},
    {"name": "Llama-3.1-70B",     "size": "~40GB",          "params": "70B",  "arch": "Dense", "active": "70B/tok", "ram": "24GB+", "hf": "meta-llama/Meta-Llama-3.1-70B-Instruct"},
    {"name": "Llama-3.1-8B",      "size": "~4.5GB",         "params": "8B",   "arch": "Dense", "active": "8B/tok",  "ram": "8GB",   "hf": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
    {"name": "Llama-3.2-3B",      "size": "~2GB",           "params": "3B",   "arch": "Dense", "active": "3B/tok",  "ram": "4GB",   "hf": "meta-llama/Llama-3.2-3B-Instruct"},
    {"name": "Mistral-7B",        "size": "~4.5GB",         "params": "7B",   "arch": "Dense", "active": "7B/tok",  "ram": "8GB",   "hf": "mistralai/Mistral-7B-Instruct-v0.3"},
    {"name": "Mixtral-8x7B",      "size": "~26GB",          "params": "47B",  "arch": "MoE",   "active": "12.9B/tok","ram": "16GB+", "hf": "mistralai/Mixtral-8x7B-Instruct-v0.1"},
    {"name": "Phi-4",             "size": "~10GB",          "params": "14B",  "arch": "Dense", "active": "14B/tok", "ram": "12GB",  "hf": "microsoft/phi-4"},
    {"name": "Phi-3-mini",        "size": "~2GB",           "params": "3.8B", "arch": "Dense", "active": "3.8B/tok","ram": "4GB",   "hf": "microsoft/Phi-3-mini-4k-instruct"},
    {"name": "Gemma-2-27B",       "size": "~16GB",          "params": "27B",  "arch": "Dense", "active": "27B/tok", "ram": "16GB",  "hf": "google/gemma-2-27b-it"},
    {"name": "Gemma-2-9B",        "size": "~6GB",           "params": "9B",   "arch": "Dense", "active": "9B/tok",  "ram": "8GB",   "hf": "google/gemma-2-9b-it"},
    {"name": "Gemma-2-2B",        "size": "~2GB",           "params": "2B",   "arch": "Dense", "active": "2B/tok",  "ram": "3GB",   "hf": "google/gemma-2-2b-it"},
]

# ─── Worker thread for inference ─────────────────────────────────────────────
class InferenceWorker(QThread):
    token_received = pyqtSignal(str)
    stats_updated  = pyqtSignal(dict)
    finished       = pyqtSignal()
    error          = pyqtSignal(str)

    def __init__(self, binary, model_dir, prompt, settings):
        super().__init__()
        self.binary    = binary
        self.model_dir = model_dir
        self.prompt    = prompt
        self.settings  = settings
        self._stop     = False
        self.process   = None

    def run(self):
        if not self.binary:
            self.error.emit("flash_moe binary not found. Please build the project first.")
            return

        cmd = [
            self.binary,
            "--model", self.model_dir,
            "--prompt", self.prompt,
            "--tokens", str(self.settings.get("max_tokens", 512)),
            "--temp",   str(self.settings.get("temperature", 0.7)),
            "--top-p",  str(self.settings.get("top_p", 0.9)),
            "--backend", self.settings.get("backend", "auto"),
        ]
        if self.settings.get("threads"):
            cmd += ["--threads", str(self.settings["threads"])]
        if self.settings.get("use_2bit"):
            cmd.append("--2bit")
        if self.settings.get("verbose"):
            cmd.append("--verbose")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            stats = {"tokens": 0, "tps": 0.0}
            for line in self.process.stdout:
                if self._stop:
                    self.process.terminate()
                    break
                self.token_received.emit(line)
                stats["tokens"] += len(line.split())
                self.stats_updated.emit(stats)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def stop(self):
        self._stop = True
        if self.process:
            self.process.terminate()

# ─── Model Manager Widget ──────────────────────────────────────────────────
class ModelManagerWidget(QWidget):
    model_selected = pyqtSignal(str)  # emits model directory path

    def __init__(self):
        super().__init__()
        self.local_models = {}  # name → path
        self._build_ui()
        self._scan_local_models()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header
        hdr = QLabel("🤖 Model Manager")
        hdr.setProperty("class", "title")
        hdr.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['accent']};")
        layout.addWidget(hdr)

        # Tabs: Local / Download
        tabs = QTabWidget()
        tabs.addTab(self._build_local_tab(),    "📁 Local Models")
        tabs.addTab(self._build_download_tab(), "⬇️ Download Models")
        layout.addWidget(tabs)

    def _build_local_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(8)

        # Search + Add
        top = QHBoxLayout()
        self.local_search = QLineEdit()
        self.local_search.setPlaceholderText("🔍 Filter models...")
        self.local_search.textChanged.connect(self._filter_local)
        top.addWidget(self.local_search)

        add_btn = QPushButton("📂 Add Folder")
        add_btn.clicked.connect(self._add_model_folder)
        top.addWidget(add_btn)
        v.addLayout(top)

        # Model list
        self.local_list = QListWidget()
        self.local_list.itemDoubleClicked.connect(self._use_model)
        v.addWidget(self.local_list)

        # Buttons
        btns = QHBoxLayout()
        use_btn = QPushButton("✅ Use Selected")
        use_btn.setStyleSheet(f"background: {COLORS['green']}; color: #1a1b26;")
        use_btn.clicked.connect(lambda: self._use_model(self.local_list.currentItem()))
        btns.addWidget(use_btn)

        info_btn = QPushButton("ℹ️ Info")
        info_btn.clicked.connect(self._show_model_info)
        btns.addWidget(info_btn)

        btns.addStretch()
        v.addLayout(btns)

        # Status
        self.local_status = QLabel("No models found. Add a folder or download a model.")
        self.local_status.setStyleSheet(f"color: {COLORS['subtext']}; font-size: 11px;")
        v.addWidget(self.local_status)
        return w

    def _build_download_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(8)

        info = QLabel("Select a model to download from Hugging Face Hub:")
        info.setStyleSheet(f"color: {COLORS['subtext']};")
        v.addWidget(info)

        # Filter
        self.dl_search = QLineEdit()
        self.dl_search.setPlaceholderText("🔍 Filter by name, size, or architecture...")
        self.dl_search.textChanged.connect(self._filter_download_list)
        v.addWidget(self.dl_search)

        # Table
        self.dl_table = QTableWidget()
        self.dl_table.setColumnCount(6)
        self.dl_table.setHorizontalHeaderLabels(["Model", "Params", "Size", "Arch", "Active Params", "Min RAM"])
        self.dl_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.dl_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.dl_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.dl_table.setAlternatingRowColors(True)
        self.dl_table.setStyleSheet(f"""
            QTableWidget {{ background: {COLORS['bg2']}; alternate-background-color: {COLORS['bg3']}; gridline-color: {COLORS['border']}; }}
            QHeaderView::section {{ background: {COLORS['bg3']}; color: {COLORS['accent']}; padding: 6px; border: none; font-weight: bold; }}
        """)
        self._populate_download_table(KNOWN_MODELS)
        v.addWidget(self.dl_table)

        # Download controls
        dl_row = QHBoxLayout()
        self.dl_dir_edit = QLineEdit()
        self.dl_dir_edit.setPlaceholderText("Save directory...")
        dl_row.addWidget(self.dl_dir_edit)
        browse = QPushButton("📁")
        browse.setFixedWidth(36)
        browse.clicked.connect(self._browse_dl_dir)
        dl_row.addWidget(browse)

        dl_btn = QPushButton("⬇️ Download Selected")
        dl_btn.clicked.connect(self._start_download)
        dl_row.addWidget(dl_btn)
        v.addLayout(dl_row)

        # Progress
        self.dl_progress = QProgressBar()
        self.dl_progress.setVisible(False)
        v.addWidget(self.dl_progress)

        self.dl_status = QLabel("")
        self.dl_status.setStyleSheet(f"color: {COLORS['subtext']}; font-size: 11px;")
        v.addWidget(self.dl_status)
        return w

    def _populate_download_table(self, models):
        self.dl_table.setRowCount(len(models))
        for i, m in enumerate(models):
            self.dl_table.setItem(i, 0, QTableWidgetItem(m["name"]))
            self.dl_table.setItem(i, 1, QTableWidgetItem(m["params"]))
            self.dl_table.setItem(i, 2, QTableWidgetItem(m["size"]))
            arch_item = QTableWidgetItem(m["arch"])
            if m["arch"] == "MoE":
                arch_item.setForeground(QColor(COLORS["accent2"]))
            self.dl_table.setItem(i, 3, arch_item)
            self.dl_table.setItem(i, 4, QTableWidgetItem(m["active"]))
            ram_item = QTableWidgetItem(m["ram"])
            if "+" in m["ram"] or int(m["ram"].replace("GB", "").replace("+","").strip()) >= 16:
                ram_item.setForeground(QColor(COLORS["yellow"]))
            self.dl_table.setItem(i, 5, ram_item)
        self.dl_table.resizeRowsToContents()

    def _filter_download_list(self, text):
        filtered = [m for m in KNOWN_MODELS
                    if text.lower() in m["name"].lower()
                    or text.lower() in m["arch"].lower()
                    or text.lower() in m["size"].lower()]
        self._populate_download_table(filtered)

    def _scan_local_models(self):
        """Scan common model directories"""
        search_dirs = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / "models",
            Path("/opt/models"),
            Path("./models"),
        ]
        self.local_list.clear()
        self.local_models = {}
        found = 0
        for d in search_dirs:
            if d.exists():
                for item in d.rglob("config.json"):
                    model_dir = item.parent
                    name = model_dir.name
                    size = self._dir_size(model_dir)
                    entry = f"{name}  ({size})"
                    self.local_models[entry] = str(model_dir)
                    self.local_list.addItem(entry)
                    found += 1

        if found:
            self.local_status.setText(f"Found {found} local model(s)")
        else:
            self.local_status.setText("No models found. Add a folder or download a model.")

    def _dir_size(self, path):
        try:
            total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            if total > 1e9: return f"{total/1e9:.1f} GB"
            if total > 1e6: return f"{total/1e6:.0f} MB"
            return f"{total/1e3:.0f} KB"
        except:
            return "?"

    def _filter_local(self, text):
        for i in range(self.local_list.count()):
            item = self.local_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def _add_model_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Model Folder")
        if d:
            name = Path(d).name
            entry = f"📁 {name}  (custom)"
            self.local_models[entry] = d
            self.local_list.addItem(entry)
            self.local_status.setText(f"Added: {d}")

    def _use_model(self, item):
        if item:
            path = self.local_models.get(item.text())
            if path:
                self.model_selected.emit(path)

    def _show_model_info(self):
        item = self.local_list.currentItem()
        if not item: return
        path = self.local_models.get(item.text(), "")
        config_path = Path(path) / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                info = json.dumps(cfg, indent=2)
                dlg = QDialog(self)
                dlg.setWindowTitle(f"Config: {Path(path).name}")
                dlg.resize(500, 400)
                lay = QVBoxLayout(dlg)
                te = QPlainTextEdit(info)
                te.setReadOnly(True)
                te.setStyleSheet(f"background:{COLORS['bg3']}; color:{COLORS['text']}; font-family: monospace;")
                lay.addWidget(te)
                dlg.exec_()
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))

    def _browse_dl_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Download Directory")
        if d:
            self.dl_dir_edit.setText(d)

    def _start_download(self):
        row = self.dl_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No selection", "Please select a model to download.")
            return
        dl_dir = self.dl_dir_edit.text() or str(Path.home() / "models")
        model_name = self.dl_table.item(row, 0).text()
        model_hf = next((m["hf"] for m in KNOWN_MODELS if m["name"] == model_name), None)
        if not model_hf:
            QMessageBox.warning(self, "Error", "Cannot find HuggingFace ID for this model.")
            return

        msg = (f"Download {model_name}?\n\n"
               f"Hugging Face: {model_hf}\n"
               f"Save to: {dl_dir}\n\n"
               f"This requires huggingface-cli installed.")
        if QMessageBox.question(self, "Confirm Download", msg) == QMessageBox.Yes:
            self.dl_status.setText(f"Downloading {model_name}...")
            self.dl_progress.setVisible(True)
            self.dl_progress.setRange(0, 0)  # indeterminate
            # Run in background
            t = threading.Thread(target=self._run_download, args=(model_hf, dl_dir, model_name), daemon=True)
            t.start()

    def _run_download(self, hf_id, dl_dir, name):
        try:
            cmd = ["huggingface-cli", "download", hf_id, "--local-dir", dl_dir]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                QMetaObject.invokeMethod(self, "_download_done",
                    Qt.QueuedConnection, Q_ARG(str, f"✅ Downloaded {name} to {dl_dir}"))
            else:
                QMetaObject.invokeMethod(self, "_download_done",
                    Qt.QueuedConnection, Q_ARG(str, f"❌ Error: {result.stderr[:200]}"))
        except FileNotFoundError:
            QMetaObject.invokeMethod(self, "_download_done",
                Qt.QueuedConnection, Q_ARG(str, "❌ huggingface-cli not found. Install: pip install huggingface_hub"))

    @pyqtSlot(str)
    def _download_done(self, msg):
        self.dl_progress.setVisible(False)
        self.dl_status.setText(msg)
        self._scan_local_models()

# ─── Chat / Inference Widget ─────────────────────────────────────────────────
class ChatWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.worker      = None
        self.model_dir   = None
        self.history     = []
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── Left panel: chat ──────────────────────────────────────────────────
        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.setSpacing(8)
        lv.setContentsMargins(16, 16, 16, 16)

        # Model selector bar
        model_bar = QHBoxLayout()
        self.model_label = QLabel("📂 No model loaded")
        self.model_label.setStyleSheet(f"color: {COLORS['yellow']}; font-weight: 600;")
        model_bar.addWidget(self.model_label)
        model_bar.addStretch()

        clear_btn = QPushButton("🗑 Clear Chat")
        clear_btn.setFixedWidth(110)
        clear_btn.clicked.connect(self._clear_chat)
        model_bar.addWidget(clear_btn)
        lv.addLayout(model_bar)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            f"background: {COLORS['bg2']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 8px; padding: 12px; font-size: 13px; line-height: 1.5;")
        lv.addWidget(self.chat_display)

        # Stats bar
        self.stats_label = QLabel("Tokens: 0  |  Speed: — tok/s  |  Context: 0")
        self.stats_label.setStyleSheet(f"color: {COLORS['subtext']}; font-size: 11px;")
        lv.addWidget(self.stats_label)

        # Input area
        input_row = QHBoxLayout()
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("Type your message here... (Ctrl+Enter to send)")
        self.input_edit.setFixedHeight(80)
        self.input_edit.setStyleSheet(
            f"background: {COLORS['bg3']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 8px; padding: 10px; font-size: 13px;")
        input_row.addWidget(self.input_edit)

        send_col = QVBoxLayout()
        self.send_btn = QPushButton("▶ Send")
        self.send_btn.setFixedWidth(90)
        self.send_btn.setFixedHeight(38)
        self.send_btn.setStyleSheet(f"background: {COLORS['green']}; color: #1a1b26; font-weight: bold; font-size: 14px;")
        self.send_btn.clicked.connect(self._send)
        send_col.addWidget(self.send_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setFixedWidth(90)
        self.stop_btn.setFixedHeight(38)
        self.stop_btn.setStyleSheet(f"background: {COLORS['red']};")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_generation)
        send_col.addWidget(self.stop_btn)
        input_row.addLayout(send_col)
        lv.addLayout(input_row)
        layout.addWidget(left, 3)

        # ── Right panel: settings ─────────────────────────────────────────────
        right = QWidget()
        right.setFixedWidth(260)
        right.setStyleSheet(f"background: {COLORS['bg2']}; border-left: 1px solid {COLORS['border']};")
        rv = QVBoxLayout(right)
        rv.setSpacing(10)
        rv.setContentsMargins(16, 16, 16, 16)

        rv.addWidget(QLabel("⚙️ Generation Settings", styleSheet=f"font-weight:bold; color:{COLORS['accent']};"))

        # Max tokens
        rv.addWidget(QLabel("Max Tokens"))
        self.tokens_spin = QSpinBox()
        self.tokens_spin.setRange(1, 32768)
        self.tokens_spin.setValue(512)
        rv.addWidget(self.tokens_spin)

        # Temperature
        rv.addWidget(QLabel("Temperature"))
        temp_row = QHBoxLayout()
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 200)
        self.temp_slider.setValue(70)
        self.temp_label  = QLabel("0.70")
        self.temp_slider.valueChanged.connect(lambda v: self.temp_label.setText(f"{v/100:.2f}"))
        temp_row.addWidget(self.temp_slider)
        temp_row.addWidget(self.temp_label)
        rv.addLayout(temp_row)

        # Top-p
        rv.addWidget(QLabel("Top-P"))
        topp_row = QHBoxLayout()
        self.topp_slider = QSlider(Qt.Horizontal)
        self.topp_slider.setRange(0, 100)
        self.topp_slider.setValue(90)
        self.topp_label  = QLabel("0.90")
        self.topp_slider.valueChanged.connect(lambda v: self.topp_label.setText(f"{v/100:.2f}"))
        topp_row.addWidget(self.topp_slider)
        topp_row.addWidget(self.topp_label)
        rv.addLayout(topp_row)

        # Backend
        rv.addWidget(QLabel("Backend"))
        self.backend_combo = QComboBox()
        backends = ["auto", "cpu"]
        if IS_APPLE_SILICON: backends.append("metal")
        backends += ["opencl", "cuda"]
        self.backend_combo.addItems(backends)
        rv.addWidget(self.backend_combo)

        # Threads
        rv.addWidget(QLabel("CPU Threads"))
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, os.cpu_count() or 8)
        self.threads_spin.setValue(0)
        self.threads_spin.setSpecialValueText("Auto")
        rv.addWidget(self.threads_spin)

        # Options
        self.check_2bit    = QCheckBox("2-bit experts (faster, lower quality)")
        self.check_verbose = QCheckBox("Verbose output")
        self.check_timing  = QCheckBox("Show layer timing")
        rv.addWidget(self.check_2bit)
        rv.addWidget(self.check_verbose)
        rv.addWidget(self.check_timing)

        rv.addStretch()

        # System info
        rv.addWidget(QLabel("System", styleSheet=f"font-weight:bold; color:{COLORS['subtext']};"))
        sys_info = QLabel(
            f"OS: {PLATFORM}\n"
            f"Arch: {ARCH}\n"
            f"Cores: {os.cpu_count()}\n"
            f"Metal: {'✅' if IS_APPLE_SILICON else '❌'}"
        )
        sys_info.setStyleSheet(f"color: {COLORS['subtext']}; font-size: 11px; font-family: monospace;")
        rv.addWidget(sys_info)

        layout.addWidget(right)

    def set_model(self, model_dir):
        self.model_dir = model_dir
        name = Path(model_dir).name
        self.model_label.setText(f"✅ {name}")
        self.model_label.setStyleSheet(f"color: {COLORS['green']}; font-weight: 600;")
        self._append_system(f"Model loaded: {model_dir}")

    def _send(self):
        if not self.model_dir:
            QMessageBox.warning(self, "No Model", "Please select a model in the Model Manager tab first.")
            return

        text = self.input_edit.toPlainText().strip()
        if not text:
            return

        self.input_edit.clear()
        self._append_user(text)

        settings = {
            "max_tokens":  self.tokens_spin.value(),
            "temperature": self.temp_slider.value() / 100.0,
            "top_p":       self.topp_slider.value() / 100.0,
            "backend":     self.backend_combo.currentText(),
            "threads":     self.threads_spin.value() if self.threads_spin.value() > 0 else None,
            "use_2bit":    self.check_2bit.isChecked(),
            "verbose":     self.check_verbose.isChecked(),
        }

        self._append_assistant_start()
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.worker = InferenceWorker(FLASH_MOE_BIN, self.model_dir, text, settings)
        self.worker.token_received.connect(self._on_token)
        self.worker.stats_updated.connect(self._on_stats)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _stop_generation(self):
        if self.worker:
            self.worker.stop()

    def _on_token(self, text):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _on_stats(self, stats):
        self.stats_label.setText(
            f"Tokens: {stats.get('tokens', 0)}  |  "
            f"Speed: {stats.get('tps', 0):.1f} tok/s  |  "
            f"Context: {stats.get('context', 0)}")

    def _on_error(self, err):
        self._append_system(f"❌ Error: {err}")
        self._on_finished()

    def _on_finished(self):
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText("\n\n")
        self.chat_display.setTextCursor(cursor)

    def _append_user(self, text):
        html = (f'<div style="margin:8px 0; padding:10px 14px; '
                f'background:{COLORS["bg3"]}; border-radius:8px; '
                f'border-left: 3px solid {COLORS["accent"]};">'
                f'<b style="color:{COLORS["accent"]}">You</b><br>{text}</div>')
        self.chat_display.append(html)
        self.chat_display.append("")

    def _append_assistant_start(self):
        html = (f'<div style="margin:8px 0; padding:10px 14px; '
                f'background:{COLORS["bg2"]}; border-radius:8px; '
                f'border-left: 3px solid {COLORS["accent2"]};">'
                f'<b style="color:{COLORS["accent2"]}">Assistant</b><br>')
        self.chat_display.append(html)

    def _append_system(self, text):
        html = f'<div style="color:{COLORS["subtext"]}; font-size:11px; margin:4px 0;">• {text}</div>'
        self.chat_display.append(html)

    def _clear_chat(self):
        self.chat_display.clear()
        self.history.clear()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Return and e.modifiers() == Qt.ControlModifier:
            self._send()
        super().keyPressEvent(e)

# ─── Settings Widget ──────────────────────────────────────────────────────────
class SettingsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        layout.addWidget(QLabel("⚙️ Application Settings",
            styleSheet=f"font-size:18px; font-weight:bold; color:{COLORS['accent']};"))

        # Build settings group
        build_grp = QGroupBox("Build & Backend")
        bv = QVBoxLayout(build_grp)

        # Binary path
        bin_row = QHBoxLayout()
        bin_row.addWidget(QLabel("flash_moe binary:"))
        self.bin_edit = QLineEdit(FLASH_MOE_BIN or "")
        self.bin_edit.setPlaceholderText("Path to flash_moe executable...")
        bin_row.addWidget(self.bin_edit)
        browse = QPushButton("📂")
        browse.setFixedWidth(36)
        browse.clicked.connect(self._browse_binary)
        bin_row.addWidget(browse)
        bv.addLayout(bin_row)

        # Build button
        build_btn = QPushButton("🔨 Build from Source")
        build_btn.clicked.connect(self._build_project)
        bv.addWidget(build_btn)

        self.build_output = QPlainTextEdit()
        self.build_output.setReadOnly(True)
        self.build_output.setFixedHeight(120)
        self.build_output.setStyleSheet(f"background:{COLORS['bg3']}; color:{COLORS['text']}; font-family:monospace; font-size:11px;")
        bv.addWidget(self.build_output)
        layout.addWidget(build_grp)

        # System info group
        sys_grp = QGroupBox("System Information")
        sv = QVBoxLayout(sys_grp)
        import psutil
        try:
            ram = psutil.virtual_memory()
            ram_info = f"{ram.total / 1e9:.1f} GB total, {ram.available / 1e9:.1f} GB available"
        except ImportError:
            ram_info = "install psutil for RAM info"

        sys_text = (
            f"Operating System : {platform.platform()}\n"
            f"Architecture     : {ARCH}\n"
            f"Python           : {sys.version.split()[0]}\n"
            f"CPU Cores        : {os.cpu_count()}\n"
            f"RAM              : {ram_info}\n"
            f"Apple Silicon    : {'Yes' if IS_APPLE_SILICON else 'No'}\n"
            f"Metal Available  : {'Yes' if IS_APPLE_SILICON else 'No'}\n"
            f"flash_moe Binary : {FLASH_MOE_BIN or 'NOT FOUND'}"
        )
        sys_lbl = QLabel(sys_text)
        sys_lbl.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {COLORS['text']};")
        sv.addWidget(sys_lbl)
        layout.addWidget(sys_grp)

        # About
        about_grp = QGroupBox("About")
        av = QVBoxLayout(about_grp)
        about_text = (
            "<b>Flash-MoE Universal</b> v1.0.0<br>"
            "Cross-platform AI inference engine<br>"
            "Based on <a href='https://github.com/danveloper/flash-moe'>flash-moe</a> by Daniel Woods<br><br>"
            "Supports: Intel x86_64 · AMD64 · ARM64 (NEON) · Apple Silicon (Metal)<br>"
            "Backends: CPU (AVX2/NEON) · Metal · OpenCL · CUDA<br>"
            "Models: Dense (1M–72B) · MoE (Mixtral, Qwen3.5-397B)"
        )
        about = QLabel(about_text)
        about.setOpenExternalLinks(True)
        about.setWordWrap(True)
        about.setStyleSheet(f"color: {COLORS['text']}; font-size: 12px; line-height: 1.6;")
        av.addWidget(about)
        layout.addWidget(about_grp)

        layout.addStretch()

    def _browse_binary(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select flash_moe binary",
                                           filter="Executable (flash_moe flash_moe.exe *);;All (*)")
        if f:
            self.bin_edit.setText(f)
            global FLASH_MOE_BIN
            FLASH_MOE_BIN = f

    def _build_project(self):
        src_dir = str(Path(__file__).parent.parent)
        build_dir = str(Path(src_dir) / "build")
        self.build_output.clear()
        self.build_output.appendPlainText(f"Building in {build_dir}...")

        def run_build():
            os.makedirs(build_dir, exist_ok=True)
            cmds = [
                ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
                ["cmake", "--build", ".", "--", "-j", str(os.cpu_count() or 4)],
            ]
            for cmd in cmds:
                result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
                QMetaObject.invokeMethod(self.build_output, "appendPlainText",
                    Qt.QueuedConnection, Q_ARG(str, result.stdout + result.stderr))
                if result.returncode != 0:
                    QMetaObject.invokeMethod(self.build_output, "appendPlainText",
                        Qt.QueuedConnection, Q_ARG(str, "❌ Build failed!"))
                    return
            QMetaObject.invokeMethod(self.build_output, "appendPlainText",
                Qt.QueuedConnection, Q_ARG(str, "✅ Build successful!"))

        threading.Thread(target=run_build, daemon=True).start()

# ─── Main Window ─────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flash-MoE Universal")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Title bar
        title_bar = QWidget()
        title_bar.setStyleSheet(f"background: {COLORS['bg2']}; border-bottom: 1px solid {COLORS['border']};")
        title_bar.setFixedHeight(52)
        tb = QHBoxLayout(title_bar)
        tb.setContentsMargins(16, 0, 16, 0)

        icon_label = QLabel("⚡")
        icon_label.setStyleSheet("font-size: 22px;")
        tb.addWidget(icon_label)

        title_label = QLabel("Flash-MoE Universal")
        title_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['accent']};")
        tb.addWidget(title_label)

        tb.addSpacing(16)
        subtitle = QLabel("Cross-platform · Intel x86 · ARM · Apple Silicon · All OS")
        subtitle.setStyleSheet(f"color: {COLORS['subtext']}; font-size: 11px;")
        tb.addWidget(subtitle)
        tb.addStretch()

        # Binary status
        if FLASH_MOE_BIN:
            status = QLabel(f"✅ {Path(FLASH_MOE_BIN).name}")
            status.setStyleSheet(f"color: {COLORS['green']}; font-size: 11px;")
        else:
            status = QLabel("⚠️ Binary not built — go to Settings → Build")
            status.setStyleSheet(f"color: {COLORS['yellow']}; font-size: 11px;")
        tb.addWidget(status)

        layout.addWidget(title_bar)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.West)
        self.tabs.setStyleSheet(f"""
            QTabBar::tab {{
                padding: 14px 8px;
                min-height: 60px;
                font-size: 12px;
                writing-mode: horizontal-tb;
            }}
        """)

        self.chat_widget    = ChatWidget()
        self.model_manager  = ModelManagerWidget()
        self.settings_widget= SettingsWidget()

        self.model_manager.model_selected.connect(self.chat_widget.set_model)

        self.tabs.addTab(self.chat_widget,    "💬\nChat")
        self.tabs.addTab(self.model_manager,  "🤖\nModels")
        self.tabs.addTab(self.settings_widget,"⚙️\nSettings")

        layout.addWidget(self.tabs)

    def closeEvent(self, event):
        if self.chat_widget.worker and self.chat_widget.worker.isRunning():
            self.chat_widget.worker.stop()
            self.chat_widget.worker.wait(2000)
        event.accept()

# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Flash-MoE Universal")
    app.setApplicationVersion("1.0.0")
    app.setStyleSheet(STYLESHEET)

    # High-DPI support
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
