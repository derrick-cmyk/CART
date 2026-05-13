import sys, os, shutil, tempfile, subprocess, re, threading, traceback, json, runpy, multiprocessing, time
import numpy as np
from PIL import Image

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QLabel, QSpinBox,
    QComboBox, QFrame, QSplitter, QAbstractItemView, QListView, QTabWidget, QButtonGroup,
    QFileDialog, QMessageBox, QTextEdit, QProgressBar, QSlider, QStackedWidget
)
from PySide6.QtCore import Qt, QSize, Signal, QMimeData, QPoint, QThread, QSettings
from PySide6.QtGui import QIcon, QPixmap, QColor, QDrag, QBrush, QFont, QPainter, QKeySequence

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

THUMB_SIZE = 100
MIN_THUMB = 50
MAX_THUMB = 200

# Bright colour for inbetweener (non‑GT) frames
INBETWEEN_BG = QColor(0, 191, 255)   # bright blue
# Light colour for GT (checked) frames
GT_BG = QColor(173, 216, 230)       # light blue


# -------------------------------------------------------------------
#  ReorderListWidget (supports drag‑n‑drop reordering with visual placeholder,
#  multi‑selection dragging and auto‑scroll)
# -------------------------------------------------------------------
class ReorderListWidget(QListWidget):
    """Shows a coloured placeholder that moves items out of the way while dragging."""
    reordered = Signal()
    delete_pressed = Signal()
    toggle_reference_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDropIndicatorShown(False)
        self.setSelectionRectVisible(True)
        self._placeholder_item = None
        self._drag_source_row = -1
        self._drag_path = None
        self._drag_paths = []          # for multi‑item drag

    def startDrag(self, supportedActions):
        items = self.selectedItems()
        if not items or self._placeholder_item in items:
            return

        # Multi‑selection drag: gather all selected paths
        if len(items) > 1:
            paths = [it.data(Qt.UserRole) for it in items if it.data(Qt.UserRole) != "__placeholder__"]
            mime = QMimeData()
            mime.setData("application/x-frame-path-list", json.dumps(paths).encode())
            pixmap = QPixmap(THUMB_SIZE, THUMB_SIZE)
            pixmap.fill(QColor(0, 0, 0, 0))
            # QOL: Nicer badge for multi-selection dragging
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QColor(30, 144, 255, 220)) # Dodger Blue
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(pixmap.rect())
            painter.setPen(Qt.white)
            font = QFont()
            font.setBold(True)
            font.setPointSize(14)
            painter.setFont(font)
            painter.drawText(pixmap.rect(), Qt.AlignCenter, f"{len(paths)}")
            painter.end()
            drag = QDrag(self)
            drag.setMimeData(mime)
            drag.setPixmap(pixmap)
            drag.setHotSpot(QPoint(THUMB_SIZE//2, THUMB_SIZE//2))
            drag.exec(Qt.MoveAction)
            return

        # Single item drag
        item = items[0]
        if item == self._placeholder_item:
            return
        self._drag_source_row = self.row(item)
        self._drag_path = item.data(Qt.UserRole)
        mime = QMimeData()
        mime.setData("application/x-frame-path", self._drag_path.encode())
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(item.icon().pixmap(THUMB_SIZE, THUMB_SIZE))
        drag.setHotSpot(QPoint(THUMB_SIZE // 2, THUMB_SIZE // 2))
        drag.exec(Qt.MoveAction)

    def dragEnterEvent(self, event):
        if (event.mimeData().hasFormat("application/x-frame-path") or
            event.mimeData().hasFormat("application/x-frame-path-list")):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if not (event.mimeData().hasFormat("application/x-frame-path") or
                event.mimeData().hasFormat("application/x-frame-path-list")):
            event.ignore()
            return
        self._remove_placeholder()
        pos = event.position().toPoint()
        target_item = self.itemAt(pos)
        indicator = self.dropIndicatorPosition()
        if target_item:
            target_row = self.row(target_item)
            if indicator == QAbstractItemView.AboveItem:
                insert_row = target_row
            elif indicator == QAbstractItemView.BelowItem:
                insert_row = target_row + 1
            else:
                insert_row = target_row + 1
        else:
            insert_row = self.count()

        placeholder = QListWidgetItem()
        placeholder.setData(Qt.UserRole, "__placeholder__")
        placeholder.setFlags(Qt.ItemIsEnabled)                     # enabled, not selectable
        placeholder.setSizeHint(self.gridSize())                   # full cell size
        # Light‑blue background + dark‑blue arrow for clarity
        placeholder.setBackground(QColor(173, 216, 230))          # light blue
        placeholder.setForeground(QColor(0, 70, 200))             # dark blue arrow
        placeholder.setTextAlignment(Qt.AlignCenter)
        placeholder.setText("▼")
        font = placeholder.font()
        font.setBold(True)
        font.setPointSize(16)
        placeholder.setFont(font)

        self.insertItem(insert_row, placeholder)
        self._placeholder_item = placeholder
        event.acceptProposedAction()

        # ---------- auto‑scroll ----------
        viewport = self.viewport()
        margin = 30

        if self.isWrapping():
            # Vertical auto-scroll for Tile View (Grid)
            scrollbar = self.verticalScrollBar()
            step = max(1, scrollbar.singleStep())
            if pos.y() < margin:
                new_val = max(scrollbar.minimum(), scrollbar.value() - step)
                scrollbar.setValue(new_val)
            elif pos.y() > viewport.height() - margin:
                new_val = min(scrollbar.maximum(), scrollbar.value() + step)
                scrollbar.setValue(new_val)
        else:
            # Horizontal auto-scroll for Player View (Single Row)
            scrollbar = self.horizontalScrollBar()
            step = max(1, scrollbar.singleStep())
            if pos.x() < margin:
                new_val = max(scrollbar.minimum(), scrollbar.value() - step)
                scrollbar.setValue(new_val)
            elif pos.x() > viewport.width() - margin:
                new_val = min(scrollbar.maximum(), scrollbar.value() + step)
                scrollbar.setValue(new_val)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.delete_pressed.emit()
            return
        if event.key() == Qt.Key_Space or event.key() == Qt.Key_R:
            self.toggle_reference_requested.emit()
            return
        super().keyPressEvent(event)

    def dragLeaveEvent(self, event):
        self._remove_placeholder()
        event.accept()

    def dropEvent(self, event):
        self._remove_placeholder()

        # Handle multi‑item drop
        if event.mimeData().hasFormat("application/x-frame-path-list"):
            paths = json.loads(bytes(event.mimeData().data("application/x-frame-path-list")).decode())
            rows = []
            for i in range(self.count()):
                item = self.item(i)
                if item.data(Qt.UserRole) in paths:
                    rows.append(i)
            if not rows:
                event.ignore()
                return
            rows.sort()
            pos = event.position().toPoint()
            target_item = self.itemAt(pos)
            indicator = self.dropIndicatorPosition()
            if target_item:
                target_row = self.row(target_item)
                if indicator == QAbstractItemView.AboveItem:
                    insert_row = target_row
                elif indicator == QAbstractItemView.BelowItem:
                    insert_row = target_row + 1
                else:
                    insert_row = target_row + 1
            else:
                insert_row = self.count()

            taken = []
            for r in reversed(rows):
                taken.append(self.takeItem(r))
            taken.reverse()

            removed_before = sum(1 for r in rows if r < insert_row)
            insert_row -= removed_before

            for item in taken:
                self.insertItem(insert_row, item)
                insert_row += 1

            self.clearSelection()
            for item in taken:
                item.setSelected(True)

            event.acceptProposedAction()
            self.reordered.emit()
            return

        # Single item drop
        if not event.mimeData().hasFormat("application/x-frame-path"):
            event.ignore()
            return
        path = bytes(event.mimeData().data("application/x-frame-path")).decode()
        pos = event.position().toPoint()
        target_item = self.itemAt(pos)
        indicator = self.dropIndicatorPosition()
        if target_item:
            target_row = self.row(target_item)
            if indicator == QAbstractItemView.AboveItem:
                insert_row = target_row
            elif indicator == QAbstractItemView.BelowItem:
                insert_row = target_row + 1
            else:
                insert_row = target_row + 1
        else:
            insert_row = self.count()
        source_row = -1
        for i in range(self.count()):
            if self.item(i).data(Qt.UserRole) == path:
                source_row = i
                break
        if source_row == -1:
            event.ignore()
            return
        if source_row < insert_row:
            insert_row -= 1
        item = self.takeItem(source_row)
        self.insertItem(insert_row, item)
        self.setCurrentItem(item)
        event.acceptProposedAction()
        self.reordered.emit()

    def _remove_placeholder(self):
        if self._placeholder_item:
            row = self.row(self._placeholder_item)
            if row >= 0:
                self.takeItem(row)
            self._placeholder_item = None


# -------------------------------------------------------------------
#  LandingPage (Introduction for new users)
# -------------------------------------------------------------------
class LandingPage(QFrame):
    """An introductory page shown when no frames are loaded."""
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LandingPage")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        welcome = QLabel("Welcome to CART")
        welcome.setStyleSheet("font-size: 32px; font-weight: bold; margin-bottom: 10px;")
        welcome.setAlignment(Qt.AlignCenter)
        layout.addWidget(welcome)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        btn_layout.setAlignment(Qt.AlignCenter)
        self.btn_add_frames = QPushButton("➕ Add Images")
        self.btn_add_folder = QPushButton("➕ Add Folder")
        btn_layout.addWidget(self.btn_add_frames)
        btn_layout.addWidget(self.btn_add_folder)
        layout.addLayout(btn_layout)

        instructions = [
            "Upload your frames using the 'Add' buttons above.",
            "Colorization options on the right panel.",
            "Tick the box on a frame if it's a reference frame.",
            "Reorder frames via drag and drop."
        ]
        
        for text in instructions:
            lbl = QLabel(f"• {text}")
            lbl.setStyleSheet("font-size: 18px;")
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl)

        layout.addSpacing(30)
        hint = QLabel("The workspace will appear as soon as you upload frames.")
        hint.setStyleSheet("font-style: italic; color: #888;")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


# -------------------------------------------------------------------
#  HelpPage (Detailed User Guide)
# -------------------------------------------------------------------
class HelpPage(QFrame):
    """A detailed documentation page explaining application usage and settings."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(40)

        # Left Panel: Workflow
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)
        
        lbl_flow = QLabel("How to Operate")
        lbl_flow.setStyleSheet("font-size: 22px; font-weight: bold; margin-bottom: 15px; color: #1e90ff;")
        left_layout.addWidget(lbl_flow)
        
        flow_text = (
            "1. <b>Add Frames</b>: Use 'Add Images' or 'Folder' to import your B&W line art.<br><br>"
            "2. <b>Designate References</b>: Find your colored keyframes in the list and <b>check the checkbox</b> "
            "on their thumbnail. These frames act as the 'Ground Truth' color source.<br><br>"
            "3. <b>Organize Timeline</b>: Drag and drop frames to ensure they are in chronological order. "
            "Proper sequence is vital for the AI to track motion accurately.<br><br>"
            "4. <b>Run</b>: Click 'Run Colorization'. The AI will propagate colors from your references to the inbetweens."
        )
        flow_lbl = QLabel(flow_text)
        flow_lbl.setWordWrap(True)
        flow_lbl.setStyleSheet("font-size: 15px; line-height: 1.5;")
        left_layout.addWidget(flow_lbl)
        layout.addWidget(left_panel, 1)

        # Right Panel: Component Guide
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setAlignment(Qt.AlignTop)
        
        lbl_comp = QLabel("Colorization Settings Guide")
        lbl_comp.setStyleSheet("font-size: 22px; font-weight: bold; margin-bottom: 15px; color: #1e90ff;")
        right_layout.addWidget(lbl_comp)
        
        comp_text = (
            "• <b>Mode</b>: <i>Auto</i> is recommended; it finds the shortest path between references. <i>Forward/Backward</i> forces direction. Nearest applies references directly to all frames.<br>"
            "• <b>Seg Type</b>: Use <i>Default</i> for clean lines. Use <i>Trappedball</i> for messy or unclosed sketches.<br>"
            "• <b>Line-mask Thr</b>: Controls line art detection. Lower values make the AI more sensitive to light gray strokes.<br>"
            "• <b>Force White Canvas</b>: Essential if using transparent PNGs to prevent black silhouette errors.<br>"
            "• <b>Treat as Final</b>: Skips line-cleaning. Use if your input frames already have shading or rough colors.<br>"
            "• <b>RAFT Res</b>: Resolution for motion tracking. 320 is standard; 640+ improves quality for complex movement but is slower.<br>"
            "• <b>Keep Line</b>: Merges your original high-res linework back onto the AI's color output."
        )
        comp_lbl = QLabel(comp_text)
        comp_lbl.setWordWrap(True)
        comp_lbl.setStyleSheet("font-size: 15px; line-height: 1.5;")
        right_layout.addWidget(comp_lbl)
        layout.addWidget(right_panel, 1)

# -------------------------------------------------------------------
#  FrameViewer (Shared component for Timeline and Results)
# -------------------------------------------------------------------
class FrameViewer(QWidget):
    """A widget that can toggle between a Tile View (grid) and Player View (one-at-a-time)."""
    
    def __init__(self, is_editable=True, parent=None):
        super().__init__(parent)
        self.is_editable = is_editable
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        self.title_label = QLabel("Frames")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header.addWidget(self.title_label)
        header.addStretch()
        
        self.btn_tile_view = QPushButton("Tile View")
        self.btn_tile_view.setCheckable(True)
        self.btn_tile_view.setChecked(True)
        self.btn_player_view = QPushButton("Player View")
        self.btn_player_view.setCheckable(True)
        
        self.view_btn_group = QButtonGroup(self)
        self.view_btn_group.addButton(self.btn_tile_view, 0)
        self.view_btn_group.addButton(self.btn_player_view, 1)
        
        header.addWidget(self.btn_tile_view)
        header.addWidget(self.btn_player_view)
        self.layout.addLayout(header)

        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack)

        # Tile View
        self.tile_page = QWidget()
        self.tile_layout = QVBoxLayout(self.tile_page)
        self.tile_layout.setContentsMargins(0, 0, 0, 0)

        self.list_widget = ReorderListWidget()
        self.list_widget.setViewMode(QListView.IconMode)
        self.list_widget.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.list_widget.setResizeMode(QListView.Adjust)
        self.list_widget.setGridSize(QSize(THUMB_SIZE + 24, THUMB_SIZE + 60))
        
        self.tile_layout.addWidget(self.list_widget)
        self.stack.addWidget(self.tile_page)

        # Player View
        player_page = QWidget()
        p_layout = QVBoxLayout(player_page)
        p_layout.setContentsMargins(0, 0, 0, 0)
        
        self.preview_label = QLabel("No frame selected")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self._player_bg_color = "#1a1a1a"   # default dark background
        self._update_player_bg()
        
        p_layout.addWidget(self.preview_label, 1)

        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setMinimum(0)
        p_layout.addWidget(self.scrubber)
        
        self.player_container = QVBoxLayout()
        p_layout.addLayout(self.player_container)
        
        self.stack.addWidget(player_page)

        self.btn_tile_view.clicked.connect(lambda: self.set_view_mode(0))
        self.btn_player_view.clicked.connect(lambda: self.set_view_mode(1))
        self.list_widget.currentRowChanged.connect(self._sync_preview)
        self.scrubber.valueChanged.connect(self.list_widget.setCurrentRow)

    def set_player_bg_color(self, color):
        """Update the background colour of the player view preview label."""
        self._player_bg_color = color
        self._update_player_bg()

    def _update_player_bg(self):
        """Apply the stored background colour to the preview label."""
        self.preview_label.setStyleSheet(
            f"background-color: {self._player_bg_color}; border-radius: 10px;"
        )

    def set_view_mode(self, index):
        self.btn_tile_view.setChecked(index == 0)
        self.btn_player_view.setChecked(index == 1)
        self.stack.setCurrentIndex(index)

        if index == 1:
            # Player View: Force horizontal scrolling
            self.list_widget.setFlow(QListView.LeftToRight)
            self.list_widget.setWrapping(False)
            self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.update_list_height()
            self.player_container.addWidget(self.list_widget)
            self._sync_preview()
            self.list_widget.setFocus()
        else:
            # Tile View: Grid layout
            self.list_widget.setFlow(QListView.LeftToRight)
            self.list_widget.setWrapping(True)
            self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.list_widget.setFixedHeight(16777215)
            self.stack.widget(0).layout().addWidget(self.list_widget)

    def update_list_height(self):
        """Adjusts the list height when in Player View to match thumbnail size."""
        if self.stack.currentIndex() == 1:
            # Add padding for text and scrollbar
            self.list_widget.setFixedHeight(THUMB_SIZE + 90)

    def _sync_preview(self):
        item = self.list_widget.currentItem()
        self.scrubber.setMaximum(max(0, self.list_widget.count() - 1))
        self.scrubber.setValue(self.list_widget.currentRow())
        if item and item.data(Qt.UserRole):
            pix = QPixmap(item.data(Qt.UserRole))
            self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


# -------------------------------------------------------------------
#  Worker thread for inference
# -------------------------------------------------------------------
class InferenceWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(str)
    progress_start_signal = Signal()
    progress_stop_signal = Signal()
    finished_signal = Signal(bool, str)          # success, message
    open_output_signal = Signal(str)

    def __init__(self, timeline_items, out_dir, mode, seg_type,
                 keep_line, raft_res, line_thr,
                 force_white_bg, treat_as_final, use_cuda, parent=None):
        super().__init__(parent)
        self.timeline_items = timeline_items
        self.out_dir = out_dir
        self.mode = mode
        self.seg_type = seg_type
        self.keep_line = keep_line
        self.raft_res = raft_res
        self.line_thr = line_thr
        self.force_white_bg = force_white_bg
        self.treat_as_final = treat_as_final
        self.use_cuda = use_cuda
        self.proc = None
        self._stopped = False

    def stop(self):
        """Stops the inference by setting a flag and terminating the subprocess."""
        self._stopped = True
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def run(self):
        self.progress_start_signal.emit()
        temp_workspace = os.path.join(BASE_DIR, "_gui_workspace")
        clip_name = "temp_clip"
        temp_clip = os.path.join(temp_workspace, clip_name)
        start_time = time.time()
        
        try:
            # ── 1. Setup temp clip folder ──
            self.log_signal.emit("[1/4] Setting up temporary clip folder…")
            if self._stopped: return

            if os.path.exists(temp_workspace):
                shutil.rmtree(temp_workspace)

            gt_dir = os.path.join(temp_clip, "gt")
            line_dir = os.path.join(temp_clip, "line")
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(line_dir, exist_ok=True)

            ref_count = 0
            for idx, item in enumerate(self.timeline_items):
                if self._stopped:
                    self.log_signal.emit("🛑 Process stopped by user during preparation.")
                    self.status_signal.emit("Stopped.")
                    self.finished_signal.emit(False, "Stopped by user")
                    return

                path = item['path']
                is_ref = item['is_ref']
                name = f"{idx:04d}.png"

                img = Image.open(path)
                if self.force_white_bg:
                    rgba = img.convert("RGBA")
                    canvas = Image.new("RGB", rgba.size, (255, 255, 255))
                    canvas.paste(rgba, mask=rgba.split()[3] if img.mode == 'RGBA' else None)
                    img = canvas

                arr = np.array(img.convert("RGB"))
                gray = np.mean(arr, axis=2)
                dst_line = os.path.join(line_dir, name)

                if is_ref:
                    dst_gt = os.path.join(gt_dir, name)
                    img.save(dst_gt)
                    ref_count += 1
                    self.log_signal.emit(f"   REF/LINE {name} ← {os.path.basename(path)}")
                else:
                    self.log_signal.emit(f"   LINE {name} ← {os.path.basename(path)}")

                line_mask = gray < self.line_thr
                if not np.any(line_mask):
                    line_mask[0, 0] = True
                    arr[0, 0] = [0, 0, 0]

                h, w = arr.shape[:2]
                out_line = np.zeros((h, w, 4), dtype=np.uint8)
                out_line[:, :, :3] = 255
                out_line[line_mask, :3] = arr[line_mask]
                out_line[:, :, 3] = np.where(line_mask, 255, 0)
                Image.fromarray(out_line, "RGBA").save(dst_line)

            if self._stopped:
                self.log_signal.emit("🛑 Process stopped by user after preparation.")
                self.status_signal.emit("Stopped.")
                self.finished_signal.emit(False, "Stopped by user")
                return

            self.log_signal.emit(f"   Clip ready: {ref_count} GT frame(s), "
                                 f"{len(self.timeline_items) - ref_count} line frame(s) to colorize ({len(self.timeline_items)} total)\n")

            # ── 2. Build command ──
            self.log_signal.emit("[2/4] Building inference command…")
            self.status_signal.emit("Running inference…")
            script = os.path.join(BASE_DIR, "inference_line_frames.py")
            # Use sys.executable instead of "python" to ensure the bundled environment is used
            cmd = [
                sys.executable, script,
                "--path", temp_clip,
                "--mode", self.mode,
                "--seg_type", self.seg_type,
                "--raft_res", self.raft_res,
                "--line_thr", str(self.line_thr),
            ]
            if self.keep_line:
                cmd.append("--keep_line")
            if self.treat_as_final:
                cmd.append("--treat_as_final")
            if not self.use_cuda: # If "Use CUDA" is toggled OFF, force CPU usage
                cmd.append("--force_cpu")
            self.log_signal.emit(f"   Command:\n   {' '.join(cmd)}\n")

            # ── 3. Run subprocess ──
            self.log_signal.emit("[3/4] Running model – see live output below…\n" + "─" * 60)
            if self._stopped:
                self.log_signal.emit("🛑 Process stopped by user before launching model.")
                self.status_signal.emit("Stopped.")
                self.finished_signal.emit(False, "Stopped by user")
                return

            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=BASE_DIR,
                bufsize=1,
            )

            friendly_error = None
            # Map technical regex patterns to user-friendly hints
            error_patterns = {
                r"torch\.cuda\.OutOfMemoryError": "Graphics Memory (VRAM) is full. Try lower RAFT resolution or CPU mode.",
                r"CUDA error": "Graphics card error. Ensure your drivers are updated or use CPU mode.",
                r"FileNotFoundError.*ckpt": "Missing model weights. Ensure 'ckpt' files are in the right folder.",
                r"ModuleNotFoundError": "Internal library error: AI dependencies not found in this environment.",
                r"TypeError.*line_thr": "Compatibility error with the 'paint' library version."
            }

            for raw_line in iter(self.proc.stdout.readline, ''):
                stripped = raw_line.rstrip('\n\r')
                self.log_signal.emit(stripped)

                # Check for known errors to provide human-readable feedback
                for pattern, hint in error_patterns.items():
                    if re.search(pattern, stripped, re.IGNORECASE):
                        friendly_error = hint

            self.proc.stdout.close()
            ret = self.proc.wait()
            self.log_signal.emit("─" * 60)

            if self._stopped:
                self.log_signal.emit("\n🛑 Process stopped by user.")
                self.status_signal.emit("Stopped.")
                self.finished_signal.emit(False, "Stopped by user")
                return

            if ret != 0:
                err_msg = friendly_error if friendly_error else f"Subprocess failed with code {ret}"
                self.log_signal.emit(f"\n✖  Process failed.")
                if friendly_error:
                    self.log_signal.emit(f"   HINT: {friendly_error}")
                self.status_signal.emit("Failed")
                self.finished_signal.emit(False, err_msg)
                return

            # ── 4. Copy results ──
            self.log_signal.emit("\n[4/4] Copying results to output directory…")
            results_src = temp_clip
            keepline_src = os.path.join(temp_workspace, clip_name + "_keepline")
            os.makedirs(self.out_dir, exist_ok=True)
            copied = 0

            if os.path.isdir(results_src):
                for fname in sorted(os.listdir(results_src)):
                    full = os.path.join(results_src, fname)
                    if os.path.isfile(full) and fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        shutil.copy2(full, os.path.join(self.out_dir, fname))
                        self.log_signal.emit(f"   → {fname}")
                        copied += 1
            else:
                self.log_signal.emit(f"   ⚠ Expected results folder not found: {results_src}")

            if os.path.isdir(keepline_src):
                keepline_out = os.path.join(self.out_dir, "keepline")
                os.makedirs(keepline_out, exist_ok=True)
                for fname in sorted(os.listdir(keepline_src)):
                    full = os.path.join(keepline_src, fname)
                    if os.path.isfile(full) and fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        shutil.copy2(full, os.path.join(keepline_out, fname))
                        self.log_signal.emit(f"   → keepline/{fname}")
                        copied += 1

            end_time = time.time()
            total_sec = end_time - start_time
            avg_sec = total_sec / len(self.timeline_items) if self.timeline_items else 0

            self.log_signal.emit(f"\n⏱  Performance Metrics:")
            self.log_signal.emit(f"   Total Time: {total_sec:.2f}s | Avg: {avg_sec:.2f}s per frame")
            self.log_signal.emit(f"✔  Done! {copied} result image(s) saved to:\n   {self.out_dir}")
            self.status_signal.emit(f"Complete – {copied} images saved.")
            self.finished_signal.emit(True, "")
            self.open_output_signal.emit(self.out_dir)

        except Exception as e:
            self.log_signal.emit(f"\n✖  Exception: {e}")
            self.log_signal.emit(traceback.format_exc())
            self.status_signal.emit("Error.")
            self.finished_signal.emit(False, str(e))
        finally:
            self.progress_stop_signal.emit()
            if os.path.exists(temp_clip):
                try:
                    shutil.rmtree(temp_clip)
                except Exception:
                    pass


# -------------------------------------------------------------------
#  Main Application
# -------------------------------------------------------------------
class ColorizationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BasicPBC – Paint Bucket Colorization")
        self.setWindowIcon(QIcon())
        self.setMinimumSize(1000, 750)

        # Theme state
        self.current_theme = "dark"
        self.show_landing_pref = True
        self.landing_dismissed_manually = False
        self.thumb_size = THUMB_SIZE

        central = QWidget()
        self.setCentralWidget(central)
        main_vbox = QVBoxLayout(central)
        main_vbox.setContentsMargins(10, 10, 10, 10)

        # Toolbar
        toolbar_layout = QHBoxLayout()
        self.btn_add_frames = QPushButton("➕ Add Images")
        self.btn_add_folder = QPushButton("➕ Add Folder")
        self.btn_remove_selected = QPushButton("🗑 Remove Selected")
        self.btn_reverse_order = QPushButton("⇅ Reverse Order")
        self.btn_move_front = QPushButton("▤ Move to Front")
        self.btn_move_back = QPushButton("▤ Move to Back")
        self.btn_clear_all = QPushButton("🗑 Clear All")
        self.btn_run = QPushButton("▶  Run Colorization")
        self.btn_run.setStyleSheet("QPushButton { background-color: #1e90ff; color: white; }")
        self.btn_stop = QPushButton("🛑 Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("QPushButton { background-color: #d9534f; color: white; }")
        toolbar_layout.addWidget(self.btn_add_frames)
        toolbar_layout.addWidget(self.btn_add_folder)
        toolbar_layout.addWidget(self.btn_remove_selected)
        toolbar_layout.addWidget(self.btn_reverse_order)
        toolbar_layout.addWidget(self.btn_move_front)
        toolbar_layout.addWidget(self.btn_move_back)
        toolbar_layout.addWidget(self.btn_clear_all)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.btn_run)
        toolbar_layout.addWidget(self.btn_stop)
        main_vbox.addLayout(toolbar_layout)

        # ---------- Tabs ----------
        # Main Content Splitter: Left (Tabs) | Right (Options)
        self.content_splitter = QSplitter(Qt.Horizontal)
        main_vbox.addWidget(self.content_splitter, 1)

        self.tabs = QTabWidget()
        self.content_splitter.addWidget(self.tabs)

        # --- Workspace tab ---
        workspace_tab = QWidget()
        workspace_layout = QVBoxLayout(workspace_tab)

        # Stack: Landing Page vs. Timeline UI
        self.workspace_stack = QStackedWidget()
        workspace_layout.addWidget(self.workspace_stack)

        self.landing_page = LandingPage()
        self.landing_page.clicked.connect(self.dismiss_landing_page)
        self.landing_page.btn_add_frames.clicked.connect(self.add_image_files)
        self.landing_page.btn_add_folder.clicked.connect(self.add_image_folder)
        self.workspace_stack.addWidget(self.landing_page)

        self.timeline_viewer = FrameViewer(is_editable=True)
        self.timeline_viewer.title_label.setText("Animation Timeline")
        self.timeline_list = self.timeline_viewer.list_widget
        self.workspace_stack.addWidget(self.timeline_viewer)

        # --- Results tab ---
        self.results_tab = QWidget()
        self.results_layout = QVBoxLayout(self.results_tab)
        self.results_viewer = FrameViewer(is_editable=False)
        self.results_viewer.title_label.setText("Colorization Results")
        self.results_list = self.results_viewer.list_widget
        
        self.no_results_label = QLabel("No results to display. Run colorization first.")
        self.no_results_label.setAlignment(Qt.AlignCenter)
        self.results_layout.addWidget(self.no_results_label)
        self.results_layout.addWidget(self.results_viewer)
        self.results_viewer.hide()

        # --- Settings tab ---
        self.settings_tab = QWidget()
        settings_layout = QVBoxLayout(self.settings_tab)
        settings_layout.setContentsMargins(20, 20, 20, 20)
        settings_layout.addStretch()

        # Theme selection buttons
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("Theme:"))
        self.btn_dark_mode = QPushButton("Dark Mode")
        self.btn_dark_mode.setCheckable(True)
        self.btn_light_mode = QPushButton("Light Mode")
        self.btn_light_mode.setCheckable(True)
        self.theme_button_group = QButtonGroup(self)
        self.theme_button_group.addButton(self.btn_dark_mode, 0)
        self.theme_button_group.addButton(self.btn_light_mode, 1)
        self.theme_button_group.setExclusive(True) # Only one can be checked
        theme_layout.addWidget(self.btn_dark_mode)
        theme_layout.addWidget(self.btn_light_mode)
        theme_layout.addStretch()
        settings_layout.addLayout(theme_layout)

        # CUDA toggle
        cuda_layout = QHBoxLayout()
        cuda_layout.addWidget(QLabel("CUDA Usage:"))
        self.btn_use_cuda = QPushButton("Use CUDA (if available)")
        self.btn_use_cuda.setCheckable(True)
        self.btn_use_cuda.setChecked(True) # Default to trying CUDA
        self.btn_use_cuda.setStyleSheet("QPushButton:checked { background-color: #1e90ff; }")
        cuda_layout.addWidget(self.btn_use_cuda)
        cuda_layout.addStretch()
        settings_layout.addLayout(cuda_layout)

        # Landing page toggle
        landing_layout = QHBoxLayout()
        landing_layout.addWidget(QLabel("Show introduction when empty:"))
        self.btn_toggle_landing = QPushButton("Enabled")
        self.btn_toggle_landing.setCheckable(True)
        self.btn_toggle_landing.setChecked(True)
        self.btn_toggle_landing.setStyleSheet("QPushButton:checked { background-color: #1e90ff; }")
        self.btn_toggle_landing.toggled.connect(self._on_landing_toggled)
        landing_layout.addWidget(self.btn_toggle_landing)
        landing_layout.addStretch()
        settings_layout.addLayout(landing_layout)

        settings_layout.addSpacing(20)

        # Reset button
        self.btn_reset_defaults = QPushButton("🔄 Reset to Standard Settings")
        self.btn_reset_defaults.setStyleSheet("QPushButton { background-color: #f0ad4e; color: white; }")
        settings_layout.addWidget(self.btn_reset_defaults)
        settings_layout.addStretch()

        # --- Help Tab ---
        self.help_page = HelpPage()

        self.tabs.addTab(workspace_tab, "Workspace")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.tabs.addTab(self.help_page, "?")

        # ----- Right panel (options + console) -----
        self.right_frame = QFrame()
        self.right_frame.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(self.right_frame)
        right_layout.setContentsMargins(20, 20, 20, 20)

        opt_group = QWidget()
        opt_layout = QVBoxLayout(opt_group)
        opt_layout.setSpacing(8)

        m_layout = QHBoxLayout()
        m_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["auto", "forward", "backward", "nearest"])
        self.mode_combo.setCurrentText("auto")
        m_layout.addWidget(self.mode_combo)
        opt_layout.addLayout(m_layout)

        s_layout = QHBoxLayout()
        s_layout.addWidget(QLabel("Seg Type:"))
        self.seg_combo = QComboBox()
        self.seg_combo.addItems(["default", "trappedball"])
        s_layout.addWidget(self.seg_combo)
        opt_layout.addLayout(s_layout)

        t_layout = QHBoxLayout()
        t_layout.addWidget(QLabel("Line-mask Thr:"))
        self.thr_spin = QSpinBox()
        self.thr_spin.setRange(0, 255)
        self.thr_spin.setValue(50)
        t_layout.addWidget(self.thr_spin)
        opt_layout.addLayout(t_layout)

        self.white_bg_check = QPushButton("Force White Canvas")
        self.white_bg_check.setCheckable(True)
        self.white_bg_check.setStyleSheet("QPushButton:checked { background-color: #1e90ff; }")
        opt_layout.addWidget(self.white_bg_check)

        self.final_line_check = QPushButton("Treat Line Images as Final")
        self.final_line_check.setCheckable(True)
        self.final_line_check.setStyleSheet("QPushButton:checked { background-color: #1e90ff; }")
        opt_layout.addWidget(self.final_line_check)

        self.keepline_check = QPushButton("Keep Line")
        self.keepline_check.setCheckable(True)
        self.keepline_check.setStyleSheet("QPushButton:checked { background-color: #1e90ff; }")
        opt_layout.addWidget(self.keepline_check)

        r_layout = QHBoxLayout()
        r_layout.addWidget(QLabel("RAFT Res:"))
        self.raft_edit = QComboBox()
        self.raft_edit.setEditable(True)
        self.raft_edit.addItems(["320", "480", "640", "800", "960", "1280", "1600"])
        self.raft_edit.setCurrentText("320")
        r_layout.addWidget(self.raft_edit)
        opt_layout.addLayout(r_layout)

        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("Output Dir:"))
        self.out_entry = QComboBox()
        self.out_entry.setEditable(True)
        out_layout.addWidget(self.out_entry)
        btn_browse = QPushButton("Browse")
        out_layout.addWidget(btn_browse)
        opt_layout.addLayout(out_layout)

        right_layout.addWidget(opt_group)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 0)
        self.progress.hide()
        right_layout.addWidget(self.progress)
        self.status_label = QLabel("Ready")
        right_layout.addWidget(self.status_label)

        right_layout.addWidget(QLabel("Console:"))
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #dcdcdc;")
        right_layout.addWidget(self.console)

        self.content_splitter.addWidget(self.right_frame)
        self.content_splitter.setSizes([650, 400])

        # Connections
        self.btn_add_frames.clicked.connect(self.add_image_files)
        self.btn_add_folder.clicked.connect(self.add_image_folder)
        self.btn_remove_selected.clicked.connect(self.remove_selected_frames)
        self.btn_reverse_order.clicked.connect(self.reverse_timeline_order)
        self.btn_move_front.clicked.connect(self.move_selected_to_front)
        self.btn_move_back.clicked.connect(self.move_selected_to_back)
        self.btn_clear_all.clicked.connect(self.clear_timeline)
        self.btn_run.clicked.connect(self.run_colorization)
        self.btn_stop.clicked.connect(self.stop_colorization)
        btn_browse.clicked.connect(self.browse_output)
        self.btn_reset_defaults.clicked.connect(self.reset_to_defaults)

        # Keyboard/Mouse QOL connections
        self.timeline_list.delete_pressed.connect(self.remove_selected_frames)
        self.timeline_list.toggle_reference_requested.connect(self.toggle_selected_references)
        self.timeline_list.itemDoubleClicked.connect(self._on_item_double_clicked)

        # Connect theme buttons
        self.btn_dark_mode.clicked.connect(lambda: self.change_theme("dark"))
        self.btn_light_mode.clicked.connect(lambda: self.change_theme("light"))
        self.timeline_list.reordered.connect(self._update_timeline_display)
        self.timeline_list.itemChanged.connect(self._on_item_changed)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.btn_dark_mode.setChecked(True) # Default to dark mode

        # Connect "Force White Canvas" toggle to update player view background
        self.white_bg_check.toggled.connect(self._on_force_white_bg_toggled)

        self.load_settings()

        # Apply initial theme
        self.apply_theme(self.current_theme)

    def _on_force_white_bg_toggled(self, checked):
        """Change the player view preview background to white when 'Force White Canvas' is on."""
        color = "#FFFFFF" if checked else "#1a1a1a"
        self.timeline_viewer.set_player_bg_color(color)
        self.results_viewer.set_player_bg_color(color)

    def _on_tab_changed(self, index):
        # Show right panel only for Workspace (0) and Help (3)
        self.right_frame.setVisible(index == 0 or index == 3)

    # ---------- Theme management ----------
    def change_theme(self, text):
        theme = text.lower()
        if theme != self.current_theme:
            self.current_theme = theme
            self.btn_dark_mode.setChecked(theme == "dark") # Set checked state
            self.btn_light_mode.setChecked(theme == "light") # Set checked state
            self.apply_theme(theme)
    
    def apply_theme(self, theme):
        if theme == "dark":
            self.setStyleSheet("""
                QMainWindow { background-color: #2b2b2b; }
                QLabel { color: #cccccc; }
                QPushButton {
                    background-color: #3a3a3a;
                    border: none; border-radius: 6px;
                    padding: 8px 15px; color: white;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #4a4a4a; }
                QPushButton:checked { background-color: #1e90ff; color: white; }
                QComboBox, QLineEdit, QTextEdit, QProgressBar, QSpinBox {
                    background-color: #3a3a3a;
                    selection-background-color: #1e90ff; /* For selected text */
                    border: 1px solid #555;
                    border-radius: 4px;
                    padding: 5px;
                    color: white;
                }
                QComboBox QAbstractItemView {
                    background-color: #3a3a3a;
                    color: white;
                }
                QFrame[frameShape="6"] {
                    background-color: #2d2d2d;
                    border-left: 1px solid #555;
                }
                QListWidget {
                    background-color: #333333;
                    border: none;
                }
                QTabWidget::pane { border: 1px solid #555; }
                QTabBar::tab { background: #3a3a3a; color: #ccc; padding: 10px 20px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
                QTabBar::tab:selected { background: #2d2d2d; color: white; border-bottom-color: #1e90ff; border-bottom: 2px solid #1e90ff; }
                QListWidget::item {
                    border-radius: 8px; margin: 4px; padding: 4px;
                    color: white;
                }
                QListWidget::item:selected {
                    background-color: #5a5a5a;
                    border: 2px solid #1e90ff;
                }
                QListWidget::item:hover {
                    background-color: #555555;
                }
                QListWidget::indicator:checked {
                    background-color: #87CEFA;
                    border: 2px solid #00BFFF;
                    border-radius: 3px;
                }
                QListWidget::indicator:unchecked {
                    background-color: #555;
                    border: 2px solid #777;
                    border-radius: 3px;
                }
                #LandingPage {
                    background-color: #333333;
                    border: 2px dashed #555;
                    border-radius: 15px;
                }
                #LandingPage QPushButton {
                    background-color: #444;
                    min-width: 120px;
                    font-size: 14px;
                }
                #LandingPage QPushButton:hover { background-color: #555; }
            """)
        else:  # light theme
            self.setStyleSheet("""
                QMainWindow { background-color: #f0f0f0; }
                QLabel { color: #333333; }
                QPushButton {
                    background-color: #e0e0e0;
                    border: none; border-radius: 6px;
                    padding: 8px 15px; color: black;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #d0d0d0; }
                QPushButton:checked { background-color: #1e90ff; color: white; }
                QComboBox, QLineEdit, QTextEdit, QProgressBar, QSpinBox {
                    background-color: white;
                    selection-background-color: #1e90ff; /* For selected text */
                    border: 1px solid #aaa;
                    border-radius: 4px;
                    padding: 5px;
                    color: black;
                }
                QComboBox QAbstractItemView {
                    background-color: white;
                    color: black;
                }
                QFrame[frameShape="6"] {
                    background-color: #e8e8e8;
                    border-left: 1px solid #aaa;
                }
                QListWidget {
                    background-color: white;
                    border: none;
                }
                QTabWidget::pane { border: 1px solid #aaa; }
                QTabBar::tab { background: #e0e0e0; color: #333; padding: 10px 20px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
                QTabBar::tab:selected { background: #f0f0f0; color: black; border-bottom-color: #1e90ff; border-bottom: 2px solid #1e90ff; }
                QListWidget::item {
                    border-radius: 8px; margin: 4px; padding: 4px;
                    color: black;
                }
                QListWidget::item:selected {
                    background-color: #c0c0c0;
                    border: 2px solid #1e90ff;
                }
                QListWidget::item:hover {
                    background-color: #dcdcdc;
                }
                QListWidget::indicator:checked {
                    background-color: #87CEFA;
                    border: 2px solid #00BFFF;
                    border-radius: 3px;
                }
                QListWidget::indicator:unchecked {
                    background-color: #ccc;
                    border: 2px solid #999;
                    border-radius: 3px;
                }
                #LandingPage {
                    background-color: #ffffff;
                    border: 2px dashed #ccc;
                    border-radius: 15px;
                }
                #LandingPage QPushButton {
                    background-color: #ddd;
                    min-width: 120px;
                    font-size: 14px;
                }
                #LandingPage QPushButton:hover { background-color: #ccc; }
            """)

        # The `btn_run` and `btn_stop` have inline styles, but they will be overridden by global QPushButton style.
        # Reapply their specific styles to ensure they stand out.
        self.btn_run.setStyleSheet("QPushButton { background-color: #1e90ff; color: white; }")
        self.btn_stop.setStyleSheet("QPushButton { background-color: #d9534f; color: white; }")
        self.btn_reset_defaults.setStyleSheet("QPushButton { background-color: #f0ad4e; color: white; }")
        self.btn_use_cuda.setStyleSheet("QPushButton:checked { background-color: #1e90ff; }")

        # Re-apply the player view background in case it was overridden by global style
        self.timeline_viewer._update_player_bg()
        self.results_viewer._update_player_bg()

    def dismiss_landing_page(self):
        self.landing_dismissed_manually = True
        self._update_workspace_view()

    def _on_landing_toggled(self, checked):
        self.show_landing_pref = checked
        self.landing_dismissed_manually = False
        self.btn_toggle_landing.setText("Enabled" if checked else "Disabled")
        self._update_workspace_view()

    def _update_workspace_view(self):
        if self.timeline_list.count() > 0:
            self.workspace_stack.setCurrentIndex(1)
        elif hasattr(self, 'show_landing_pref') and self.show_landing_pref and not self.landing_dismissed_manually:
            self.workspace_stack.setCurrentIndex(0)
        else:
            self.workspace_stack.setCurrentIndex(1)

    # ---------- Zoom (Ctrl + / Ctrl -) ----------
    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_A:
                if self.tabs.currentIndex() == 0:
                    self.timeline_list.selectAll()
                elif self.tabs.currentIndex() == 1:
                    self.results_list.selectAll()
                return
            elif event.key() == Qt.Key_R or event.key() == Qt.Key_Return:
                if self.btn_run.isEnabled():
                    self.run_colorization()
                return

            if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
                self.zoom_in()
                return
            elif event.key() == Qt.Key_Minus:
                self.zoom_out()
                return
        super().keyPressEvent(event)

    def zoom_in(self):
        new_size = min(self.thumb_size + 10, MAX_THUMB)
        if new_size != self.thumb_size:
            self.thumb_size = new_size
            self.update_thumbnail_size()

    def zoom_out(self):
        new_size = max(self.thumb_size - 10, MIN_THUMB)
        if new_size != self.thumb_size:
            self.thumb_size = new_size
            self.update_thumbnail_size()

    def update_thumbnail_size(self):
        global THUMB_SIZE
        THUMB_SIZE = self.thumb_size
        # Update timeline and results lists
        for viewer in [self.timeline_viewer, self.results_viewer]:
            lst = viewer.list_widget
            lst.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
            lst.setGridSize(QSize(THUMB_SIZE + 24, THUMB_SIZE + 60))
            viewer.update_list_height()
            for i in range(lst.count()):
                item = lst.item(i)
                if item and item.data(Qt.UserRole) and item.data(Qt.UserRole) != "__placeholder__":
                    path = item.data(Qt.UserRole)
                    pix = QPixmap(path).scaled(THUMB_SIZE, THUMB_SIZE,
                                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    item.setIcon(QIcon(pix))
                    item.setSizeHint(QSize(THUMB_SIZE + 24, THUMB_SIZE + 60))
        # Also update the placeholder's size hint in dragMoveEvent uses self.gridSize() which will be correct.

    # ---------- Image list management ----------
    def _add_single_image_to_list(self, file_path):
        pix = QPixmap(file_path).scaled(self.thumb_size, self.thumb_size,
                                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
        item = QListWidgetItem()
        item.setIcon(QIcon(pix))
        item.setData(Qt.UserRole, file_path)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        item.setCheckState(Qt.Unchecked)
        item.setSizeHint(QSize(self.thumb_size + 24, self.thumb_size + 60))
        self._apply_item_bg(item)
        self.timeline_list.addItem(item)

    def _apply_item_bg(self, item):
        if item.checkState() == Qt.Checked:
            item.setBackground(QBrush(GT_BG))
        else:
            item.setBackground(QBrush(INBETWEEN_BG))

    def _on_item_changed(self, item):
        if item.data(Qt.UserRole) == "__placeholder__":
            return
        self._apply_item_bg(item)

    def toggle_selected_references(self):
        selected = self.timeline_list.selectedItems()
        if not selected: return
        # If any are unchecked, check them all. Otherwise uncheck all.
        any_unchecked = any(it.checkState() == Qt.Unchecked for it in selected)
        new_state = Qt.Checked if any_unchecked else Qt.Unchecked
        for it in selected:
            it.setCheckState(new_state)

    def _on_item_double_clicked(self, item):
        # Only toggle if it's the timeline workspace
        if self.tabs.currentIndex() == 0:
            new_state = Qt.Checked if item.checkState() == Qt.Unchecked else Qt.Unchecked
            item.setCheckState(new_state)

    def add_image_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        for f in files:
            self._add_single_image_to_list(f)
        self._update_timeline_display()

    def add_image_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder with Images")
        if folder_path:
            image_files = []
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_files.append(os.path.join(folder_path, fname))
            image_files.sort()
            for f in image_files:
                self._add_single_image_to_list(f)
            self._update_timeline_display()

    def remove_selected_frames(self):
        selected_items = self.timeline_list.selectedItems()
        if not selected_items:
            return
        reply = QMessageBox.question(self, 'Remove Frames',
                                     f"Are you sure you want to remove {len(selected_items)} selected frame(s)?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            for item in selected_items:
                self.timeline_list.takeItem(self.timeline_list.row(item))
            self._update_timeline_display()

    def reverse_timeline_order(self):
        count = self.timeline_list.count()
        if count <= 1:
            return
        items = []
        for _ in range(count):
            items.append(self.timeline_list.takeItem(0))
        for item in reversed(items):
            self.timeline_list.addItem(item)
        self._update_timeline_display()

    def move_selected_to_front(self):
        selected = self.timeline_list.selectedItems()
        if not selected:
            return
        selected.sort(key=lambda it: self.timeline_list.row(it))
        for item in selected:
            self.timeline_list.takeItem(self.timeline_list.row(item))
            self.timeline_list.insertItem(0, item)
        self.timeline_list.clearSelection()
        for item in selected:
            item.setSelected(True)
        self._update_timeline_display()

    def move_selected_to_back(self):
        selected = self.timeline_list.selectedItems()
        if not selected:
            return
        selected.sort(key=lambda it: self.timeline_list.row(it))
        for item in selected:
            self.timeline_list.takeItem(self.timeline_list.row(item))
            self.timeline_list.addItem(item)
        self.timeline_list.clearSelection()
        for item in selected:
            item.setSelected(True)
        self._update_timeline_display()

    def clear_timeline(self):
        if self.timeline_list.count() == 0:
            return
        reply = QMessageBox.question(self, 'Clear Timeline',
                                     "Are you sure you want to clear all frames from the timeline?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.timeline_list.clear()
            self._update_timeline_display()

    def _update_timeline_display(self):
        for i in range(self.timeline_list.count()):
            item = self.timeline_list.item(i)
            if item.data(Qt.UserRole) != "__placeholder__":
                file_path = item.data(Qt.UserRole)
                filename = os.path.basename(file_path)
                item.setText(f"Frame {i:04d}\n{filename}")
        self._update_workspace_view()

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.out_entry.setCurrentText(path)

    def run_colorization(self):
        timeline_items = []
        for i in range(self.timeline_list.count()):
            item = self.timeline_list.item(i)
            if item.data(Qt.UserRole) != "__placeholder__":
                timeline_items.append({
                    "path": item.data(Qt.UserRole),
                    "is_ref": item.checkState() == Qt.Checked
                })

        if not timeline_items:
            QMessageBox.critical(self, "Error", "Please add at least one frame.")
            return
        
        has_ref = any(item["is_ref"] for item in timeline_items)
        if not has_ref:
            QMessageBox.critical(self, "Error", "Please mark at least one frame as a reference (GT).")
            return

        out_dir = self.out_entry.currentText().strip()
        if not out_dir:
            QMessageBox.critical(self, "Error", "Please select an output directory.")
            return

        ckpt_name = "basicpbc.pth"
        ckpt_path = os.path.join(BASE_DIR, "ckpt", ckpt_name)
        if not os.path.exists(ckpt_path):
            QMessageBox.critical(
                self, "Missing checkpoint",
                f"Checkpoint not found at:\n{ckpt_path}\n\n"
                "Please download the pretrained model from the BasicPBC GitHub releases "
                "and place it in the 'ckpt' folder."
            )
            return

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.console.clear()
        self.progress.show()
        self.results_list.clear()
        self.status_label.setText("Preparing data…")

        if self.btn_use_cuda.isChecked():
            self.console.append("ℹ️ UI Preference: Attempting to use CUDA GPU acceleration.")
        else:
            self.console.append("ℹ️ UI Preference: Forcing CPU mode.")

        self.worker = InferenceWorker(
            timeline_items, out_dir,
            self.mode_combo.currentText(),
            self.seg_combo.currentText(),
            self.keepline_check.isChecked(),
            self.raft_edit.currentText(),
            self.thr_spin.value(),
            self.white_bg_check.isChecked(),
            self.final_line_check.isChecked(),
            self.btn_use_cuda.isChecked(), # Pass CUDA preference
        )
        self.worker.log_signal.connect(self.console.append)
        self.worker.status_signal.connect(self.status_label.setText)
        self.worker.progress_start_signal.connect(self.progress.show)
        self.worker.progress_stop_signal.connect(self.progress.hide)
        self.worker.open_output_signal.connect(self.open_output_folder)
        self.worker.finished_signal.connect(self.on_inference_finished)
        self.worker.start()

    def stop_colorization(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            self.btn_stop.setEnabled(False)
            self.status_label.setText("Stopping...")

    def on_inference_finished(self, success, message):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if success:
            self.status_label.setText("Colorization completed successfully.")
        else:
            self.status_label.setText(f"Colorization failed: {message}")
            if message and message != "Stopped by user":
                QMessageBox.warning(self, "Error", message)

    def populate_results(self, path):
        self.no_results_label.hide()
        self.results_viewer.show()
        self.results_list.clear()
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        files.sort()
        for f in files:
            pix = QPixmap(f).scaled(self.thumb_size, self.thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item = QListWidgetItem(QIcon(pix), os.path.basename(f))
            item.setData(Qt.UserRole, f)
            item.setSizeHint(QSize(self.thumb_size + 24, self.thumb_size + 60))
            self.results_list.addItem(item)
        self.tabs.setCurrentIndex(1)

    def open_output_folder(self, path):
        self.populate_results(path)
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Colorization Complete")
        msg_box.setText("Animation colorization finished successfully!")
        msg_box.setInformativeText(f"Results saved to:\n{path}")
        msg_box.setIcon(QMessageBox.Information)
        open_button = msg_box.addButton("Open Output Folder", QMessageBox.ActionRole)
        ok_button = msg_box.addButton(QMessageBox.Ok)
        msg_box.exec()
        if msg_box.clickedButton() == open_button:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    def save_settings(self):
        settings = QSettings("BasicPBC", "GUI")
        settings.setValue("theme", self.current_theme)
        settings.setValue("use_cuda", self.btn_use_cuda.isChecked())
        settings.setValue("show_landing", self.show_landing_pref)
        settings.setValue("output_dir", self.out_entry.currentText())
        settings.setValue("mode", self.mode_combo.currentText())
        settings.setValue("seg_type", self.seg_combo.currentText())
        settings.setValue("line_thr", self.thr_spin.value())
        settings.setValue("force_white_bg", self.white_bg_check.isChecked())
        settings.setValue("treat_as_final", self.final_line_check.isChecked())
        settings.setValue("keep_line", self.keepline_check.isChecked())
        settings.setValue("raft_res", self.raft_edit.currentText())
        settings.setValue("thumb_size", self.thumb_size)
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("ws_view_mode", self.timeline_viewer.stack.currentIndex())
        settings.setValue("res_view_mode", self.results_viewer.stack.currentIndex())

    def load_settings(self):
        settings = QSettings("BasicPBC", "GUI")
        self.current_theme = settings.value("theme", "dark")
        self.btn_use_cuda.setChecked(settings.value("use_cuda", True, type=bool))
        self.show_landing_pref = settings.value("show_landing", True, type=bool)
        self.out_entry.setCurrentText(settings.value("output_dir", ""))
        self.mode_combo.setCurrentText(settings.value("mode", "auto"))
        self.seg_combo.setCurrentText(settings.value("seg_type", "default"))
        self.thr_spin.setValue(int(settings.value("line_thr", 50)))
        self.white_bg_check.setChecked(settings.value("force_white_bg", False, type=bool))
        self.final_line_check.setChecked(settings.value("treat_as_final", False, type=bool))
        self.keepline_check.setChecked(settings.value("keep_line", False, type=bool))
        self.raft_edit.setCurrentText(settings.value("raft_res", "320"))
        self.thumb_size = int(settings.value("thumb_size", THUMB_SIZE))
        
        if self.current_theme == "light":
            self.btn_light_mode.setChecked(True)
            self.btn_dark_mode.setChecked(False)
        else:
            self.btn_dark_mode.setChecked(True)
            self.btn_light_mode.setChecked(False)

        self.btn_toggle_landing.setChecked(self.show_landing_pref)
        self.btn_toggle_landing.setText("Enabled" if self.show_landing_pref else "Disabled")
        self._update_workspace_view()

        if self.thumb_size != THUMB_SIZE:
            self.update_thumbnail_size()

        # Restore view modes
        self.timeline_viewer.set_view_mode(int(settings.value("ws_view_mode", 0)))
        self.results_viewer.set_view_mode(int(settings.value("res_view_mode", 0)))

        geom = settings.value("geometry")
        if geom:
            self.restoreGeometry(geom)

        # Apply the initial player view background according to the loaded setting
        self._on_force_white_bg_toggled(self.white_bg_check.isChecked())

    def reset_to_defaults(self):
        """Restores the UI settings to their standard default values."""
        reply = QMessageBox.question(self, 'Reset Settings',
                                     "Are you sure you want to reset all colorization options and preferences to their standard values?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Reset colorization options (right panel)
            self.mode_combo.setCurrentText("auto")
            self.seg_combo.setCurrentText("default")
            self.thr_spin.setValue(50)
            self.white_bg_check.setChecked(False)
            self.final_line_check.setChecked(False)
            self.keepline_check.setChecked(False)
            self.raft_edit.setCurrentText("320")
            self.out_entry.setCurrentText("")
            
            # Reset UI preferences
            self.btn_use_cuda.setChecked(True)
            self.btn_toggle_landing.setChecked(True)
            self.change_theme("dark")
            
            # Reset thumbnail size
            self.thumb_size = 100
            self.update_thumbnail_size()
            self.status_label.setText("Settings reset to defaults.")
            
            # Player view background will be updated automatically by the toggle signal


if __name__ == "__main__":
    # Necessary for PyInstaller bundles using multiprocessing or subprocess re-entry on Windows
    multiprocessing.freeze_support()

    # Check if the EXE is being called to run a bundled script (like inference_line_frames.py)
    if len(sys.argv) > 1 and sys.argv[1].endswith("inference_line_frames.py"):
        script_path = sys.argv[1]
        # Remove the script path from argv so the script's argparse sees the correct flags
        sys.argv.pop(1)
        # Ensure the bundled modules can be found in the temporary extraction directory
        script_dir = os.path.dirname(script_path)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        runpy.run_path(script_path, run_name="__main__")
        sys.exit(0)

    app = QApplication(sys.argv)
    window = ColorizationApp()
    window.show()
    sys.exit(app.exec())