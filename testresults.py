import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import glob
import cv2
import numpy as np
from skimage import io, color
from skimage.color import deltaE_ciede2000, rgb2lab


# =============================================================================
#  Robust image loader – always returns an RGB (3‑channel) array
# =============================================================================
def read_image_rgb(path):
    """
    Load an image and convert it to a float64 RGB array.
    - Grayscale images are expanded to 3 identical channels.
    - RGBA images keep only the first 3 channels (alpha discarded).
    """
    im = io.imread(path)
    if im.ndim == 2:                     # grayscale
        im = np.stack([im, im, im], axis=-1)
    elif im.shape[-1] == 4:              # RGBA → drop alpha
        im = im[..., :3]
    return im


# =============================================================================
#  Black frame detection
# =============================================================================
def is_mostly_black(image_path, black_threshold=10, black_fraction=0.9):
    """
    Returns True if more than `black_fraction` of pixels have intensity
    (in grayscale) below `black_threshold`.
    """
    img = read_image_rgb(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    dark_pixels = np.sum(gray < black_threshold)
    total_pixels = gray.size
    return (dark_pixels / total_pixels) > black_fraction


# =============================================================================
#  Temporal Variance (Warp Error) – fixed version with stop flag
# =============================================================================
def warp_error_percent(frame1_path, frame2_path):
    # Load both images as uint8 RGB
    im1 = read_image_rgb(frame1_path)   # shape (H, W, 3), dtype uint8
    im2 = read_image_rgb(frame2_path)

    # Force same dimensions: resize im2 to im1's size if needed
    if im1.shape[:2] != im2.shape[:2]:
        im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Convert to Lab color space (float64)
    lab1 = rgb2lab(im1)
    lab2 = rgb2lab(im2)

    # Convert to grayscale for optical flow (uint8)
    gray1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

    # Compute Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    h, w = flow.shape[:2]

    # Build remap coordinates: x' = x + flow_x, y' = y + flow_y
    map_x = (np.arange(w) + flow[..., 0]).astype(np.float32)
    map_y = (np.arange(h)[:, np.newaxis] + flow[..., 1]).astype(np.float32)

    # Warp the first Lab image using the flow
    warped_lab1 = cv2.remap(lab1, map_x, map_y, cv2.INTER_LINEAR)

    # Compute pixel-wise Euclidean distance in Lab space
    diff = np.sqrt(np.sum((warped_lab1 - lab2) ** 2, axis=-1))

    # Convert mean error to a percentage (original heuristic)
    warp_err_pct = (np.mean(diff) / 255.0) * 100.0
    return warp_err_pct


def temporal_variance_over_sequence(frame_paths, max_frames=None, stop_flag=None):
    """
    Compute warp errors for consecutive frame pairs.
    If max_frames is given, only the first `max_frames` frames are used.
    If stop_flag is a threading.Event, the loop checks it and stops early.
    Returns (mean_we, max_we, pair_errors, stopped_early)
    """
    if max_frames is not None and max_frames > 0:
        frames_to_use = frame_paths[:max_frames]
    else:
        frames_to_use = frame_paths

    pair_errors = []
    stopped_early = False
    for i in range(len(frames_to_use) - 1):
        if stop_flag and stop_flag.is_set():
            stopped_early = True
            break
        we = warp_error_percent(frames_to_use[i], frames_to_use[i + 1])
        pair_errors.append(we)
    mean_we = np.mean(pair_errors) if pair_errors else 0.0
    max_we = np.max(pair_errors) if pair_errors else 0.0
    return mean_we, max_we, pair_errors, stopped_early


# =============================================================================
#  Color Accuracy functions (with stop flag support for batch processing)
# =============================================================================
def pixel_color_accuracy_deltaE(gt_path, pred_path, threshold=2.0):
    gt_rgb = read_image_rgb(gt_path)
    pred_rgb = read_image_rgb(pred_path)
    gt_lab = color.rgb2lab(gt_rgb)
    pred_lab = color.rgb2lab(pred_rgb)
    deltaE_map = deltaE_ciede2000(gt_lab, pred_lab)
    accurate_mask = deltaE_map < threshold
    accuracy_percent = np.mean(accurate_mask) * 100.0
    mean_deltaE = np.mean(deltaE_map)
    return accuracy_percent, mean_deltaE


def pixel_color_accuracy_euclidean(gt_path, pred_path, threshold=2.3):
    gt_rgb = read_image_rgb(gt_path)
    pred_rgb = read_image_rgb(pred_path)
    gt_lab = color.rgb2lab(gt_rgb)
    pred_lab = color.rgb2lab(pred_rgb)
    diff = np.sqrt(np.sum((gt_lab - pred_lab) ** 2, axis=-1))
    accurate_mask = diff < threshold
    accuracy_percent = np.mean(accurate_mask) * 100.0
    mean_diff = np.mean(diff)
    return accuracy_percent, mean_diff


def color_accuracy_over_list(gt_path, pred_paths, metric, threshold, max_images=None, stop_flag=None):
    """
    Process a list of predicted images.
    If max_images is given, only the first `max_images` are used.
    If stop_flag is a threading.Event, the loop checks it and stops early.
    Returns (results_list, stopped_early)
    """
    if max_images is not None and max_images > 0:
        paths_to_use = pred_paths[:max_images]
    else:
        paths_to_use = pred_paths

    results = []
    stopped_early = False
    for pred in paths_to_use:
        if stop_flag and stop_flag.is_set():
            stopped_early = True
            break
        if metric == "ΔE2000":
            acc, me = pixel_color_accuracy_deltaE(gt_path, pred, threshold)
        else:
            acc, me = pixel_color_accuracy_euclidean(gt_path, pred, threshold)
        results.append((os.path.basename(pred), acc, me))
    return results, stopped_early


# =============================================================================
#  Main Application
# =============================================================================
class MultiAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CART‑Colour Analysis Suite")
        self.root.geometry("950x750")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_warp = ttk.Frame(self.notebook)
        self.tab_color = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_warp, text="Temporal Variance (Warp Error)")
        self.notebook.add(self.tab_color, text="Color Accuracy (ΔE / Euclidean)")

        self.build_warp_tab()
        self.build_color_tab()

        self.status_label = tk.Label(root, text="Ready.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Stop flags for each tab
        self.stop_warp_flag = None
        self.stop_color_flag = None

    # ------------------------------------------------------------------
    #  Warp Error Tab (already has limit & stop)
    # ------------------------------------------------------------------
    def build_warp_tab(self):
        self.warp_paths = []

        # Top button row
        btn_frame = tk.Frame(self.tab_warp)
        btn_frame.pack(pady=5)

        self.warp_folder_btn = tk.Button(btn_frame, text="Select Folder", command=self.warp_select_folder)
        self.warp_folder_btn.pack(side=tk.LEFT, padx=5)

        self.warp_files_btn = tk.Button(btn_frame, text="Select Images", command=self.warp_select_files)
        self.warp_files_btn.pack(side=tk.LEFT, padx=5)

        self.warp_run_btn = tk.Button(btn_frame, text="Run Analysis", command=self.warp_run_analysis, state=tk.DISABLED)
        self.warp_run_btn.pack(side=tk.LEFT, padx=5)

        self.warp_stop_btn = tk.Button(btn_frame, text="Stop Analysis", command=self.warp_stop_analysis, state=tk.DISABLED)
        self.warp_stop_btn.pack(side=tk.LEFT, padx=5)

        # Frame limit options
        limit_frame = tk.Frame(self.tab_warp)
        limit_frame.pack(pady=5, fill=tk.X)
        self.limit_var = tk.BooleanVar(value=False)
        tk.Checkbutton(limit_frame, text="Limit frames to first", variable=self.limit_var).pack(side=tk.LEFT)
        self.max_frames_var = tk.StringVar(value="20")
        tk.Entry(limit_frame, textvariable=self.max_frames_var, width=6).pack(side=tk.LEFT, padx=5)
        tk.Label(limit_frame, text="frames (0 = all)").pack(side=tk.LEFT)

        # Listbox for selected files
        list_frame = tk.Frame(self.tab_warp)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.warp_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, width=100, height=8)
        self.warp_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.warp_listbox.yview)

        # Results text area
        result_frame = tk.Frame(self.tab_warp)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.warp_result_text = scrolledtext.ScrolledText(result_frame, width=100, height=15, state=tk.DISABLED)
        self.warp_result_text.pack(fill=tk.BOTH, expand=True)

    def warp_set_status(self, msg):
        self.status_label.config(text=msg)
        self.root.update_idletasks()

    def warp_update_file_list(self):
        self.warp_listbox.delete(0, tk.END)
        for p in self.warp_paths:
            self.warp_listbox.insert(tk.END, os.path.basename(p))
        if self.warp_paths:
            self.warp_run_btn.config(state=tk.NORMAL)
            self.warp_set_status(f"{len(self.warp_paths)} frames loaded.")
        else:
            self.warp_run_btn.config(state=tk.DISABLED)
            self.warp_set_status("No frames loaded.")
        self.warp_result_text.config(state=tk.NORMAL)
        self.warp_result_text.delete(1.0, tk.END)
        self.warp_result_text.config(state=tk.DISABLED)

    def warp_select_folder(self):
        folder = filedialog.askdirectory(title="Select Folder Containing Image Sequence")
        if folder:
            extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
            paths = []
            for ext in extensions:
                paths.extend(glob.glob(os.path.join(folder, ext)))
                paths.extend(glob.glob(os.path.join(folder, ext.upper())))
            paths = sorted(set(paths))
            if not paths:
                messagebox.showwarning("No Images", "No image files found.")
                return
            self.warp_paths = paths
            self.warp_update_file_list()

    def warp_select_files(self):
        files = filedialog.askopenfilenames(
            title="Select Image Files",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        if files:
            self.warp_paths = sorted(list(files))
            self.warp_update_file_list()

    def warp_check_black_frames(self):
        """Return (black_frames_list, continue_flag)"""
        black_frames = []
        for path in self.warp_paths:
            if is_mostly_black(path):
                black_frames.append(os.path.basename(path))
        if black_frames:
            msg = (
                f"The following frames appear to be >90% black (or very dark):\n"
                f"{', '.join(black_frames)}\n\n"
                f"Warp error calculation on very dark sequences can produce unreliable results.\n"
                f"Do you still want to proceed?"
            )
            answer = messagebox.askyesno("Black Frames Detected", msg)
            return black_frames, answer
        return [], True

    def warp_run_analysis(self):
        if len(self.warp_paths) < 2:
            messagebox.showwarning("Not Enough Frames", "Select at least 2 consecutive frames.")
            return

        # Check for black frames before starting
        black_frames, cont = self.warp_check_black_frames()
        if not cont:
            self.warp_set_status("Analysis cancelled by user.")
            return

        # Parse max frames limit
        max_frames = None
        if self.limit_var.get():
            try:
                max_frames = int(self.max_frames_var.get())
                if max_frames <= 0:
                    max_frames = None
            except ValueError:
                messagebox.showerror("Invalid Limit", "Frame limit must be a positive integer.")
                return

        # Disable UI during analysis
        self.warp_run_btn.config(state=tk.DISABLED)
        self.warp_folder_btn.config(state=tk.DISABLED)
        self.warp_files_btn.config(state=tk.DISABLED)
        self.warp_stop_btn.config(state=tk.NORMAL)

        # Create stop event
        self.stop_warp_flag = threading.Event()

        self.warp_set_status("Computing warp error...")
        threading.Thread(target=self.warp_compute, args=(max_frames,), daemon=True).start()

    def warp_stop_analysis(self):
        if self.stop_warp_flag:
            self.stop_warp_flag.set()
            self.warp_set_status("Stopping analysis... (will finish current pair)")
            self.warp_stop_btn.config(state=tk.DISABLED)

    def warp_compute(self, max_frames):
        try:
            mean_err, max_err, pair_errors, stopped_early = temporal_variance_over_sequence(
                self.warp_paths, max_frames, self.stop_warp_flag
            )
            self.root.after(0, self.warp_display, mean_err, max_err, pair_errors, stopped_early)
        except Exception as e:
            self.root.after(0, self.warp_error, str(e))

    def warp_display(self, mean_err, max_err, pair_errors, stopped_early):
        t = self.warp_result_text
        t.config(state=tk.NORMAL)
        t.delete(1.0, tk.END)
        if stopped_early:
            t.insert(tk.END, "*** ANALYSIS STOPPED BY USER ***\n")
            t.insert(tk.END, f"Pairs evaluated before stop: {len(pair_errors)}\n")
        else:
            t.insert(tk.END, f"Pairs evaluated: {len(pair_errors)}\n")
        t.insert(tk.END, f"Mean Warp Error: {mean_err:.4f}%\n")
        t.insert(tk.END, f"Max Warp Error: {max_err:.4f}%\n\n")
        if mean_err < 2.0:
            t.insert(tk.END, "Target achieved: <2% temporal variance\n\n")
        else:
            t.insert(tk.END, "Target not achieved (>=2%)\n\n")
        for i, e in enumerate(pair_errors, 1):
            t.insert(tk.END, f"  Pair {i}: {e:.4f}%\n")
        t.config(state=tk.DISABLED)

        # Re-enable UI
        self.warp_run_btn.config(state=tk.NORMAL)
        self.warp_folder_btn.config(state=tk.NORMAL)
        self.warp_files_btn.config(state=tk.NORMAL)
        self.warp_stop_btn.config(state=tk.DISABLED)
        self.stop_warp_flag = None
        if stopped_early:
            self.warp_set_status("Analysis stopped by user.")
        else:
            self.warp_set_status("Warp analysis complete.")

    def warp_error(self, msg):
        messagebox.showerror("Error", msg)
        self.warp_run_btn.config(state=tk.NORMAL)
        self.warp_folder_btn.config(state=tk.NORMAL)
        self.warp_files_btn.config(state=tk.NORMAL)
        self.warp_stop_btn.config(state=tk.DISABLED)
        self.stop_warp_flag = None
        self.warp_set_status("Error.")

    # ------------------------------------------------------------------
    #  Color Accuracy Tab (now with limit & stop)
    # ------------------------------------------------------------------
    def build_color_tab(self):
        self.color_gt_path = tk.StringVar()
        self.color_pred_paths = []

        # Ground Truth selection
        gt_frame = tk.LabelFrame(self.tab_color, text="Ground‑Truth Image (single file)")
        gt_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Entry(gt_frame, textvariable=self.color_gt_path, width=80).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(gt_frame, text="Browse...", command=self.select_color_gt).pack(side=tk.LEFT, padx=5)

        # Predicted images list
        pred_frame = tk.LabelFrame(self.tab_color, text="Predicted Images (multiple)")
        pred_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        pred_btn_frame = tk.Frame(pred_frame)
        pred_btn_frame.pack(fill=tk.X, padx=5, pady=2)

        tk.Button(pred_btn_frame, text="Add Files...", command=self.add_color_pred_files).pack(side=tk.LEFT, padx=2)
        tk.Button(pred_btn_frame, text="Add Folder...", command=self.add_color_pred_folder).pack(side=tk.LEFT, padx=2)
        tk.Button(pred_btn_frame, text="Remove Selected", command=self.remove_color_pred).pack(side=tk.LEFT, padx=2)
        tk.Button(pred_btn_frame, text="Clear All", command=self.clear_color_preds).pack(side=tk.LEFT, padx=2)

        list_frame = tk.Frame(pred_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.color_pred_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                             selectmode=tk.EXTENDED, width=100, height=10)
        self.color_pred_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.color_pred_listbox.yview)

        # Parameters and limits
        param_frame = tk.Frame(self.tab_color)
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        # Metric and threshold
        tk.Label(param_frame, text="Metric:").pack(side=tk.LEFT)
        self.metric_var = tk.StringVar(value="ΔE2000")
        self.metric_menu = ttk.Combobox(param_frame, textvariable=self.metric_var,
                                        values=["ΔE2000", "Euclidean Lab Distance"],
                                        state="readonly", width=22)
        self.metric_menu.pack(side=tk.LEFT, padx=5)
        self.metric_menu.bind("<<ComboboxSelected>>", self.update_threshold_label)

        tk.Label(param_frame, text="Threshold (<):").pack(side=tk.LEFT, padx=(15, 2))
        self.threshold_var = tk.StringVar(value="2.0")
        tk.Entry(param_frame, textvariable=self.threshold_var, width=6).pack(side=tk.LEFT)
        tk.Label(param_frame, text="(ΔE2000 default 2.0, Euclidean default 2.3)").pack(side=tk.LEFT, padx=5)

        # Limit options (new)
        limit_frame = tk.Frame(self.tab_color)
        limit_frame.pack(fill=tk.X, padx=10, pady=5)
        self.color_limit_var = tk.BooleanVar(value=False)
        tk.Checkbutton(limit_frame, text="Limit to first", variable=self.color_limit_var).pack(side=tk.LEFT)
        self.color_max_images_var = tk.StringVar(value="20")
        tk.Entry(limit_frame, textvariable=self.color_max_images_var, width=6).pack(side=tk.LEFT, padx=5)
        tk.Label(limit_frame, text="images (0 = all)").pack(side=tk.LEFT)

        # Buttons (Run and Stop)
        button_frame = tk.Frame(self.tab_color)
        button_frame.pack(pady=10)
        self.color_run_btn = tk.Button(button_frame, text="Compute Accuracy", command=self.color_run_analysis)
        self.color_run_btn.pack(side=tk.LEFT, padx=5)
        self.color_stop_btn = tk.Button(button_frame, text="Stop Analysis", command=self.color_stop_analysis, state=tk.DISABLED)
        self.color_stop_btn.pack(side=tk.LEFT, padx=5)

        # Results text area
        result_frame = tk.Frame(self.tab_color)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.color_result_text = scrolledtext.ScrolledText(result_frame, width=100, height=18, state=tk.DISABLED)
        self.color_result_text.pack(fill=tk.BOTH, expand=True)

    def update_threshold_label(self, event=None):
        if self.metric_var.get() == "ΔE2000":
            self.threshold_var.set("2.0")
        else:
            self.threshold_var.set("2.3")

    def select_color_gt(self):
        f = filedialog.askopenfilename(title="Select Ground‑Truth Image")
        if f:
            self.color_gt_path.set(f)

    def add_color_pred_files(self):
        files = filedialog.askopenfilenames(
            title="Select Predicted Images",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        if files:
            added = 0
            for f in files:
                if f not in self.color_pred_paths:
                    self.color_pred_paths.append(f)
                    self.color_pred_listbox.insert(tk.END, os.path.basename(f))
                    added += 1
            if added:
                self.color_set_status(f"{added} file(s) added.")

    def add_color_pred_folder(self):
        folder = filedialog.askdirectory(title="Select Folder Containing Predicted Images")
        if folder:
            extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(folder, ext)))
                files.extend(glob.glob(os.path.join(folder, ext.upper())))
            files = sorted(set(files))
            added = 0
            for f in files:
                if f not in self.color_pred_paths:
                    self.color_pred_paths.append(f)
                    self.color_pred_listbox.insert(tk.END, os.path.basename(f))
                    added += 1
            if added:
                self.color_set_status(f"{added} file(s) added from folder.")
            else:
                self.color_set_status("No new images found (all already in list).")

    def remove_color_pred(self):
        selected_indices = self.color_pred_listbox.curselection()
        if not selected_indices:
            return
        for i in sorted(selected_indices, reverse=True):
            del self.color_pred_paths[i]
            self.color_pred_listbox.delete(i)
        self.color_set_status(f"{len(selected_indices)} item(s) removed.")

    def clear_color_preds(self):
        self.color_pred_paths.clear()
        self.color_pred_listbox.delete(0, tk.END)
        self.color_set_status("Predicted list cleared.")

    def color_set_status(self, msg):
        self.status_label.config(text=msg)
        self.root.update_idletasks()

    def color_run_analysis(self):
        gt = self.color_gt_path.get().strip()
        if not gt or not os.path.isfile(gt):
            messagebox.showwarning("Missing GT", "Please select a valid ground‑truth image.")
            return
        if not self.color_pred_paths:
            messagebox.showwarning("No Predictions", "Add at least one predicted image.")
            return

        metric = self.metric_var.get()
        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror("Invalid Threshold", "Threshold must be a number.")
            return

        # Parse max images limit
        max_images = None
        if self.color_limit_var.get():
            try:
                max_images = int(self.color_max_images_var.get())
                if max_images <= 0:
                    max_images = None
            except ValueError:
                messagebox.showerror("Invalid Limit", "Image limit must be a positive integer.")
                return

        # Disable UI during analysis
        self.color_run_btn.config(state=tk.DISABLED)
        self.color_stop_btn.config(state=tk.NORMAL)
        # Also disable the add/remove buttons to prevent list changes mid‑analysis
        for child in self.tab_color.winfo_children():
            if isinstance(child, tk.Frame):
                for btn in child.winfo_children():
                    if isinstance(btn, tk.Button) and btn not in [self.color_run_btn, self.color_stop_btn]:
                        btn.config(state=tk.DISABLED)

        # Create stop event
        self.stop_color_flag = threading.Event()

        self.color_set_status("Computing color accuracy...")
        threading.Thread(target=self.color_compute, args=(gt, metric, threshold, max_images), daemon=True).start()

    def color_stop_analysis(self):
        if self.stop_color_flag:
            self.stop_color_flag.set()
            self.color_set_status("Stopping analysis... (will finish current image)")
            self.color_stop_btn.config(state=tk.DISABLED)

    def color_compute(self, gt, metric, threshold, max_images):
        try:
            results, stopped_early = color_accuracy_over_list(
                gt, self.color_pred_paths, metric, threshold, max_images, self.stop_color_flag
            )
            self.root.after(0, self.color_display, metric, threshold, results, stopped_early)
        except Exception as e:
            self.root.after(0, self.color_error, str(e))

    def color_display(self, metric, threshold, results, stopped_early):
        t = self.color_result_text
        t.config(state=tk.NORMAL)
        t.delete(1.0, tk.END)

        if stopped_early:
            t.insert(tk.END, "*** ANALYSIS STOPPED BY USER ***\n")
            t.insert(tk.END, f"Images processed before stop: {len(results)}\n\n")
        else:
            t.insert(tk.END, f"Number of predictions evaluated: {len(results)}\n\n")

        t.insert(tk.END, f"Ground Truth: {os.path.basename(self.color_gt_path.get())}\n")
        t.insert(tk.END, f"Metric: {metric}\nThreshold: {threshold}\n")

        if not results:
            t.insert(tk.END, "No results.\n")
            t.config(state=tk.DISABLED)
            self.color_reenable_ui()
            self.color_set_status("Done (no results).")
            return

        accuracies = [r[1] for r in results]
        mean_errs = [r[2] for r in results]
        avg_acc = np.mean(accuracies)
        avg_err = np.mean(mean_errs)
        count_pass = sum(1 for a in accuracies if a >= 90.0)

        t.insert(tk.END, f"\nAverage Accuracy: {avg_acc:.2f}%\n")
        t.insert(tk.END, f"Average Mean {'ΔE' if metric=='ΔE2000' else 'Euclidean Dist'}: {avg_err:.3f}\n")
        t.insert(tk.END, f"Images meeting ≥90% target: {count_pass}/{len(results)}\n\n")

        t.insert(tk.END, "Per‑image results:\n")
        for name, acc, me in results:
            status = "PASS" if acc >= 90.0 else "FAIL"
            t.insert(tk.END, f"  {name}: Acc={acc:.2f}%  Mean={'ΔE' if metric=='ΔE2000' else 'Dist'}={me:.3f}  [{status}]\n")

        t.config(state=tk.DISABLED)
        self.color_reenable_ui()
        if stopped_early:
            self.color_set_status("Analysis stopped by user.")
        else:
            self.color_set_status("Analysis complete.")

    def color_reenable_ui(self):
        """Re-enable all color tab UI elements after analysis finishes."""
        self.color_run_btn.config(state=tk.NORMAL)
        self.color_stop_btn.config(state=tk.DISABLED)
        # Re-enable add/remove buttons
        for child in self.tab_color.winfo_children():
            if isinstance(child, tk.Frame):
                for btn in child.winfo_children():
                    if isinstance(btn, tk.Button) and btn not in [self.color_run_btn, self.color_stop_btn]:
                        btn.config(state=tk.NORMAL)
        self.stop_color_flag = None

    def color_error(self, msg):
        messagebox.showerror("Error", msg)
        self.color_reenable_ui()
        self.color_set_status("Error.")


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiAnalysisApp(root)
    root.mainloop()