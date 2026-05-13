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
#  Temporal Variance (Warp Error) – unchanged
# =============================================================================
def warp_error_percent(frame1_path, frame2_path):
    im1 = io.imread(frame1_path)
    im2 = io.imread(frame2_path)
    lab1 = rgb2lab(im1)
    lab2 = rgb2lab(im2)
    gray1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    h, w = flow.shape[:2]
    flow_map = np.column_stack([
        flow[..., 0].ravel() + np.tile(np.arange(w), h),
        flow[..., 1].ravel() + np.tile(np.arange(h).reshape(-1, 1), (1, w)).ravel()
    ]).reshape(h, w, 2).astype(np.float32)
    warped_lab1 = cv2.remap(lab1, flow_map, None, cv2.INTER_LINEAR)
    diff = np.sqrt(np.sum((warped_lab1 - lab2) ** 2, axis=-1))
    warp_err_pct = (np.mean(diff) / 255.0) * 100.0
    return warp_err_pct


def temporal_variance_over_sequence(frame_paths):
    pair_errors = []
    for i in range(len(frame_paths) - 1):
        we = warp_error_percent(frame_paths[i], frame_paths[i + 1])
        pair_errors.append(we)
    mean_we = np.mean(pair_errors) if pair_errors else 0.0
    max_we = np.max(pair_errors) if pair_errors else 0.0
    return mean_we, max_we, pair_errors


# =============================================================================
#  Color Accuracy functions
# =============================================================================
def pixel_color_accuracy_deltaE(gt_path, pred_path, threshold=2.0):
    """Accuracy using ΔE2000."""
    gt_rgb = io.imread(gt_path)
    pred_rgb = io.imread(pred_path)
    gt_lab = color.rgb2lab(gt_rgb)
    pred_lab = color.rgb2lab(pred_rgb)
    deltaE_map = deltaE_ciede2000(gt_lab, pred_lab)
    accurate_mask = deltaE_map < threshold
    accuracy_percent = np.mean(accurate_mask) * 100.0
    mean_deltaE = np.mean(deltaE_map)
    return accuracy_percent, mean_deltaE


def pixel_color_accuracy_euclidean(gt_path, pred_path, threshold=2.3):
    """Accuracy using Euclidean distance in CIELab."""
    gt_rgb = io.imread(gt_path)
    pred_rgb = io.imread(pred_path)
    gt_lab = color.rgb2lab(gt_rgb)
    pred_lab = color.rgb2lab(pred_rgb)
    diff = np.sqrt(np.sum((gt_lab - pred_lab) ** 2, axis=-1))
    accurate_mask = diff < threshold
    accuracy_percent = np.mean(accurate_mask) * 100.0
    mean_diff = np.mean(diff)
    return accuracy_percent, mean_diff


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

    # ------------------------------------------------------------------
    #  Warp Error Tab (unchanged)
    # ------------------------------------------------------------------
    def build_warp_tab(self):
        self.warp_paths = []

        btn_frame = tk.Frame(self.tab_warp)
        btn_frame.pack(pady=5)

        self.warp_folder_btn = tk.Button(btn_frame, text="Select Folder", command=self.warp_select_folder)
        self.warp_folder_btn.pack(side=tk.LEFT, padx=5)

        self.warp_files_btn = tk.Button(btn_frame, text="Select Images", command=self.warp_select_files)
        self.warp_files_btn.pack(side=tk.LEFT, padx=5)

        self.warp_run_btn = tk.Button(btn_frame, text="Run Analysis", command=self.warp_run_analysis, state=tk.DISABLED)
        self.warp_run_btn.pack(side=tk.LEFT, padx=5)

        list_frame = tk.Frame(self.tab_warp)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.warp_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, width=100, height=8)
        self.warp_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.warp_listbox.yview)

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

    def warp_run_analysis(self):
        if len(self.warp_paths) < 2:
            messagebox.showwarning("Not Enough Frames", "Select at least 2 consecutive frames.")
            return
        self.warp_run_btn.config(state=tk.DISABLED)
        self.warp_folder_btn.config(state=tk.DISABLED)
        self.warp_files_btn.config(state=tk.DISABLED)
        self.warp_set_status("Computing warp error...")
        threading.Thread(target=self.warp_compute, daemon=True).start()

    def warp_compute(self):
        try:
            mean_err, max_err, pair_errors = temporal_variance_over_sequence(self.warp_paths)
            self.root.after(0, self.warp_display, mean_err, max_err, pair_errors)
        except Exception as e:
            self.root.after(0, self.warp_error, str(e))

    def warp_display(self, mean_err, max_err, pair_errors):
        t = self.warp_result_text
        t.config(state=tk.NORMAL)
        t.delete(1.0, tk.END)
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
        self.warp_run_btn.config(state=tk.NORMAL)
        self.warp_folder_btn.config(state=tk.NORMAL)
        self.warp_files_btn.config(state=tk.NORMAL)
        self.warp_set_status("Warp analysis complete.")

    def warp_error(self, msg):
        messagebox.showerror("Error", msg)
        self.warp_run_btn.config(state=tk.NORMAL)
        self.warp_folder_btn.config(state=tk.NORMAL)
        self.warp_files_btn.config(state=tk.NORMAL)
        self.warp_set_status("Error.")

    # ------------------------------------------------------------------
    #  Color Accuracy Tab  (IMPROVED – multiple predictions against one GT)
    # ------------------------------------------------------------------
    def build_color_tab(self):
        self.color_gt_path = tk.StringVar()          # single GT file
        self.color_pred_paths = []                   # list of predicted file paths

        # ----- Ground Truth selection -----
        gt_frame = tk.LabelFrame(self.tab_color, text="Ground‑Truth Image (single file)")
        gt_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Entry(gt_frame, textvariable=self.color_gt_path, width=80).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(gt_frame, text="Browse...", command=self.select_color_gt).pack(side=tk.LEFT, padx=5)

        # ----- Predicted images list -----
        pred_frame = tk.LabelFrame(self.tab_color, text="Predicted Images (multiple)")
        pred_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Buttons to add/remove predictions
        pred_btn_frame = tk.Frame(pred_frame)
        pred_btn_frame.pack(fill=tk.X, padx=5, pady=2)

        tk.Button(pred_btn_frame, text="Add Files...", command=self.add_color_pred_files).pack(side=tk.LEFT, padx=2)
        tk.Button(pred_btn_frame, text="Add Folder...", command=self.add_color_pred_folder).pack(side=tk.LEFT, padx=2)
        tk.Button(pred_btn_frame, text="Remove Selected", command=self.remove_color_pred).pack(side=tk.LEFT, padx=2)
        tk.Button(pred_btn_frame, text="Clear All", command=self.clear_color_preds).pack(side=tk.LEFT, padx=2)

        # Listbox with scrollbar for predicted images
        list_frame = tk.Frame(pred_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.color_pred_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                             selectmode=tk.EXTENDED, width=100, height=10)
        self.color_pred_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.color_pred_listbox.yview)

        # ----- Parameters (metric, threshold) -----
        param_frame = tk.Frame(self.tab_color)
        param_frame.pack(fill=tk.X, padx=10, pady=5)

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

        # Run button
        self.color_run_btn = tk.Button(self.tab_color, text="Compute Accuracy", command=self.color_run_analysis)
        self.color_run_btn.pack(pady=10)

        # Results area
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
        # Remove in reverse order to keep indices valid
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

        self.color_run_btn.config(state=tk.DISABLED)
        self.color_set_status("Computing color accuracy...")
        threading.Thread(target=self.color_compute, args=(gt, metric, threshold), daemon=True).start()

    def color_compute(self, gt, metric, threshold):
        results = []
        try:
            for pred in self.color_pred_paths:
                if metric == "ΔE2000":
                    acc, me = pixel_color_accuracy_deltaE(gt, pred, threshold)
                else:
                    acc, me = pixel_color_accuracy_euclidean(gt, pred, threshold)
                results.append((os.path.basename(pred), acc, me))
            self.root.after(0, self.color_display, metric, threshold, results)
        except Exception as e:
            self.root.after(0, self.color_error, str(e))

    def color_display(self, metric, threshold, results):
        t = self.color_result_text
        t.config(state=tk.NORMAL)
        t.delete(1.0, tk.END)

        t.insert(tk.END, f"Ground Truth: {os.path.basename(self.color_gt_path.get())}\n")
        t.insert(tk.END, f"Metric: {metric}\nThreshold: {threshold}\n")
        t.insert(tk.END, f"Number of predictions evaluated: {len(results)}\n\n")

        if not results:
            t.insert(tk.END, "No results.\n")
            t.config(state=tk.DISABLED)
            self.color_run_btn.config(state=tk.NORMAL)
            self.color_set_status("Done.")
            return

        # Overall statistics
        accuracies = [r[1] for r in results]
        mean_errs = [r[2] for r in results]
        avg_acc = np.mean(accuracies)
        avg_err = np.mean(mean_errs)
        count_pass = sum(1 for a in accuracies if a >= 90.0)

        t.insert(tk.END, f"Average Accuracy: {avg_acc:.2f}%\n")
        t.insert(tk.END, f"Average Mean {'ΔE' if metric=='ΔE2000' else 'Euclidean Dist'}: {avg_err:.3f}\n")
        t.insert(tk.END, f"Images meeting ≥90% target: {count_pass}/{len(results)}\n\n")

        t.insert(tk.END, "Per‑image results:\n")
        for name, acc, me in results:
            status = "PASS" if acc >= 90.0 else "FAIL"
            t.insert(tk.END, f"  {name}: Acc={acc:.2f}%  Mean={'ΔE' if metric=='ΔE2000' else 'Dist'}={me:.3f}  [{status}]\n")

        t.config(state=tk.DISABLED)
        self.color_run_btn.config(state=tk.NORMAL)
        self.color_set_status("Analysis complete.")

    def color_error(self, msg):
        messagebox.showerror("Error", msg)
        self.color_run_btn.config(state=tk.NORMAL)
        self.color_set_status("Error.")


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiAnalysisApp(root)
    root.mainloop()