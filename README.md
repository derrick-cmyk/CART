# CART (Colorization AI with Reference Transfer) - User Manual

Welcome to **CART**, an ethical, artist-centered AI tool designed to automate the repetitive task of coloring 2D animation frames while preserving your creative authority. CART uses your own artwork as a reference to colorize sequences of line art.

---

## 1. Installation & Preparation

### System Requirements
*   **OS:** Windows 10/11
*   **Python:** Version 3.8 or higher is recommended.
*   **Hardware:** 
    *   **Minimum:** 8GB RAM, Intel Core i5 (or equivalent).
    *   **Recommended:** 16GB RAM, NVIDIA GPU (RTX 3060+) for CUDA acceleration.

### Environment Setup
1.  **Clone/Extract the Repository:**
    Ensure you are in the `BasicPBC` root directory.

2.  **Install Base Dependencies:**
    Open your terminal/PowerShell and run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install PyTorch (CUDA vs CPU):**
    *   **For GPU Acceleration (NVIDIA only):**
        Check your CUDA version (`nvcc --version`) and install the matching build:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```
    *   **For CPU-only:**
        ```bash
        pip install torch torchvision torchaudio
        ```

4.  **Download Pre-trained Models:**
    Place your model checkpoints (e.g., `basicpbc_pbch.pth`) into the `ckpt/` folder. Ensure the RAFT weights are located at `raft/ckpt/raft-animerun-v2-ft_again.pth`.

---

## 2. Running the Program

To launch the Graphical User Interface, run:
```bash
python CART_GUI.py
```

---

## 3. Workflow & Usage

### Workspace Tab
1.  **Load Frames:** Use "Add Images" or "Add Folder" to import your line art sequence.
2.  **Timeline Management:** Drag and drop frames to reorder them chronologically.
3.  **Designate References (GT):** Check the checkbox next to a frame to mark it as a **Ground Truth (Reference)**. This frame must be already colored.
    *   **GT Frames:** Indicated by a light blue background.
    *   **In-betweens:** Indicated by a bright blue background.
4.  **Run:** Click the "Run Colorization" button (Blue).

### Results Tab
Once processing finishes, colorized frames will appear here for review. You can toggle between **Tile View** (overview) and **Player View** (focus/scrubbing).

---

## 4. Colorization Options Explained

| Option | Description |
| :--- | :--- |
| **Mode: Auto** | **Recommended.** Uses BFS to find the shortest path to the nearest Reference frame, minimizing color bleed. |
| **Mode: Forward** | Processes frames sequentially from the first Reference frame to the end. |
| **Mode: Backward** | Processes frames in reverse from a Reference frame. |
| **Mode: Nearest** | Copies colors from the chronologically closest Reference frame without model inference. |
| **Seg Type** | `default` is fast; `trappedball` is better for sketches with unclosed lines (holes). |
| **Line-mask Thr** | Adjusts the sensitivity of line detection. Default is 50. |
| **Force White Canvas** | Composites transparent PNGs onto white to prevent "black silhouette" bugs. |
| **Treat Line as Final**| Bypasses line extraction. Use this if your "line art" is already shaded or contains color you want the AI to see. |
| **RAFT Res** | Resolution for optical flow. Use `320` for speed or `640` for better tracking in complex motion. |

---

## 5. Keybinds & Shortcuts

Use these shortcuts to speed up your animation pipeline:

*   **Ctrl + A:** Select all frames in the timeline.
*   **Delete / Backspace:** Remove selected frames from the project.
*   **Space / R:** Toggle "Reference Frame" status for selected items.
*   **Double-Click:** Toggle "Reference Frame" status.
*   **Ctrl + Plus (+):** Increase thumbnail size.
*   **Ctrl + Minus (-):** Decrease thumbnail size.
*   **Ctrl + R / Ctrl + Enter:** Execute the colorization process.

---

## 6. Settings & Diagnostics

*   **Enable CUDA Acceleration:** When toggled ON, CART will use your GPU. If the software is running slow (15s+ per frame), ensure this is active and your drivers are updated.
*   **Theme Selection:** Toggle between  **Dark Mode** and  **Light Mode**.
*   **Diagnostic Banner:** When you run the model, check the console for the `--> Execution Device` message. It will confirm if you are running on **GPU (CUDA)** or **CPU**.

---

## This project was built with using https://github.com/ykdai/BasicPBC Thank You so much for making this possible

*CART is an open-source project dedicated to empowering the animation community. Please respect the intellectual property of fellow artists.*