# CART (Colorization AI with Reference Transfer) - Sequential Development Log

## [Initial Infrastructure & Hardware Stability]
### 1. CUDA Hardware Fallback Integration
- **Problem**: The core inference scripts (`inference_line_frames.py` and `pbc_model.py`) contained hardcoded `.cuda()` calls, which caused immediate crashes on systems without an NVIDIA GPU.
- **Solution**: Updated the backend to utilize `torch.device` auto-detection and replaced hardcoded calls with `model.to(device)`.
- **Rationale**: Ensured cross-hardware compatibility, allowing the software to run on CPU-only machines without specialized modifications.

### 2. Pure-PyTorch Architecture (Torch-Scatter Removal)
- **Problem**: The `torch_scatter` dependency frequently failed to load on Windows systems (`WinError 127`) due to binary mismatches in its native C++ extensions.
- **Solution**: Replaced `torch_scatter` with a custom implementation using native `torch.scatter_add_()` within the model architecture files.
- **Rationale**: Removed a high-friction dependency, simplifying the installation process and increasing the overall portability of the software.

### 3. Dependency Alignment & DeformConv Fix
- **Problem**: Version mismatches between `torchvision` and `pytorch` prevented the deformable convolution (`deform_conv2d`) operators from loading.
- **Solution**: Synchronized environment requirements to matching versions (e.g., PyTorch 2.5.1 paired with Torchvision 0.20.1).
- **Rationale**: Critical for the stability of the model's spatial alignment module.

### 4. Automated Path Resolution
- **Problem**: The GUI could not locate backend scripts when running as a standalone compiled executable.
- **Solution**: Improved internal script calling logic to use absolute path resolution relative to the application's runtime directory.
- **Rationale**: Guaranteed that the "Run Colorization" feature functions correctly after distribution via PyInstaller.

## [GUI Architecture & Workflow Transformation]
### 5. Unified Animation Timeline
- **Problem**: The original interface separated "Reference" and "Target" frames into different lists, making it difficult to visualize the actual animation sequence.
- **Solution**: Redesigned the GUI to use a single chronological timeline where "Ground Truth" (GT) frames are designated via checkboxes.
- **Rationale**: Native preservation of the animation's chronological order, making it significantly more intuitive for artists to seed color references.

### 6. Timeline UX & Efficiency Improvements
- **Problem**: Frame management was slow and lacked modern application feedback.
- **Solution**: Added multi-item dragging with a count badge, horizontal auto-scrolling, and professional shortcuts (`Delete` for removal, `Space` for GT toggle, `Ctrl+A`).
- **Rationale**: Accelerated production-scale workflows where users handle dozens or hundreds of frames at once.

### 7. Interactive Landing Page & Persistence
- **Problem**: A blank workspace on startup provided no guidance for new users.
- **Solution**: Added a landing page splash for empty projects and implemented `QSettings` to remember window size, themes, and view modes between sessions.
- **Rationale**: Reduced the software learning curve and created a persistent, personalized user environment.

## [Advanced Algorithmic & Pipeline Hardening]
### 8. Smart BFS Propagation Engine (Auto Mode)
- **Problem**: Simple sequential processing caused "color drift" (accumulating errors) the further a frame was from a reference.
- **Solution**: Overhauled propagation logic to use a Breadth-First Search (BFS) strategy. In "Auto" mode, the system calculates the shortest chronological path from any uncolored frame to its nearest GT.
- **Rationale**: Dramatically reduces color errors and bleed by ensuring every frame is colored using the most reliable available source.

### 9. Dynamic Bounding Box Calculation
- **Problem**: Segment matching requires bounding box keypoints, but these are often unavailable during non-sequential paths (like backward passes).
- **Solution**: Integrated on-the-fly calculation and spatial scaling of bounding boxes from mask data directly in `pbc_model.py`.
- **Rationale**: Necessary to enable robust "Backward" and "Auto" propagation modes where pre-indexed dataset coordinates are bypassed.

### 10. Alpha-Channel Resilience
- **Problem**: Optical flow (RAFT) and convolutional layers expect 3-channel RGB data, but transparent digital line art (RGBA) contains 4, leading to dimension mismatch crashes.
- **Solution**: Implemented automatic channel slicing (RGBA to RGB) across the entire inference pipeline and architectures.
- **Rationale**: Hardened the pipeline against varied digital art exports, ensuring transparent PNGs no longer cause system failures.

### 11. Preprocessing: White Canvas & Final Treatment
- **Problem**: Transparent images appeared as solid black silhouettes to the model, and pre-colored artwork was stripped during the line extraction phase.
- **Solution**: Added "Force White Canvas" (compositing onto white) and "Treat Line Images as Final" (skipping extraction) checkboxes.
- **Rationale**: Solved transparency-related rendering bugs and allowed artists to preserve existing colors or complex shading when using the AI.

### 12. Leaky Interpolation Fix
- **Problem**: A `RuntimeError` occurred when line art tensors remained at internal RAFT-resolution after flow estimation, causing a size mismatch during concatenation.
- **Solution**: Added explicit re-scaling of line art tensors to match target dimensions after the optical flow estimation step.
- **Rationale**: Resolved a primary cause of system crashes during Forward and Nearest propagation modes.

## [Stability & Diagnostic Phase]
### 13. Hardware Status Banner & Unicode Fix
- **Problem**: Terminal output used special characters (★) that caused `UnicodeEncodeError` in Windows environments, and hardware utilization was unclear.
- **Solution**: Replaced problematic characters with safe equivalents and integrated a prominent execution device banner (CUDA GPU vs CPU).
- **Rationale**: Provided transparent feedback on performance while ensuring terminal stability across different PowerShell configurations.

### 14. Refined Workspace Cleanup
- **Problem**: Temporary data in `_gui_workspace` was accumulating on the user's hard drive after each run.
- **Solution**: Updated `InferenceWorker` to perform a full recursive deletion of the workspace folder upon completion.
- **Rationale**: Prevented long-term disk clutter and maintained a clean application state.

### 15. Benchmark Verification
- **Activity**: Verified system performance against standard `laughing_girl` and `smoke_explosion` test sets.
- **Result**: Confirmed that all propagation modes (Forward, Backward, Nearest, Auto) are robust and production-ready.

## [Baseline Architecture Summary]
*The following components represent the stable foundation of the current system.*

### Key GUI Components
- **Main Window**: Tabbed interface separating Workspace, Results, and Settings.
- **FrameViewer**: A custom widget supporting focused "Player View" and bird's-eye "Tile View."
- **ReorderListWidget**: Custom `QListWidget` with advanced drag-and-drop logic for timeline manipulation.
- **InferenceWorker**: Background thread manager utilizing `subprocess` to keep the GUI responsive during heavy AI computation.

### Core Features
- **Intuitive Timeline**: Checkbox-based Reference Frame designation with visual color coding.
- **Hardware Optimization**: One-click toggle between CUDA acceleration and CPU-only modes.
- **Processing Flexibility**: Adjustable RAFT resolution, segmentation types (`default` vs `trappedball`), and grayscale thresholding.
- **Auto-Persistence**: Remembers all user preferences and window states using `QSettings`.