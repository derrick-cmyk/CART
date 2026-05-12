# CART (Colorization AI with Reference Transfer) - GUI Documentation & Change Log

## Overview of the GUI System
The CART GUI is a sophisticated, artist-centric application built using **Python** and **PySide6 (Qt6)**. It provides a seamless interface for animators to manage their production pipeline, specifically focusing on automating the tedious task of frame colorization using reference-based AI.

### Key Components
- **Main Window**: A tabbed interface providing distinct workflows for `Workspace` (preparation), `Results` (review), and `Settings` (configuration).
- **FrameViewer**: A reusable custom widget used in both the timeline and results views. It features a "Dual View" system:
    - **Tile View**: A traditional grid layout for broad oversight and batch selection.
    - **Player View**: A focused playback environment featuring a large preview, frame scrubber, and a horizontal "filmstrip" for navigation.
- **ReorderListWidget**: A customized `QListWidget` that implements advanced drag-and-drop logic, allowing users to reorder frames or move selections of frames intuitively.
- **InferenceWorker**: A background thread manager that handles the execution of the AI model via subprocesses, ensuring the GUI remains responsive during heavy computation.

### Core Features
- **Intuitive Timeline Management**: Drag-and-drop reordering, multi-selection support, and quick actions (reverse, move to front/back).
- **Advanced Processing Options**: Real-time control over RAFT resolution, segmentation types (`default` vs `trappedball`), and line-mask thresholds.
- **Reference Frame System**: Simple checkbox-based designation of Ground Truth (GT) frames which guide the AI.
- **Responsive UI**: Fully integrated Dark and Light themes with dynamic switching.
- **Hardware Optimization**: One-click toggle between CUDA (NVIDIA GPU) and CPU processing modes.
- **Auto-Persistence**: Remembers window size, processing preferences, and view modes between sessions using `QSettings`.

## Recent Changes and QoL Improvements

### Navigation & User Experience
- **Interactive Landing Page**: Added a splash overlay for the Workspace tab explaining the basic workflow. It includes direct "Add Images/Folder" buttons and automatically disappears once frames are uploaded or clicked.
- **Landing Page Toggle**: Added a preference in the Settings tab to enable or disable the introduction screen on startup.
- **Horizontal Player View**: Redesigned the frame list in Player View to be strictly horizontal, maximizing vertical space for the preview area.
- **Horizontal Auto-Scroll**: Implemented intelligent auto-scrolling during drag-and-drop in the horizontal filmstrip view.

### Efficiency & Shortcuts
- **Multi-item Dragging**: Enabled the ability to select and drag multiple frames simultaneously in both Tile and Player views, complete with a visual badge indicating the number of items being moved.
- **Enhanced Keyboard Shortcuts**:
    - `Delete` / `Backspace`: Quickly remove selected frames.
    - `Space` / `R`: Toggle "Reference Frame" status for all selected items.
    - `Ctrl + A`: Select all frames in the current list.
    - `Ctrl + R` / `Ctrl + Enter`: Instantly trigger the colorization process.
- **Double-Click Interaction**: Users can now double-click any frame in the workspace to toggle its reference (GT) status.

### Visuals & Customization
- **Dynamic List Resizing**: Integrated the list height with the `Ctrl + / -` zoom feature. Resizing thumbnails now automatically adjusts the filmstrip height in Player View.
- **View Mode Persistence**: The application now saves whether the user was in Tile View or Player View independently for both the Workspace and Results tabs.
- **Theming Updates**: Enhanced the "Landing Page" and UI components to be fully compatible and visually distinct in both Light and Dark modes.
- **Multi-selection Badge**: Improved the visual feedback when dragging multiple items by adding a high-contrast count badge.

### Technical Improvements
- **CUDA Logic Integration**: Explicitly connected the UI "Use CUDA" toggle to the inference engine, allowing users to force CPU usage on systems with limited VRAM.
- **Automated Path Resolution**: Improved how the GUI handles internal paths for bundled scripts when running as a standalone executable.
- **Pure-PyTorch Compatibility**: Replaced native `torch_scatter` dependencies with a custom `scatter_add_` implementation in the architecture files, resolving binary compatibility errors (`WinError 127`) common on Windows systems.
- **Alpha-Channel Resilience**: Implemented automatic channel slicing in the inference pipeline. This ensures the model and RAFT (optical flow) receive 3-channel RGB data even if the input line art contains an unexpected alpha channel (RGBA).
- **Dynamic Bounding Box Calculation**: Added logic to `pbc_model.py` to compute and scale segment bounding boxes (keypoints) on-the-fly. This was critical for enabling model inference during the backward pass and "Auto" mode where pre-indexed coordinates are unavailable.
- **Smart Propagation Engine (BFS)**: Overhauled the propagation logic to use a Breadth-First Search strategy. In "Auto" mode, the system now calculates the shortest chronological path from any uncolored frame to its nearest Ground Truth, significantly reducing cumulative color drift.
- **Leaky Interpolation Fix**: Resolved a `RuntimeError` where line art tensors were not being correctly re-scaled after the RAFT optical flow estimation, ensuring dimension alignment before concatenation.

## Stability & UX Refinements

### Reliability
- **Refined Workspace Cleanup**: updated the `InferenceWorker` logic to perform a full recursive deletion of the `_gui_workspace` folder upon completion, preventing disk clutter.
- **Hardware Status Banner**: Integrated a prominent terminal status banner that confirms exactly which execution device (CUDA GPU vs. CPU) is being utilized at the start of inference.
- **Encoding Compatibility**: Fixed a `UnicodeEncodeError` in the terminal output triggered by special character printing in standard Windows PowerShell environments.

### Secondary Platforms
- **Ollama Local UI**: Developed a standalone companion platform (HTML/Flask) to interface with locally run Ollama models (DeepSeek-R1, Llama 3.2, etc.).
    - **Paper Mode**: A specialized creative writing tab allowing users to collaboratively build stories with AI, supporting both append and targeted edit modes.
    - **Flask Proxy**: Implemented a Python-based bridge to bypass browser CORS restrictions without requiring manual environment configuration.
    - **Connection Diagnostics**: Added a dedicated "Check Connection" utility to verify Ollama status and list installed local models.

## Maintenance & Tests
- **Standardized Benchmarks**: Verified the system against industry-standard tests including the `laughing_girl` and `smoke_explosion` sets.
- **Dependency Alignment**: Documented the requirement for matching `torchvision` and `pytorch` versions to prevent `deform_conv2d` loading failures.