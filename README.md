<div align="center">

# 🧠 CSRNet — Real-Time Crowd Counting & Density Estimation

### Deep learning-powered head counting on crowded scenes, trained end-to-end on ShanghaiTech and deployed on real railway platform footage.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Video%20Inference-green?logo=opencv)](https://opencv.org)
[![Colab](https://img.shields.io/badge/Google%20Colab-GPU%20T4-yellow?logo=googlecolab)](https://colab.research.google.com)

</div>

---

## What We Built

A full end-to-end crowd counting system based on the **CSRNet** architecture — from raw dataset preprocessing and adaptive density map generation, through 150-epoch phased transfer learning with VGG16, to real-time video inference with live count overlays and jet-colormap density heatmaps.

The model was trained on the **ShanghaiTech Part A** dataset and deployed on real-world **railway platform surveillance footage**, producing a count video, a density heatmap video, and a combined side-by-side demo.

---

## Sample Inference Result

> Predicted density map overlaid on a test image from ShanghaiTech Part A:

![Inference Result](inference_result.png)

---


## Evaluation Results — ShanghaiTech Part A Test Set

| Metric | Value |
|---|---|
| Avg Loss (pixel-level RMSE) | **34.40** |
| Count MAE | **86.18 people** |
| Count MSE | **123.39 people** |
| Dataset | ShanghaiTech Part A |
| Total Epochs | 150 (3 phases × 50) |
| Model Params | 16.26M total (14.5M trainable) |

> MAE and MSE on the held-out test split are the standard benchmarks in crowd counting research — not classification accuracy.

---

## Full Training Pipeline

```mermaid
flowchart TD
    A([🗂️ ShanghaiTech Part A\nTrain Images + .mat Annotations]) --> B

    subgraph PREPROCESS ["📦 Preprocessing & Augmentation  —  DataGenerator"]
        B[Load image\nNormalize ÷255\nPer-channel mean/std normalization] --> C
        C{Flip flag?}
        C -- Yes --> D[Horizontal flip image\nMirror x-coordinates of annotations]
        C -- No --> E[Keep original]
        D --> F
        E --> F[Pad to multiple of 32\nbottom & right only]
        F --> G[Scale annotation points\n÷ subsampling factor 8\nto density map space]
        G --> H[Generate Adaptive Gaussian\nDensity Map via KD-Tree\nσ = 0.3 × avg dist to 3 nearest neighbours]
        H --> I{Random < 0.3?}
        I -- 30% Full Image --> J[Resize to 3×H/4 × 3×W/4\nRescale density map\npreserve total count]
        I -- 70% Patch Mosaic --> K[Extract 9 patches\n4 fixed quadrants + 5 random\nAssemble into 3×3 grid]
        J --> L([Augmented Image + Density Map])
        K --> L
    end

    subgraph MODEL ["🧠 CSRNet Model"]
        M[Input: H × W × 3] --> N
        N["VGG16 Backbone\n(pretrained ImageNet)\noutput: block4_conv3\n→ H/8 × W/8 × 512"] --> O
        O["Dilated Conv Head\nConv2D 512 dilation=2\nConv2D 512 dilation=2\nConv2D 512 dilation=2\nConv2D 256 dilation=2\nConv2D 128 dilation=2\nConv2D  64 dilation=2\nConv2D   1 dilation=1"] --> P
        P["Density Map Output\nH/8 × W/8\nsum = predicted count"]
    end

    subgraph TRAIN ["🏋️ Phased Training  —  150 Epochs"]
        Q["Phase 1  |  Epochs 1–50\nBackbone FROZEN  |  LR = 1e-4\nTrain density head only"] --> R
        R["Phase 2  |  Epochs 51–100\nBackbone UNFROZEN  |  LR = 1e-5\nFine-tune full network"] --> S
        S["Phase 3  |  Epochs 101–150\nFull model  |  LR = 1e-6\nFinal convergence"]
        S --> T["Patience-based LR halving\nif no improvement after N epochs\nmin LR = 1e-7"]
        T --> U[Save best weights\nCSRNet.weights.h5]
    end

    subgraph LOSS ["📉 Loss Function"]
        V["custom_loss = Σ (y_true − y_pred)²\nPixel-wise sum of squared errors\nover the density map"]
    end

    L --> MODEL
    MODEL --> LOSS
    LOSS --> TRAIN

    subgraph EVAL ["📊 Evaluation — ShanghaiTech Part A Test Set"]
        W["Load test images + .mat annotations\nGenerate ground truth density maps\nRun model inference"] --> X
        X["Avg Loss RMSE : 34.40\nCount MAE      : 86.18 people\nCount MSE      : 123.39 people"]
    end

    U --> EVAL
```

---

## Video Inference Pipeline

```mermaid
flowchart LR
    A([🎥 Input Video\nRailway Platform .mp4]) --> B[Read frame with OpenCV]
    B --> C{frame_idx % FRAME_SKIP == 0?}
    C -- Yes → run inference --> D[Preprocess frame\nBGR→RGB, ÷255\nper-channel normalize\npad to multiple of 32]
    D --> E[Model forward pass\npredict density map]
    E --> F[sum density map\n= crowd count]
    C -- No → reuse cached --> G[Last count + density map]
    F --> H
    G --> H[Render count overlay\nblack box + white text]
    F --> I[Render density heatmap\njet colormap → BGR\nresize to original resolution]
    H --> J([🎥 Real-time-count.mp4])
    I --> K([🎥 Real-time-density.mp4])
    J --> L[Side-by-side merge\nlabel panels]
    K --> L
    L --> M([🎥 combined_side_by_side.mp4])
```

---

## Architecture

| Layer | Output Shape | Params |
|---|---|---|
| Input | (None, None, None, 3) | 0 |
| VGG16 → block4_conv3 | (None, None, None, 512) | 7,635,264 |
| Conv2D 512, 3×3, dilation=2 | (None, None, None, 512) | 2,359,808 |
| Conv2D 512, 3×3, dilation=2 | (None, None, None, 512) | 2,359,808 |
| Conv2D 512, 3×3, dilation=2 | (None, None, None, 512) | 2,359,808 |
| Conv2D 256, 3×3, dilation=2 | (None, None, None, 256) | 1,179,904 |
| Conv2D 128, 3×3, dilation=2 | (None, None, None, 128) | 295,040 |
| Conv2D 64, 3×3, dilation=2 | (None, None, None, 64) | 73,792 |
| Conv2D 1, 1×1, dilation=1 | (None, None, None, 1) | 65 |
| **Total** | | **16,263,489** |

---

## Repository Structure

```
📦 CSRNet-Crowd-Counting
├── 📓 CSRnet-final-trianing-testing-code.ipynb   ← full training + evaluation pipeline
├── 🏋️ CSRNet.weights.h5                          ← trained model weights (16M params)
├── 🖼️  inference_result.png                       ← sample test image inference
├── 📄 CSRnet-Research-paper(Reference).pdf        ← original CSRNet paper
│
├── 📁 video1/                                     ← Railway platform video #1 (7.6s, 60fps, 457 frames)
│   ├── 📓 CSRnet_Inference_on_video(frame-skip-1).ipynb   ← inference every frame
│   ├── 📓 CSRnet_Inference_on_video(frame-skip-5).ipynb   ← inference every 5th frame
│   ├── 🎥 Railway-platform-video.mp4
│   ├── 🎥 Real-time-count(frame-skip-1).mp4
│   ├── 🎥 Real-time-count(frame-skip-5).mp4
│   ├── 🎥 Real-time-density(frame-skip-1).mp4
│   └── 🎥 Real-time-density(frame-skip-5).mp4
│
└── 📁 video2/                                     ← Railway platform video #2 (15.4s, 60fps, 924 frames)
    ├── 📓 CSRnet_Inference_on_video(frame-skip-5).ipynb   ← inference + side-by-side export
    ├── 🎥 Railway-platform-video.mp4
    ├── 🎥 Real-time-count.mp4
    ├── 🎥 Real-time-density.mp4
    └── 🎥 combined_side_by_side(frame-skip-5).mp4         ← ⭐ main demo video
```

---

## Tech Stack

| Library | Usage |
|---|---|
| TensorFlow / Keras | Model definition, custom training loop, GradientTape |
| VGG16 (ImageNet) | Pretrained feature extractor backbone |
| OpenCV | Video I/O, frame preprocessing, rendering |
| SciPy (KDTree) | Adaptive Gaussian kernel computation |
| NumPy | Array ops, density map generation, patch mosaic |
| Matplotlib (cm.jet) | Density heatmap colormap for video output |
| Google Colab (T4 GPU) | Training and inference environment |

---

## How to Run

**1. Setup dataset**
```
Unzip ShanghaiTech dataset and place under Google Drive or local path.
Update train_images, train_maps, test_images, test_maps paths in the notebook.
```

**2. Train the model**
```
Open CSRnet-final-trianing-testing-code.ipynb in Colab
Run all cells — trains for 150 epochs across 3 phases
Best weights auto-saved as CSRNet.weights.h5
```

**3. Run video inference**
```
Open video1/ or video2/ inference notebook
Set VIDEO_PATH to your input video
Set FRAME_SKIP (1 = every frame, 5 = faster)
Outputs: Real-time-count.mp4 + Real-time-density.mp4
```

**4. Generate side-by-side demo**
```
Run the final cell in video2/CSRnet_Inference_on_video(frame-skip-5).ipynb
Outputs combined_side_by_side.mp4 with labeled panels
```

---

## Reference

> Li, Y., Zhang, X., & Chen, D. (2018). **CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes.** CVPR 2018.
> [https://arxiv.org/abs/1802.10062](https://arxiv.org/abs/1802.10062)
