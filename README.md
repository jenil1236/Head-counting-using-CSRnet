# Head Counting using CSRNet

## Project Overview
This repository implements a head-counting and crowd density estimation system using CSRNet. The project trains a CSRNet-inspired deep neural network on dense crowd images and evaluates it using count-based metrics (MAE and MSE) rather than classification accuracy.

The core deliverables include:
- `CSRnet-final-trianing-testing-code.ipynb` — full training and evaluation pipeline with model architecture, data preprocessing, augmentation, training schedule, and inference.
- `CSRNet.weights.h5` — trained CSRNet model weights.
- `inference_result.png` — saved inference visualization for a sample test image.
- `video2/combined_side_by_side(frame-skip-5).mp4` — side-by-side inference demo video.
- `CSRnet-Research-paper(Reference).pdf` — reference paper for CSRNet.
- `ShanghaiTech.zip` — dataset archive used for training/testing.

## Key Achievement
- Best observed training performance: **count MAE ≈ 43.08** and **count MSE ≈ 60.77**.
- The project uses CSRNet-style density estimation for robust head counting in crowded scenes.

> Note: In crowd counting research, mean absolute error (MAE) and mean squared error (MSE) are the accepted performance metrics. This project reports those metrics rather than classic classification accuracy.

## Files and Structure

```
CSRnet-final-trianing-testing-code.ipynb
CSRNet.weights.h5
inference_result.png
CSRnet-Research-paper(Reference).pdf
ShanghaiTech.zip
video1/
  CSRnet_Inference_on_video(frame-skip-1).ipynb
  CSRnet_Inference_on_video(frame-skip-5).ipynb
  Railway-platform-video.mp4
  Real-time-count(frame-skip-1).mp4
  Real-time-count(frame-skip-5).mp4
  Real-time-density(frame-skip-1).mp4
  Real-time-density(frame-skip-5).mp4
video2/
  CSRnet_Inference_on_video(frame-skip-5).ipynb
  Real-time-count.mp4
  Real-time-density.mp4
  combined_side_by_side(frame-skip-5).mp4
```

## Tech Stack
- Python 3
- TensorFlow / Keras
- NumPy
- OpenCV
- SciPy
- Matplotlib
- Google Colab compatibility (`/content/drive` paths used in notebook)

## Core Approach
The implementation follows the CSRNet architecture:
- A pre-trained `VGG16` backbone is used as the feature extractor.
- A series of dilated convolutional layers form the density estimation head.
- The model predicts a density map whose spatial sum approximates the crowd count.
- The network is trained using a custom pixel-wise regression loss.

### Data augmentation and preprocessing
- Images are normalized per channel and padded to multiples of 32.
- Training augmentation includes image flipping and patch-mosaic generation.
- Ground truth density maps are generated using adaptive Gaussian kernels, following CSRNet conventions.

## Training Pipeline
Training is implemented inside `CSRnet-final-trianing-testing-code.ipynb` with a phased schedule:
- Phase 1: backbone frozen, learning rate `1e-4`
- Phase 2: backbone unfrozen, learning rate `1e-5`
- Phase 3: final fine-tuning, learning rate `1e-6`

This staged training helps stabilize learning and improves count estimation performance.

## Evaluation and Metrics
The notebook runs a test evaluation loop on the ShanghaiTech dataset split and reports:
- `Avg Loss` (pixel-level regression loss)
- `Count MAE` (mean absolute error)
- `Count MSE` (root mean squared error)

The notebook also includes a point-based localization analysis using peak extraction from predicted density maps.

## Demo and Inference
- `inference_result.png` shows a sample inference result with predicted density overlay.
- `video2/combined_side_by_side(frame-skip-5).mp4` demonstrates side-by-side video inference for model predictions versus original input.
- Additional demo videos and notebooks are available in `video1/` and `video2/`.

## How to Run
1. Unzip `ShanghaiTech.zip` and place the dataset under a local folder or Google Drive.
2. Open `CSRnet-final-trianing-testing-code.ipynb` in Jupyter or Google Colab.
3. Update dataset paths if needed:
   - `train_images`
   - `train_maps`
   - `test_images`
   - `test_maps`
4. Execute the notebook cells to train, evaluate, and run inference.
5. Use `CSRNet.weights.h5` for direct inference loading.

## Important Files
- `CSRnet-final-trianing-testing-code.ipynb`: main training + testing notebook
- `CSRNet.weights.h5`: trained model weights
- `inference_result.png`: evaluation snapshot
- `video2/combined_side_by_side(frame-skip-5).mp4`: demonstration video
- `CSRnet-Research-paper(Reference).pdf`: CSRNet reference paper

## References
- CSRNet research paper: [https://arxiv.org/abs/1802.10062](https://arxiv.org/abs/1802.10062)

## Interviewer-Friendly Summary
This project is a strong implementation of a crowd counting system using CSRNet. It demonstrates:
- end-to-end dataset preprocessing and augmentation,
- transfer learning with VGG16 backbone,
- density map regression with dilated convolutions,
- iterative training with phased learning rate scheduling,
- and clear evaluation using MAE/MSE metrics.

The repository is ready for review, with model weights, sample inference outputs, and demonstration videos included.
