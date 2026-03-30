# Fashion-MINIST Neural Network Classification

A complete deep learning pipeline for classifying fashion items using the **Fashion-MNIST** dataset. Built with TensorFlow/Keras and extended with an **OpenCV-powered custom image prediction** system — including webcam support.

---

## 📌 Overview

This project trains a dense neural network to classify 28×28 grayscale images into 10 fashion categories. It goes beyond basic training by comparing multiple architectures, analyzing overfitting, and enabling real-world predictions on custom images.

---

## 🗂️ Dataset

**Fashion-MNIST** — a drop-in replacement for MNIST with real-world clothing items.

| Split    | Samples |
|----------|---------|
| Training | 60,000  |
| Test     | 10,000  |

### 10 Classes

| ID | Label        | ID | Label      |
|----|--------------|----|------------|
| 0  | T-shirt/top  | 5  | Sandal     |
| 1  | Trouser      | 6  | Shirt      |
| 2  | Pullover     | 7  | Sneaker    |
| 3  | Dress        | 8  | Bag        |
| 4  | Coat         | 9  | Ankle boot |

---

## 🧠 Model Architecture (Base)

```
Input  → 784 neurons  (flattened 28×28 image)
Hidden1 → 64 neurons  (ReLU)
Hidden2 → 32 neurons  (ReLU)
Output  → 10 neurons  (Softmax)
```

| Setting       | Value                          |
|---------------|--------------------------------|
| Optimizer     | Adam                           |
| Loss Function | Sparse Categorical Crossentropy|
| Epochs        | 15                             |
| Batch Size    | 32                             |
| Val Split     | 20%                            |

---

## 🔀 Architecture Variations

| Variant         | Hidden Layers | Test Accuracy | Parameters |
|-----------------|---------------|---------------|------------|
| Base (64→32)    | 64, 32        | ~87–88%       | ~52K       |
| V1 (128→64)     | 128, 64       | ~88–89%       | ~109K      |
| V2 (32→16)      | 32, 16        | ~86–87%       | ~26K       |
| V3 (128→128)    | 128, 128      | ~88–89%       | ~117K      |

---

## 📊 Output Files

| File                          | Description                              |
|-------------------------------|------------------------------------------|
| `01_sample_images.png`        | One sample image per class               |
| `02_base_training_curves.png` | Accuracy & loss curves for base model    |
| `03_confusion_matrix.png`     | Confusion matrix heatmap                 |
| `04_sample_predictions.png`   | 10 test predictions with confidence      |
| `05_variation_curves.png`     | Training curves for all 4 architectures  |
| `06_summary_comparison.png`   | Accuracy & parameter comparison bar chart|
| `07_correct_vs_incorrect.png` | Correct vs incorrect prediction per class|
| `08_opencv_demo_predictions.png` | OpenCV pipeline demo on all 10 classes|

---

## 🖼️ OpenCV Custom Image Prediction

Predict **your own image** using the trained model:

```python
# Predict from an image file
predict_custom_image('your_image.jpg')

# Predict from webcam (press SPACE to capture, ESC to cancel)
predict_from_camera()
```

### Preprocessing Pipeline

```
1. Read image (JPG, PNG, BMP, WEBP supported)
2. Convert BGR → Grayscale
3. Auto-invert if background is bright
4. Resize to 28×28 (INTER_AREA)
5. Normalize to [0, 1]
6. Flatten to (1, 784)
```

---

## 🔍 Key Findings

- **Base model** achieves ~87–89% test accuracy — strong for a dense-only model
- **Shirt** and **Coat** are hardest to classify — visually similar silhouettes
- **Trouser**, **Sandal**, and **Bag** are easiest — distinctive shapes
- Larger architectures (V1, V3) offer marginal gains at higher compute cost
- V2 (32→16) is the most lightweight — ideal for resource-constrained deployment
- Generalisation gap of ~2–4% indicates slight but normal overfitting

---


## ⚙️ Requirements

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn opencv-python
```

---


## 📁 Project Structure

```
├── fashion_minist_1_.py          # Main script
├── demo_images/                  # Auto-generated demo images (all 10 classes)
├── 01_sample_images.png
├── 02_base_training_curves.png
├── 03_confusion_matrix.png
├── 04_sample_predictions.png
├── 05_variation_curves.png
├── 06_summary_comparison.png
├── 07_correct_vs_incorrect.png
└── 08_opencv_demo_predictions.png
```

---

## 🙌 Acknowledgements

- Dataset: [Fashion-MNIST by Zalando Research](https://github.com/zalandoresearch/fashion-mnist)
- Framework: [TensorFlow / Keras](https://www.tensorflow.org/)
- Originally developed as a Google Colab notebook
