# Image Classifier

A real-time image classification web app powered by **MobileNetV2** trained on ImageNet. Upload any photo and get the model's top three predictions instantly — built with Python, Keras, and Streamlit.

---

## Demo

> Upload an image → the model scans it → three predictions ranked by confidence, rendered in under a second.

![App Screenshot](screenshot.png)

---

## Features

- **Instant inference** — MobileNetV2 runs lightweight enough to classify in real time directly in the browser
- **Top-3 predictions** — returns the three most confident labels with confidence scores and visual progress bars
- **1,000 object classes** — trained on ImageNet, covering animals, vehicles, food, everyday objects, and more
- **Clean, production-grade UI** — custom-styled Streamlit interface; not a default template
- **Cached model loading** — model is loaded once and reused across sessions for performance

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit + Custom CSS |
| Model | MobileNetV2 (ImageNet weights) |
| Deep Learning | TensorFlow / Keras |
| Image Processing | OpenCV, Pillow, NumPy |
| Language | Python 3.10+ |

---

## Project Structure

```
image-classifier/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── screenshot.png      # App preview
└── README.md
```

---


## Requirements

```
streamlit>=1.30.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
tensorflow>=2.13.0
keras>=2.13.0
```

---

## How It Works

1. **Upload** — the user uploads a JPEG or PNG image via the file uploader
2. **Preprocess** — the image is resized to 224×224 pixels and normalized using MobileNetV2's `preprocess_input`
3. **Inference** — the preprocessed image is passed through MobileNetV2 with ImageNet weights
4. **Decode** — `decode_predictions` maps the raw output logits to human-readable class labels
5. **Display** — the top 3 predictions are rendered with their confidence scores

### Why MobileNetV2?

MobileNetV2 was designed for efficiency — it achieves strong ImageNet accuracy with a fraction of the parameters of larger models like VGG or ResNet. This makes it ideal for real-time web applications where speed and resource usage matter.

---

## Model Details

| Property | Value |
|---|---|
| Architecture | MobileNetV2 |
| Input Size | 224 × 224 × 3 |
| Training Dataset | ImageNet (1.2M images) |
| Output Classes | 1,000 |
| Weights | Pre-trained (no fine-tuning) |

---
