
```markdown
# Preprocessing for PD vs Healthy Handwriting Dataset

This repository provides the **image preprocessing and augmentation** pipeline for the Parkinson’s Disease (PD) vs Healthy handwriting dataset.  
It prepares spiral and wave handwriting images for deep learning by:

- Converting to grayscale  
- Resizing to `224 × 224`  
- Applying random on-the-fly augmentations (rotation, shift, shear, blur, Gaussian noise)  
- Normalizing pixel values  

---

## 📂 Dataset Structure

Organize the dataset as follows:

```

dataset/
│
├── PD/
│   ├── wave/
│   └── spiral/
│
└── Healthy/
├── wave/
└── spiral/

````

Each folder contains `.jpg`, `.jpeg`, or `.png` images.

---

## ⚡ Features

- **Two preprocessing pipelines**:
  - `augment_transform` → for training (with on-the-fly augmentation)
  - `standard_transform` → for evaluation/testing (no augmentation)
- **Custom Gaussian noise** transform (applied after tensor conversion)
- **Label mapping**:
  - PD → `1`
  - Healthy → `0`

---

## 🛠️ Usage

Run the preprocessing script:

```bash
python Preprocessing.py
````

The script will:

* Recursively load all images
* Apply the selected transform
* Stack images into tensors
* Print dataset statistics

---

## 👀 Optional Visualization

The script includes a visualization function to preview augmentations.

At the bottom of the file:

```python
if False:  # change to True to preview augmentations
    sample_path = paths[0]
    show_augmented_versions(sample_path, augment_transform)
```

* Change `False` → `True` to show 5 augmented versions of one image
* Default is `False` (no preview)

---

## 🔧 Integration with Training

⚠️ This script handles **preprocessing only**.
To use for training:

* Apply `augment_transform` to your **training dataset**
* Apply `standard_transform` to your **validation/test datasets**
* Wrap these in a **PyTorch Dataset + DataLoader** for multi-epoch training

---

## ✅ Requirements

Install dependencies:

```bash
pip install torch torchvision pillow matplotlib
```

---

## 👤 Notes

* Augmentations are **on-the-fly** → every epoch produces slightly different images
* Dataset remains **202 images total** (101 PD, 101 Healthy)
* Noise, blur, and rotations are applied **probabilistically**

```

---

