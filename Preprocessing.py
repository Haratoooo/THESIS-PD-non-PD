import os
from PIL import Image
import torch
import torchvision.transforms as transforms

# ✅ Preprocessing Transform (for feature extraction and evaluation)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match CNN input
    transforms.ToTensor(),          # Convert to tensor with pixel values in [0, 1]
    transforms.Normalize(           # Normalize using ImageNet mean and std
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 📂 Recursively collect image paths and assign labels
def get_labeled_image_paths(base_dir):
    image_paths = []
    labels = []
    label_map = {'PD': 1, 'Healthy': 0}

    for label_name, label_value in label_map.items():
        class_dir = os.path.join(base_dir, label_name)
        for subfolder in ['wave', 'spiral']:
            sub_dir = os.path.join(class_dir, subfolder)
            for root, _, files in os.walk(sub_dir):
                for filename in files:
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(root, filename))
                        labels.append(label_value)
    return image_paths, labels

# 🖼️ Preprocess a single image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')  # Ensure 3 channels
    return preprocess(img)  # Output: torch.Size([3, 224, 224])

# 📦 Preprocess all images in the dataset
def preprocess_all_images(base_dir):
    image_paths, labels = get_labeled_image_paths(base_dir)
    preprocessed_images = []
    for path in image_paths:
        img_tensor = preprocess_image(path)
        preprocessed_images.append(img_tensor)
    return preprocessed_images, labels, image_paths

# ✅ Example usage
if __name__ == "__main__":
    base_dir = "D:\\dataset"
    images, labels, paths = preprocess_all_images(base_dir)

    # Optional: Stack into a batch tensor
    batch = torch.stack(images)  # Shape: [N, 3, 224, 224]

    # Print summary
    print(f"Total images: {len(images)}")
    print(f"First image shape: {images[0].shape}")
    print(f"First label: {labels[0]}")
    print(f"First path: {paths[0]}")

    from collections import Counter
    label_counts = Counter(labels)
    print(f"PD samples: {label_counts[1]}")
    print(f"Healthy samples: {label_counts[0]}")
