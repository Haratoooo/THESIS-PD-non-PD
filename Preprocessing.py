import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from collections import Counter

# ✅ Custom Gaussian Noise Transform
class GaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Add Gaussian noise directly to the tensor
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


# ✅ Augmentation Transform (for training)
augment_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # grayscale
    transforms.Resize((224, 224)),                 # resize
    transforms.RandomAffine(                       # affine transforms
        degrees=15,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
        shear=5
    ),
    transforms.RandomApply([                       # occasional blur
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    ], p=0.3),
    transforms.ToTensor(),                         # convert to tensor before noise
    transforms.RandomApply([                       # add Gaussian noise
        GaussianNoise(mean=0., std=0.05)
    ], p=0.2),
    transforms.Normalize(mean=[0.5], std=[0.5])    # normalize
])

# ✅ Standard Transform (for evaluation or feature extraction)
standard_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
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
def preprocess_image(image_path, transform):
    img = Image.open(image_path).convert('L')  # force grayscale
    return transform(img)


# 📦 Preprocess all images in the dataset
def preprocess_all_images(base_dir, transform):
    image_paths, labels = get_labeled_image_paths(base_dir)
    preprocessed_images = []
    for path in image_paths:
        img_tensor = preprocess_image(path, transform)
        preprocessed_images.append(img_tensor)
    return preprocessed_images, labels, image_paths


# ✅ Example usage
# ✅ Example usage
if __name__ == "__main__":
    base_dir = r"D:\dataset"  # Use raw string or forward slashes

    # Choose transform: augment_transform for training, standard_transform for evaluation
    images, labels, paths = preprocess_all_images(base_dir, transform=augment_transform)

    batch = torch.stack(images)  # Shape: [N, 1, 224, 224]

    print(f"Total images: {len(images)}")
    print(f"First image shape: {images[0].shape}")
    print(f"First label: {labels[0]}")
    print(f"First path: {paths[0]}")

    label_counts = Counter(labels)
    print(f"PD samples: {label_counts[1]}")
    print(f"Healthy samples: {label_counts[0]}")

    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F

    # 🔍 Visualize multiple augmented versions of one image
    def show_augmented_versions(image_path, transform, num_versions=5):
        fig, axes = plt.subplots(1, num_versions, figsize=(15, 3))
        for i in range(num_versions):
            img = Image.open(image_path).convert('L')
            augmented = transform(img)
            img_display = F.to_pil_image(augmented)  # tensor → image
            axes[i].imshow(img_display, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Augmented {i+1}')
        plt.tight_layout()
        plt.show()

    # ⚡ Optional: enable to preview augmentations
    if False:  # change to True if you want to visualize
        sample_path = paths[0]
        show_augmented_versions(sample_path, augment_transform)


