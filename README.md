🗂️ Dataset Structure
The dataset is organized into four subfolders under two main classes:
/dataset/

├── PD/  
→ Label: 1
│   ├── wave/
│   └── spiral/
└── Healthy/    
→ Label: 0
    ├── wave/
    └── spiral/
    


Each subfolder contains static handwriting images in .jpg, .jpeg, or .png format.
The preprocessing script performs the following steps:
- Recursively loads all images from the dataset
- Converts each image to RGB (if grayscale)
- Resizes to 224×224 pixels
- Normalizes using ImageNet mean and standard deviation
- Converts each image to a PyTorch tensor of shape [3, 224, 224]
- Assigns labels based on folder name:
- PD → 1
- Healthy → 0


After running the script, you’ll get:
- images: a list of PyTorch tensors
- labels: a list of integers (0 or 1)
- paths: a list of full image file paths

🚀 How to Run
Update the dataset path in the script:
base_dir = r"D:\dataset"  # or use forward slashes: "D:/dataset"
Then run the script:
python Preprocessing.py


📊 Sample Output
Total images: 204
PD samples: 102
Healthy samples: 102
First image shape: torch.Size([3, 224, 224])
First label: 1
First path: D:\dataset\PD\wave\V01PO01.png

