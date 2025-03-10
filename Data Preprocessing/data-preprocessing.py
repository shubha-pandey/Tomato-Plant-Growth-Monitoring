# RENAME, RESIZE, AND NORMALIZE 

# Import libraries
import os
import cv2 as cv


# input and output paths
input_dirs = {
    "flowering": "D:/Tomato Plant Growth Monitoring Project/Dataset/Train01/Flowering",
    "semi_ripe": "D:/Tomato Plant Growth Monitoring Project/Dataset/Train01/Fruiting/Half-Ripe",
    "ripe": "D:/Tomato Plant Growth Monitoring Project/Dataset/Train01/Fruiting/Ripe",
    "raw": "D:/Tomato Plant Growth Monitoring Project/Dataset/Train01/Fruiting/Raw"
}

output_dirs = {
    "flowering": "D:/Tomato Plant Growth Monitoring Project/Dataset/Data/Flowering",
    "semi_ripe": "D:/Tomato Plant Growth Monitoring Project/Dataset/Data/Fruiting/Half-Ripe",
    "ripe": "D:/Tomato Plant Growth Monitoring Project/Dataset/Data/Fruiting/Ripe",
    "raw": "D:/Tomato Plant Growth Monitoring Project/Dataset/Data/Fruiting/Raw"
}


# Ensure output directories exist
for category, path in output_dirs.items():
    os.makedirs(path, exist_ok=True)
    print(f"\nCreated directory {category} at {path}.")


# Function to resize and normalize image
def preprocess_image(input_path, output_path, size=(224, 224)):
    img = cv.imread(input_path)     # read image
    if img is None:
        print(f"\nError reading image: {input_path}")
        return False
    img_resized = cv.resize(img, size)
    img_normalized = img_resized / 255.0     # Normalize pixel values
    cv.imwrite(output_path, (img_normalized * 255).astype('uint8'))     # Save normalized image
    return True


# Function to rename, resize and normalize images
def process_category(input_dir, output_dir, category):
    images = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"\nProcessing category: {category}")
    for idx, img_name in enumerate(images):
        new_filename = f'{category}_{idx:02d}.jpg'     # Sequential renaming
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, new_filename)

        if not preprocess_image(input_path, output_path):
            print(f"\nFailed to process {img_name}")
            continue

    print(f"Processing completed for category: {category}")


# Process each category
for category, input_dir in input_dirs.items():
    process_category(input_dir, output_dirs[category], category)
    
