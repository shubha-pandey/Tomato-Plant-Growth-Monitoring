# AUGMENTATION


# import libraries
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


# input and output paths
input_dirs = {
    "flowering": "D:/Tomato Plant Growth Monitoring Project/Dataset/Train01/Flowering",
    "semi_ripe": "D:/Tomato Plant Growth Monitoring Project/Dataset/Train01/Fruiting/Half-Ripe",
    "ripe": "D:/Tomato Plant Growth Monitoring Project/Dataset/Train01/Fruiting/Ripe",
    "raw": "D:/Tomato Plant Growth Monitoring Project/Dataset/Train01/Fruiting/Raw"
}

output_dirs = {
    "flowering": "D:/Tomato Plant Growth Monitoring Project/Dataset/Train01/Flowering",
    "semi_ripe": "D:/Tomato Plant Growth Monitoring Project/Dataset/Train01/Fruiting/Half-Ripe",
    "ripe": "D:/Tomato Plant Growth Monitoring Project/Dataset/Train01/Fruiting/Ripe",
    "raw": "D:/Tomato Plant Growth Monitoring Project/Dataset/Train01/Fruiting/Raw"
}


# Create output directories if they don't exist
for output_dir in output_dirs.values():
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created directory {output_dir}")


# Augmentation configuration
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)


# Function to augment images until the target count is reached
def augment_images(input_dir, output_dir, target_count=1000):     # set the target count for the required number of images, generally to balance images of all categories
    existing_images = len(os.listdir(output_dir))     # Count existing images
    print(f"Existing images in {output_dir}: {existing_images}")
    
    if existing_images >= target_count:     # check if the images sufficient
        print(f"Target count of {target_count} already reached for {output_dir}. No augmentation needed.")
        return

    needed_images = target_count - existing_images     # the number of images to be generated
    print(f"Need to generate {needed_images} additional images for {output_dir}")
    
    image_files = os.listdir(input_dir)
    total_images = len(image_files)
    if total_images == 0:
        print(f"No images found in {input_dir}. Skipping augmentation.")
        return

    while needed_images > 0:
        for img_name in image_files:
            if needed_images <= 0:
                break

            print(f"Processing: {img_name}")
            img_path = os.path.join(input_dir, img_name)
            img = load_img(img_path)  # Load the image
            img_array = img_to_array(img)  # Convert to an array
            img_array = img_array.reshape((1,) + img_array.shape)  # Reshape for datagen

            # Generate augmented images
            for batch in datagen.flow(
                img_array,
                batch_size=5,
                save_to_dir=output_dir,
                save_prefix=f"aug_{img_name.split('.')[0]}",
                save_format="jpg"
            ):
                needed_images -= 1
                if needed_images <= 0:
                    break

    print(f"Augmentation completed for {output_dir}. Total images: {len(os.listdir(output_dir))}")


# Augment images for each category
target_count = 900     # specify a different count if needed

for category, input_dir in input_dirs.items():
    print(f"\nAugmenting category: {category}")
    augment_images(input_dir, output_dirs[category], target_count) 