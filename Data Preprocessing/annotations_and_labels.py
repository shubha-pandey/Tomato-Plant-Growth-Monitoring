# CREATE ANNOTATIONS AND LABELS FILE

# import libraries
import os
import csv
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString


# Paths
output_dirs = {
    "flowering": "D:/Tomato Plant Growth Monitoring Project/Dataset/Data/Flowering",
    "semi_ripe": "D:/Tomato Plant Growth Monitoring Project/Dataset/Data/Fruiting/Half-Ripe",
    "ripe": "D:/Tomato Plant Growth Monitoring Project/Dataset/Data/Fruiting/Ripe",
    "raw": "D:/Tomato Plant Growth Monitoring Project/Dataset/Data/Fruiting/Raw"
}

annotations_dir = "D:/Tomato Plant Growth Monitoring Project/Dataset/Data/Annotations"
labels_csv_path = "D:/Tomato Plant Growth Monitoring Project/Dataset/Data/labels.csv"


# Ensure annotations directory exists
os.makedirs(annotations_dir, exist_ok=True)
print(f"Created directory at {annotations_dir}.")


# Open CSV file for writing
csv_file = open(labels_csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['filename', 'label'])     # CSV header


# Function to create annotation files
def create_annotation(image_name, category, bbox, image_size, output_dir):
    annotation = Element("annotation")
    folder = SubElement(annotation, "folder")
    folder.text = category

    filename = SubElement(annotation, "filename")
    filename.text = image_name

    size = SubElement(annotation, "size")
    width = SubElement(size, "width")
    height = SubElement(size, "height")
    depth = SubElement(size, "depth")
    width.text, height.text, depth.text = map(str, image_size)

    obj = SubElement(annotation, "object")
    name = SubElement(obj, "name")
    name.text = category

    bndbox = SubElement(obj, "bndbox")
    xmin, ymin, xmax, ymax = bbox
    SubElement(bndbox, "xmin").text = str(xmin)
    SubElement(bndbox, "ymin").text = str(ymin)
    SubElement(bndbox, "xmax").text = str(xmax)
    SubElement(bndbox, "ymax").text = str(ymax)

    # Save annotation
    annotation_file = os.path.join(output_dir, image_name.replace('.jpg', '.xml'))
    dom = parseString(tostring(annotation))
    with open(annotation_file, "w") as f:
        f.write(dom.toprettyxml(indent="  "))
    print(f"Saved annotation: {annotation_file}")

# Generate annotations and CSV entries
for category, category_dir in output_dirs.items():
    for img_name in os.listdir(category_dir):
        if not img_name.endswith('.jpg'):
            continue

        # Simulated bounding box (centered)
        image_size = (224, 224, 3)
        h, w = image_size[:2]
        xmin, ymin = w // 4, h // 4
        xmax, ymax = 3 * w // 4, 3 * h // 4

        # Save annotation
        create_annotation(
            image_name=img_name,
            category=category,
            bbox=[xmin, ymin, xmax, ymax],
            image_size=image_size,
            output_dir=annotations_dir
        )

        # Add entry to CSV with absolute paths
        csv_writer.writerow([os.path.abspath(os.path.join(category_dir, img_name)), category])


# Close CSV file
csv_file.close()
print(f"Labels CSV saved to: {labels_csv_path}")
