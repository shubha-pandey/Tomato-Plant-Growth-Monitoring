# REAL TIME PREDICTION
# CNN + YOLO


import cv2
import numpy as np
import math
from tensorflow import keras
from ultralytics import YOLO
from sort import Sort     # Tracking


# Load YOLO model
yolo_model = YOLO(r"D:\Tomato Plant Growth Monitoring Project\tomato_3classes_2000images_21_4_2023.pt")
print("YOLO model loaded successfully!")

# YOLO labels
yolo_labels = ["Tomato Fully-ripe", "Tomato Semi-ripe", "Tomato Unripe"]  

# Load CNN model
cnn_model = keras.models.load_model(r"D:\Tomato Plant Growth Monitoring Project\tomato_model2_results\Models\fine_tuned_tomato_classifier.keras")
print("CNN model loaded successfully!")

# CNN labels
cnn_labels = ["Flowering", "Semi-Ripe", "Raw", "Ripe"]


# Mapping YOLO to CNN labels
category_mapping = {
    "Ripe": "Tomato Ripe",
    "Tomato Fully-ripe": "Tomato Ripe",
    "Semi-Ripe": "Tomato Semi-ripe",
    "Tomato Semi-ripe": "Tomato Semi-ripe",
    "Raw": "Tomato Unripe",
    "Unripe": "Tomato Unripe",
    "Tomato Unripe": "Tomato Unripe",
    "Unsure": "Unsure"
}

# Bounding box colors
colors = {
    "Tomato Fully-ripe": (0, 0, 255),  # Red
    "Tomato Semi-ripe": (0, 255, 255),  # Yellow
    "Tomato Unripe": (0, 255, 0)  # Green
}


# Object Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Track counts
total_count = set()
category_counts = {"Tomato Ripe": 0, "Tomato Semi-ripe": 0, "Tomato Unripe": 0}
track_category = {}


# Confidence threshold
CONFIDENCE_THRESHOLD = 0.7

# HSV Color Ranges for Filtering (Tomato-Specific Colors)
LOWER_RED = np.array([0, 100, 50])
UPPER_RED = np.array([10, 255, 255])

LOWER_ORANGE = np.array([10, 150, 100])  
UPPER_ORANGE = np.array([20, 255, 255])

LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([30, 255, 255])

LOWER_GREEN = np.array([35, 100, 100])
UPPER_GREEN = np.array([85, 255, 255])


# Function to filter colors
def is_tomato_color(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    red_mask = cv2.inRange(hsv, LOWER_RED, UPPER_RED)
    yellow_mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    orange_mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)  

    tomato_pixels = np.count_nonzero(red_mask) + np.count_nonzero(yellow_mask) + np.count_nonzero(green_mask) + np.count_nonzero(orange_mask)
    total_pixels = crop.shape[0] * crop.shape[1]

    return (tomato_pixels / total_pixels) > 0.3    # At least 30% of the crop must be tomato-colored


# Function to filter size
def is_valid_tomato(x1, y1, x2, y2, img_width, img_height):
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / height

    # **Ignore detections that are too small or have a weird shape**
    #if width < 30 or height < 30:  # Avoid detecting small objects
        #return False
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:    # Avoid detecting weirdly shaped objects
        return False
    if width > img_width * 0.5 or height > img_height * 0.5:    # Ignore objects that are too large 
        return False

    return True


# Function to preprocess images for CNN
def preprocess_img(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Function to combine YOLO and CNN predictions
def combine_yolo_cnn(yolo_label, yolo_confidence, cnn_result):
    cnn_confidence = np.max(cnn_result)
    cnn_class_id = np.argmax(cnn_result)  # Highest probability class
    cnn_label = cnn_labels[cnn_class_id]

    if yolo_confidence > 0.8:
        return category_mapping.get(yolo_label, yolo_label)
    elif cnn_confidence > 0.9:
        return category_mapping.get(cnn_label, cnn_label)
    else:
        return "Unsure"


# Open Camera (Real-Time)
vid = cv2.VideoCapture(0)   # 0 = Default camera, change to 1,2,.. for external cameras

while True:
    isTrue, frame = vid.read()
    if not isTrue:
        print("Camera not detected or error in reading frame.")
        break  
    
    frame = cv2.resize(frame, (980, 420))    # change accordingly
    img_height, img_width, _ = frame.shape   # Get image dimensions

    # Run YOLO detection
    results = yolo_model(frame, stream=True)
    detections = np.empty((0,5))
    refined_labels = {}  # Store refined labels for tracking
    
    # Process YOLO detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0]
            if conf < CONFIDENCE_THRESHOLD:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            yolo_label = yolo_labels[cls]
            color = colors.get(yolo_label, (255, 255, 255))

            # Filter out non-tomato objects
            if not is_valid_tomato(x1, y1, x2, y2, img_width, img_height):
                continue

            # Crop and classify with CNN
            tomato_crop = frame[y1:y2, x1:x2]

            # Apply Color Filtering
            if not is_tomato_color(tomato_crop):
                continue  # Skip non-tomato-colored detections
        
            if tomato_crop.size != 0:
                cnn_result = cnn_model.predict(preprocess_img(tomato_crop))
                refined_label = combine_yolo_cnn(yolo_label, conf, cnn_result)
            else:
                refined_label = yolo_label   # Use YOLO label if crop is empty

            # Store refined label for tracking
            refined_labels[(x1, y1, x2, y2)] = refined_label

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  
            label_text = f"{refined_label} {conf:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

    # Object tracking
    track_results = tracker.update(detections)
    for result in track_results:
        x1, y1, x2, y2, Id = map(int, result)

        if Id not in total_count:
            total_count.add(Id)

            # Get the refined label based on bounding box
            detected_category = refined_labels.get((x1, y1, x2, y2), "Unsure")

            # Ensure detected class name exists in category_counts
            if detected_category in category_counts:
                category_counts[detected_category] += 1
            else:
                print(f" Warning: {detected_category} not found in category_counts!")

            track_category[Id] = detected_category


    # Flower Detection Using CNN
    image_resized = preprocess_img(frame)
    cnn_overall_result = cnn_model.predict(image_resized)
    overall_class = np.argmax(cnn_overall_result)
    flower_detected = cnn_labels[overall_class] == 'Flowering'


    # Display Counts
    cv2.putText(frame, f"Total: {len(total_count)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, f"Unripe: {category_counts['Tomato Unripe']}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f"Semi-Ripe: {category_counts['Tomato Semi-ripe']}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f"Fully-Ripe: {category_counts['Tomato Ripe']}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


    # Print Counts
    total_tomatoes = sum(category_counts.values())

    if total_tomatoes > 0:
        print("\nTomato Counts:")
        for label, count in category_counts.items():
            print(f"- {count} {label}")
        print(f"Total Tomatoes: {total_tomatoes}")
    else:
        print("No tomatoes detected.")

    # Print flower detection message
    if flower_detected:
        print("\nFlowers detected!")

    # Show frame
    cv2.imshow('Real-Time Tomato Detection', frame)
    
    # Quit on 'x'
    if cv2.waitKey(20) & 0xFF == ord('x'):
        break


vid.release()
cv2.destroyAllWindows()
