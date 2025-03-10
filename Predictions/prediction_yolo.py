# SIMPLE YOLO MODEL PREDICTION


#import libraries
from ultralytics import YOLO
import cv2
import supervision as sv
import time

#paths
img_path = r"D:\Tomato Plant Growth Monitoring Project\Dataset\Test\IMG_20250206_113229.jpg"
model_path = r"D:\Tomato Plant Growth Monitoring Project\tomato_3classes_2000images_21_4_2023.pt"

#read image
img = cv2.imread(img_path)
if img is None:
    print(f"Error reading file at {img_path}")

# resize image
#img = cv2.resize(img, (640, 440))
#img = cv2.resize(img, (1240, 720))
#img = cv2.resize(img, (1220, 740))
img = cv2.resize(img, (224, 224))

# load model
model = YOLO(model_path)
print("Model loaded successfully") 

# make predictions
result = model(img, show=True)
 
cv2.waitKey(0)