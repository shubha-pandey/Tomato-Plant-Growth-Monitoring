# CNN MODEL PREDICTION OF VIDEO FILES


# import libraries
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from ultralytics import YOLO


# Load the trained model
model = tf.keras.models.load_model(r"D:\Tomato Plant Growth Monitoring Project\tomato_model2_results\Models\fine_tuned_tomato_classifier.keras") 
#model = YOLO("D:/GitHub/tomato_detection/weight/tomato_3classes_2000images_21_4_2023.pt") 

# Create a list of class labels (disease names) used during training
class_labels = ['Fully-Ripe', 'Semi-Ripe', 'Unripe']  # Corrected class name

# Initialize the camera
cap = cv2.VideoCapture(r"D:\Tomato Plant Growth Monitoring Project\Dataset\Test\test_video1.mp4")  # Use 0 for the default camera, change it if using an external camera

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error capturing frame")
        break
    
    # Resize the frame to match the input size of your model
    img = cv2.resize(frame, (224, 224))
    
    # Preprocess the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize the image data
    
    # Make predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    
    # Get the disease name from the class_labels list
    result = class_labels[predicted_class[0]]
    
    # Display the image and disease name
    cv2.imshow("Tomato Detection", frame)
    print(f'Predicted Outcome: {result}')
    break
    
    # Press 'q' to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()