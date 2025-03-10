# YOLO PREDICTIONS OF VIDEO FILES


# import libraries
from ultralytics import YOLO
import cv2
import numpy as np
import math
import cvzone
from sort import Sort

# Load video 
vid = cv2.VideoCapture(r"D:\Tomato Plant Growth Monitoring Project\Dataset\Test\test_video1.mp4")
print('Video is processed!', vid)

#Load YOLO model
model = YOLO(r"D:\Tomato Plant Growth Monitoring Project\tomato_3classes_2000images_21_4_2023.pt")
print("Model loaded successfully!")

classNames = ["Fully-Ripe", "Semi-Ripe", "Unripe"]

colors = {
    "Fully-Ripe" : (0, 0, 255),
    "Semi-Ripe" : (0, 255, 255),
    "Unripe" : (0, 255, 0)
}

#tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

total_count = set()
category_counts = {"Fully-Ripe": 0, "Semi-Ripe": 0, "Unripe": 0}

track_category = {}


while True:
    isTrue, frame = vid.read()

    if not isTrue:
        print("Video finished or cannot load frame.")
        break  

    frame = cv2.resize(frame, (980, 420))

    # Run YOLO model
    results = model(frame, stream=True)

    detections = np.empty((0,5))

    # Loop through detections and annotate frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0]
            if conf < 0.4 :
                continue

            # BOUNDING BOX
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence and class
            conf = math.ceil((box.conf[0] * 100)) / 100  # Round confidence to 2 decimals
            cls = int(box.cls[0])
            class_name = classNames[cls]

            color = colors.get(class_name, (255, 255, 255))  # Default to white if class not found


            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  
            #cv2.putText(frame, f"{class_name} {conf}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)

            #cvzone.cornerRect(frame, (x1, y1, w, h), l=6, rt=4, colorR=color, colorC=(255, 0, 0))
            cvzone.putTextRect(frame, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=0.75, thickness=1, colorT=(0, 0, 0), colorR=color, offset =4)

            currentArr = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, currentArr))

    
    track_results = tracker.update(detections)

    for result in track_results:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #print(result)
        #w, h = x2-x1, y2-y1
        Id = int(Id)

        if Id not in total_count:
            total_count.add(Id)
            
            for box in boxes:
                if box.xyxy[0][0] == x1 and box.xyxy[0][1] == y1 and box.xyxy[0][2] == x2 and box.xyxy[0][3] == y2:
                    cls = int(box.cls[0])
                    class_name = classNames[cls]
                    break

            category_counts[class_name] += 1
            track_category[Id] = class_name


    
    cvzone.putTextRect(frame, f"Total: {len(total_count)}", (50, 50), scale=1, thickness=1, colorR=(0, 0, 0), offset=5)
    cvzone.putTextRect(frame, f"Unripe: {category_counts['Unripe']}", (50, 70), scale=0.6, thickness=1, colorR=(0, 0, 0), offset=4)
    cvzone.putTextRect(frame, f"Semi-Ripe: {category_counts['Semi-Ripe']}", (50, 80), scale=0.6, thickness=1, colorR=(0, 0, 0), offset=4)
    cvzone.putTextRect(frame, f"Fully-Ripe: {category_counts['Fully-Ripe']}", (50, 90), scale=0.6, thickness=1, colorR=(0, 0, 0), offset=4)

    # Display the frame with annotations
    cv2.imshow('Tomato Detection', frame)

    # Quit on 'x'
    if cv2.waitKey(20) & 0xFF == ord('x'):
        break

# Release resources
vid.release()
cv2.destroyAllWindows()
