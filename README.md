# 🍅 Tomato Plant Growth Monitoring
A deep learning and computer vision-based system for automating tomato plant growth monitoring. This project detects plant growth stages, classifies fruit ripeness, and estimates yield, helping farmers optimize crop management and make data-driven decisions.

## 🌱 Project Overview
This system leverages YOLO (You Only Look Once) and MobileNetV2 to monitor tomato plant growth. It performs real-time plant stage detection, fruit ripeness classification, and yield estimation using object detection techniques.

## 📌 Key Features
- **Plant Growth Stage Detection** – Identifies whether a plant is in the flowering, or fruiting stage.
- **Fruit Ripeness Classification** – Categorizes tomatoes as unripe, semi-ripe, or fully ripe using deep learning models.
- **Real-Time Object Detection** – Counts tomatoes at different stages and visualizes results with bounding boxes.
- **Camera Integration** – Tracks plant growth continuously with live monitoring.
- **Data-Driven Insights** – Helps farmers optimize farming practices and reduce manual effort and time.

  
## 📁 Dataset  
The dataset contains images of tomato fruits at different stages (ripe, unripe, semi-ripe) and tomato flowers. It has been collected from Kaggle, [GitHub](https://github.com/laboroai/LaboroTomato?tab=readme-ov-file#dataset-details), and [other sources](https://redu.unicamp.br/dataset.xhtml?persistentId=doi:10.25824/redu/EP4NGO).  

### 📂 Dataset Structure  

```
data/
│   ├── Flowering/
│   └── Fruiting/
│       ├── Ripe/
│       ├── Semi-Ripe/
│       └── Raw/
```

- **Flowering:** Flowers are visible, indicating the plant is in the reproductive phase.  
- **Fruiting:** Tomatoes are categorized based on ripeness:  
  - **Ripe:** Completely red, **ready to harvest** (90%+ red).  
  - **Semi-Ripe:** Greenish, needs time to ripen (**30-89% red**).  
  - **Raw:** Mostly green or white, sometimes with **rare red parts** (**0-30% red**). 

Find the final processed dataset here on [Kaggle](https://www.kaggle.com/datasets/shubhapandey/tomato-plant-dataset).


## 📊 Methodology

- **Preprocessing & Dataset Preparation** – Images of tomato plants are collected, preprocessed, and labeled for training.
- **Model Training** – YOLO & MobileNetV2 models are trained to classify growth stages and fruit ripeness.
- **Real-Time Inference** – The trained model analyzes video feeds or images to detect plant stages & count fruits.
- **Result Visualization** – Outputs are displayed with bounding boxes & classification labels in real time.


## 📌 Tools and Technologies

- Python 3.x
- Jupyter Notebook
- Ultralytics (YOLO)
- TensorFlow/Keras
- Scikit-Learn
- NumPy, Pandas, Matplotlib, OpenCV
- Kaggle


## 📂 Repository Structure

```
📁 Data Preprocessing        # Preprocessing codes for dataset preparation
📁 Predictions               # Prediction examples/samples
📁 __pycache__               # Cached Python files
📁 tomato_model1_results     # Initial model results: training/validation graphs, cross-validation, and model
📁 tomato_model2_results     # Fine-tuned model results: training/validation graphs, cross-validation, and model
📄 README.md                 # Readme file
📄 real-time-pred.py         # Real-time prediction using camera
📄 requirements.txt          # Requirements 
📄 sort.py                   # File sorter/tracker to count fruits in code
📄 tomato_3classes_2000images_21_4_2023.pt  # YOLO weight for tomato detection and prediction
📄 tomato_model1.ipynb       # Initial model building and training notebook
📄 tomato_model2.ipynb       # Fine-tuned model training notebook
📄 vid_predictions_cnn_yolo.py # Video prediction using both CNN and YOLO models
```


## 🚀 Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/tomato-growth-monitoring.git
   cd tomato-growth-monitoring
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run real-time prediction:**
   ```bash
   python real-time-pred.py
   ```
4. **Run video prediction:**
   ```bash
   python vid_predictions_cnn_yolo.py
   ```


## 🔗 References

- [sort file](https://github.com/abewley/sort/blob/master/sort.py)

---

