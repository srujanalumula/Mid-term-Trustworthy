White Blood Cell Classification – CNN, VGG16 & ResNet50

A deep learning-based system for classifying white blood cell (WBC) images into five types using Custom CNN, VGG16, and ResNet50 architectures. The model is deployed as a Streamlit app for real-time classification.

🛠️ Technologies Used

Python, TensorFlow, Keras – Core ML framework

NumPy, Pandas, Pillow – Data handling & image processing

Matplotlib, Seaborn – Visualization

Scikit-learn – Evaluation metrics

Streamlit – User Interface

KaggleHub – Dataset management

📊 Dataset

Source: Kaggle – White Blood Cells Dataset

Classes: Neutrophil, Lymphocyte, Monocyte, Eosinophil, Basophil

Images: ~14,500 total (Train/Validation/Test split)

Input Sizes: 128×128 (CNN), 224×224 (VGG16/ResNet50)

🧠 Model Architectures

Custom CNN

4 Conv Blocks (64–512 filters) + BN + Dropout

Dense layers: 1024 → 512 → Softmax(5)

VGG16 & ResNet50 (Transfer Learning)

Pretrained on ImageNet

Top layers replaced with Dense → Dropout → Softmax
⚙️ Installation
pip install --upgrade pip
pip install tensorflow>=2.15 keras>=3.0.0 streamlit pillow numpy pandas matplotlib seaborn scikit-learn opencv-python tqdm kagglehub
Attribution
**Attribution:** Some dataset and model utility functions are adapted from open Kaggle sources: [White Blood Cells Dataset](https://www.kaggle.com/datasets/masoudnickparvar/white-blood-cells-dataset)
🚀 How to Run

1️⃣ Train Models (Optional)

Run Code_midtermcode.ipynb notebook to train and save:

best_wbc_model.keras

vgg16_wbc_model.keras

resnet50_wbc_model.keras

2️⃣ Launch Streamlit App

cd "D:\new project live imple\trustworthy"
streamlit run app_streamlit.py


or in IDLE: open run_streamlit.py → press F5

3️⃣ Use the App

Upload a WBC image (JPG/PNG)

Model Path → D:\new project live imple\trustworthy\best_wbc_model.keras

See predicted class, confidence, and Grad-CAM heatmap.

💾 Saved Models
File	Description
best_wbc_model.keras	Custom CNN
vgg16_wbc_model.keras	VGG16
resnet50_wbc_model.keras	ResNet50
