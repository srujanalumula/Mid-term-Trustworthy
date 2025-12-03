

# **White Blood Cell Classification Using Deep Learning & Trustworthiness Evaluation**

A complete research-grade framework for automated **White Blood Cell (WBC) classification**, combined with rigorous **trustworthiness assessments**, including robustness testing, Fisher Information Matrix (FIM), Influence Functions (IF), and explainability using Grad-CAM.

This repository contains the entire workflow—from dataset preparation and model training to sensitivity analysis, robustness evaluation, and UI deployment.

---

# **1. Project Overview**

This project presents a comprehensive deep learning pipeline for accurate and trustworthy **White Blood Cell classification** using microscopy images. The system integrates:

* A custom high-capacity CNN
* Transfer learning models (VGG16, ResNet50)
* Robustness and sensitivity evaluation
* Explainability with Grad-CAM
* Streamlit deployment for real-time use

The goal is to build an AI system that is **accurate**, **robust**, **interpretable**, and **clinically reliable**.

---

# **2. Project Goals**

The objectives of this project are:

### **Model Performance**

* Train and evaluate three deep learning architectures
* Achieve high accuracy across 5 WBC classes

### **Trustworthiness**

* Analyze model robustness under perturbations
* Identify sensitive and unstable samples using FIM
* Detect harmful or influential training samples using Influence Functions
* Provide localized visual explanations using Grad-CAM

### **Deployment**

* Provide a user-friendly Streamlit interface
* Enable real-time WBC classification with heatmap visualization

---

# **3. Dataset Details**

### **Source**

White Blood Cells Dataset – *Masoud Nickparvar* (Kaggle)

### **Classes**

1. Neutrophil
2. Lymphocyte
3. Monocyte
4. Eosinophil
5. Basophil

### **Dataset Size**

| Split      | Images     |
| ---------- | ---------- |
| Training   | 10,175     |
| Validation | 2,035      |
| Test       | 4,339      |
| **Total**  | **14,514** |

### **Preprocessing**

* RGB conversion
* Resize

  * CNN: 128×128
  * VGG16/ResNet50: 224×224
* Normalize to [0,1]
* One-hot encoding
* Stratified dataset split

### **Augmentation**

* Rotation (±20°)
* Zoom (20%)
* Horizontal/vertical flips
* Brightness variation
* Width/height shift

---

# **4. Environment Setup**

Install all dependencies:

```
pip install --upgrade pip
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn pillow opencv-python tqdm kagglehub streamlit
```

GPU support (optional):

```
pip install tensorflow-gpu
```

---

# **5. Folder Structure**

```
root/
│── models/
│     ├── best_wbc_model.keras
│     ├── vgg16_wbc_model.keras
│     └── resnet50_wbc_model.keras
│
│── figures/
│     ├── fisher_scores.png
│     ├── influence_curve.png
│     ├── robustness_accuracy.png
│     ├── gradcam_correct.png
│     └── gradcam_wrong.png
│
│── notebooks/
│     ├── Training_Pipeline.ipynb
│     └── Trustworthiness_Evaluation.ipynb
│
│── app_streamlit.py
│── utils/
│── README.md
```

---

# **6. Models Trained**

Three deep learning models were evaluated:

### **1. Custom CNN (Designed from scratch)**

* Multi-stage convolutional blocks
* Batch normalization
* Dropout regularization
* 1024 + 512 dense layers

### **2. VGG16**

* Pretrained on ImageNet
* Frozen convolution layers
* Lightweight classifier head

### **3. ResNet50**

* Deep residual architecture
* Best generalization ability
* Best robustness

---

# **7. Performance Metrics (Accuracy & Validation)**

### **Test Accuracy**

| Model      | Accuracy |
| ---------- | -------- |
| Custom CNN | ~95%     |
| VGG16      | ~97%     |
| ResNet50   | **~98%** |

### **Validation Curves**

* Stable convergence
* Limited overfitting due to augmentation
* VGG16/ResNet50 outperform custom CNN

---

# **8. Confusion Matrix & Classification Metrics**

Detailed classification report includes:

* Precision
* Recall
* F1-score
* Support per class

Confusion matrix visualizations illustrate:

* High accuracy for Neutrophils/Lymphocytes
* Moderate confusion between Monocytes & Eosinophils
* Basophils remain challenging due to limited data

---

# **9. Robustness Evaluation (Trustworthiness)**

Robustness tests were conducted under:

* **Gaussian Noise**
* **Gaussian Blur**
* **Brightness Variation**

### **Key Findings**

| Distortion | CNN    | VGG16 | ResNet50      |
| ---------- | ------ | ----- | ------------- |
| Noise      | Medium | High  | **Very High** |
| Blur       | Medium | High  | **Very High** |
| Brightness | High   | High  | **Very High** |

### **Robustness Accuracy Visual**

```
![Robustness Accuracy](figures/robustness_accuracy.png)
```

ResNet50 demonstrates the strongest robustness across all perturbations.

---

# **10. Explainability Evaluation (Grad-CAM)**

Grad-CAM visualizations highlight decision-critical regions.

### **Correct Prediction**

```
![GradCAM Correct](figures/gradcam_correct.png)
```

Model focuses on nucleus morphology.

### **Incorrect Prediction**

```
![GradCAM Wrong](figures/gradcam_wrong.png)
```

Diffuse, misaligned activation indicates uncertainty.

---

# **11. Fisher Information Matrix (Sensitivity)**

The FIM quantifies prediction confidence and local stability.

### **Formula**

```
g = p – y
Fisher Score = || g ||²
```

### **Interpretation**

* **Low FIM** → Stable prediction
* **High FIM** → Uncertain, boundary-case image

### **Fisher Score Distribution**

```
![Fisher Score Distribution](figures/fisher_scores.png)
```

High-FIM samples often contain:

* Poor contrast
* Partial cells
* Atypical shapes

---

# **12. Influence Functions (Training-Sample Impact)**

Influence Functions show how each training image affects predictions.

### **Formula (approx.)**

```
Influence = – ( g_train · g_test )
```

### **Meaning**

* Positive → Supports prediction
* Negative → Confuses or harms prediction

### **Influence Curve**

```
![Influence Curve](figures/influence_curve.png)
```

Harmful samples often indicate label noise or severe distortion.

---

# **13. Cross-Dataset Generalization (PlantDoc)**

The model was optionally tested on cross-domain samples such as **PlantDoc** to explore generalization limits.

Findings:

* CNN generalizes moderately
* ResNet50 retains better feature extraction
* Domain shift significantly reduces accuracy → expected behavior

---

# **14. Why the Custom CNN is Competitive**

Even though ResNet50 achieves the highest accuracy, the custom CNN has advantages:

* Lightweight
* Faster inference
* Requires less training time
* Performs competitively for its size

However, transfer learning architectures outperform it in robustness and trustworthiness.

---

# **15. Reliability & Limitations**

### **Reliability Strengths**

* High accuracy
* Excellent robustness (ResNet50)
* Strong nucleus-focused explainability

### **Limitations**

* Basophil class is underrepresented
* Sensitivity varies for low-contrast cells
* Influence analysis depends on gradient quality

---

# **16. Download Final Model**

All trained models are available in:

```
/models/
```

Including:

* best_wbc_model.keras
* vgg16_wbc_model.keras
* resnet50_wbc_model.keras

---

# **17. Streamlit Deployment**

### **Run Locally**

```
streamlit run app_streamlit.py
```

### **Features**

* Upload image
* Predict class & confidence
* Show probability distribution
* Generate Grad-CAM visualization

---

# **18. Run the Application**

Place the model file in the project directory then run:

```
streamlit run app_streamlit.py
```

Browser will open automatically.

---

# **19. Docker Deployment**

(Dockerfile can be added)

Example:

```
docker build -t wbc_app .
docker run -p 8501:8501 wbc_app
```

---

# **20. Test the Model**

After training:

```
python test_model.py --image sample.png
```

or use the Streamlit UI.

---

# **21. Workflow Summary (Step-by-Step)**

1. Download dataset
2. Preprocess & augment images
3. Train CNN, VGG16, ResNet50
4. Evaluate performance
5. Conduct trustworthiness tests:

   * FIM
   * Influence Functions
   * Robustness curves
   * Grad-CAM
6. Deploy final model via Streamlit

---

# **22. Quick Commands Summary**

```
pip install -r requirements.txt
python train.py
python evaluate.py
streamlit run app_streamlit.py
```

---

# **23. Troubleshooting**

| Issue                  | Solution                        |
| ---------------------- | ------------------------------- |
| TensorFlow DLL error   | Install correct TF version      |
| Grad-CAM heatmap blank | Ensure last conv layer exists   |
| Low GPU memory         | Reduce batch size               |
| Model not loading      | Ensure correct .keras file path |

---

# **24. Future Enhancements**

* Self-supervised pretraining
* Attention-based explainability (ViT or Grad-CAM++)
* Balanced dataset creation
* Federated learning for privacy-preserving WBC classification
* Adversarial robustness testing (FGSM, PGD)

---

# **25. References**

* Nickparvar, Masoud. *White Blood Cells Dataset*, Kaggle
* Selvaraju et al., "Grad-CAM"
* Koh & Liang, "Influence Functions"
* Goodfellow et al., "Deep Learning"

---

# **26. Final Summary**

This repository presents a complete trustworthy-AI framework for WBC classification using deep learning. Beyond achieving high accuracy, the system incorporates:

* **Robustness evaluation** under real-world distortions
* **Sensitivity measurement** via Fisher Information Matrix
* **Training-sample introspection** via Influence Functions
* **Clinically interpretable heatmaps** via Grad-CAM

With strong model performance and comprehensive trustworthiness analysis, this framework represents a reliable, practical foundation for medical imaging applications.



Just tell me.
