# 🌿 Multi-Crop Plant Leaf Nutrient Deficiency Classification

This project identifies nutrient deficiencies in the leaves of banana, rice, and citrus crops using deep learning models. By training individual CNNs and combining them with an ensemble approach, we achieve high accuracy and model interpretability using Grad-CAM visualizations. This work aims to support farmers and agronomists with actionable insights.

---

## 📌 Project Highlights

- 🧠 Trained CNN models: EfficientNetB0, ResNet50, DenseNet121
- 🔀 Used **mixup augmentation** for regularization
- 📈 Achieved **>90% classification accuracy** with ensemble
- 🔍 Interpretability using **Grad-CAM**
- 📊 Performance comparisons and confusion matrices
- 💻 Scripts built with **PyTorch**, visualizations with **Matplotlib**, metrics from **scikit-learn**

---


## 🧠 Models Used

| Model          | Notes                             |
|----------------|-----------------------------------|
| EfficientNetB0 | Used Mixup augmentation           |
| ResNet50       | Trained with standard technique   |
| DenseNet121    | Trained with standard technique   |
| Ensemble       | Averaged predictions from above   |

---

## 🔗 Datasets Used

⚠️ Due to size limitations, datasets are not included. Please download them from the sources below:

| Crop   | Dataset & Source                                                                                   | Description                              |
|--------|-----------------------------------------------------------------------------------------------------|------------------------------------------|
| Banana | [Kaggle](https://www.kaggle.com/datasets/smaranjitghose/banana-leaf-nutrient-deficiency-dataset)   | 8 deficiency classes + 1 healthy         |
| Rice   | [Mendeley](https://data.mendeley.com/datasets/y4wz6bx6yy/1)                                        | Nitrogen, Phosphorus, Potassium classes |
| Citrus | [Kaggle](https://www.kaggle.com/datasets/muhammedkalan/citrus-nutrient-deficiency)                 | Multiple annotated citrus leaf images   |

---

📊 Sample Output

| Model          | Accuracy  | Precision | Recall   | F1-Score |
| -------------- | --------- | --------- | -------- | -------- |
| EfficientNetB0 | 90.1%     | 0.91      | 0.90     | 0.90     |
| ResNet50       | 89.3%     | 0.90      | 0.89     | 0.89     |
| DenseNet121    | 91.0%     | 0.92      | 0.91     | 0.91     |
| Ensemble       | **92.4%** | **0.93**  | **0.92** | **0.92** |

---

🧠 Techniques Used
📦 PyTorch: CNN model development and training

📊 Matplotlib / Seaborn: Metrics & plots

🔬 Grad-CAM: Model explainability

📁 ImageFolder: For loading datasets

🧪 Soft Voting Ensemble: Averaging predictions across models

👨‍💻 Author
Akshay D Shetty
📧 akshay18shetty@gmail.com
🌐 insightbyakshay.in
🔗 www.linkedin.com/in/akshay18shetty


