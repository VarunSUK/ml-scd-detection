# 🧠 Sickle Cell Disease Detection from MRI Brain Scans

A deep learning project using **TensorFlow/Keras** to classify MRI brain scans for **early detection of Sickle Cell Disease (SCD)**. This model supports research in **clinical screening, medical imaging analysis**, and **health equity**.

---

## 🔬 Project Overview

This project leverages Convolutional Neural Networks (CNNs) and transfer learning techniques to analyze and classify MRI scans for early signs of Sickle Cell Disease. Built on a Kaggle dataset of over **3,000 labeled brain MRI images**, the model achieves **87% test accuracy**, making it a promising tool for diagnostic assistance and academic research.

---

## 🚀 Key Features

- ✅ **Achieved 87% test accuracy** on real-world MRI scan data
- 🧠 Built a custom **CNN architecture** using TensorFlow/Keras
- 🔁 Fine-tuned **ResNet** and **MobileNet** using **transfer learning**
- 📈 Applied **preprocessing**: resizing, normalization, data augmentation
- 🧪 Designed a **flexible pipeline** for model training and evaluation
- 🌍 Supports **health equity research** and potential deployment in low-resource settings

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- OpenCV for image processing
- Transfer learning (ResNet50, MobileNetV2)
- Scikit-learn (for evaluation metrics)

---

## 📊 Model Workflow

1. **Data Collection**  
   - Kaggle dataset with over 3,000 labeled MRI scans (SCD-positive and negative)

2. **Preprocessing**
   - Image resizing to 224x224
   - Normalization to [0,1] pixel scale
   - Data augmentation (rotation, flip, zoom) to reduce overfitting

3. **Model Training**
   - Baseline CNN from scratch
   - Fine-tuned pretrained models (ResNet50, MobileNetV2)
   - Early stopping, checkpointing, and learning rate scheduling

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix and ROC curve
   - Grad-CAM for model interpretability (optional)

---

## 📁 Project Structure

