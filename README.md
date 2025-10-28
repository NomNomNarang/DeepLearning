# 🧠 Sonar Object Classification using Deep Learning  

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)

> Deep Learning model that classifies sonar signals as **Rock (R)** or **Mine (M)** using an Artificial Neural Network (ANN) built with TensorFlow and Keras.


## 📘 Project Overview

This project implements a **binary classification** neural network trained on the  
🎯 [Sonar Dataset (UCI Machine Learning Repository)](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)).  

Each instance contains 60 sonar signal features, and the target label identifies whether the object is a **metal cylinder (Mine)** or a **rock (Rock)**.  
Two models were built:
1. **Without Dropout** – baseline model.
2. **With Dropout Regularization** – for better generalization.

---

## 🧩 Model Architecture

### **Model 1 – Baseline ANN**

python
modeld = keras.Sequential([
    keras.layers.Dense(60, input_shape=(60,), activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

Model 2 – With Dropout Regularization
model = keras.Sequential([
    keras.layers.Dense(60, input_shape=(60,), activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])
⚙️ Training Configuration
Parameter	Value
Optimizer	Adam
Loss	binary_crossentropy
Metric	accuracy
Epochs	100
Batch Size	8
Validation Split	0.2

Data Split:

python
Copy code
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
🧾 Evaluation
The model was evaluated using Accuracy, Confusion Matrix, and Classification Report from
sklearn.metrics.

python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
📊 Example Output
markdown
              precision    recall  f1-score   support

           0       0.88      0.81      0.85        27
           1       0.71      0.80      0.75        15

    accuracy                           0.81        42
   macro avg       0.79      0.81      0.80        42
weighted avg       0.82      0.81      0.81        42
📉 Results Summary
Model	Train Accuracy	Test Accuracy	Observation
Without Dropout	~99–100%	~81%	Overfitting
With Dropout (0.5)	~70–75%	~80–82%	Better Generalization ✅


💡 Key Learnings
Dropout prevents overfitting by randomly deactivating neurons during training.

Lower training accuracy with Dropout indicates improved model generalization.

Evaluation beyond accuracy (Precision, Recall, F1-score) gives better insights into model performance.

CPU-based training is slow; using a GPU can significantly improve training time.

🧰 Tech Stack
Tool	Purpose
Python	Core Language
TensorFlow	Deep Learning Framework
Keras	High-level API for Neural Networks
NumPy	Numerical Computation
Pandas	Data Manipulation
Matplotlib	Data Visualization
Scikit-learn	Data Splitting & Metrics

📈 Future Improvements
Add Batch Normalization

Apply EarlyStopping to prevent overtraining

Perform Hyperparameter Tuning using KerasTuner

Use TensorBoard for training visualization

🧑‍💻 Author
👩‍💻 Namita Narang
🎓 Deep Learning & AI Enthusiast | Data Analyst in Progress
