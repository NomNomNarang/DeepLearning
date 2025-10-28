# DeepLearning
My own notes so that you all guys can understand, ask any doubts free to help everyone.

#Drop out regualrization
🔬 Sonar Object Classification using Deep Learning

This project uses Artificial Neural Networks (ANN) to classify sonar signals as either Rock (R) or Mine (M).
The dataset used is the Sonar Dataset from the UCI Machine Learning Repository.
The model is built using TensorFlow and Keras, demonstrating how dropout regularization affects overfitting and model performance.

🧠 Objective

To build a binary classification neural network that predicts whether a sonar signal reflects off a metal cylinder (mine) or a rock, based on 60 input features extracted from sonar signals.

📁 Dataset

Dataset Name: Sonar Dataset

Source: UCI Machine Learning Repository

Shape: (207, 61) — 60 features + 1 target column

Target Variable:

R → Rock (1 after encoding)

M → Mine (0 after encoding)

⚙️ Libraries Used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

🧩 Data Preprocessing

Loaded the dataset using Pandas

Encoded the target variable (R, M) into numeric format (1, 0)

Split the data into:

Training set → 80%

Test set → 20%

🏗️ Model Architecture
Model 1 – Without Dropout
modeld = keras.Sequential([
    keras.layers.Dense(60, input_shape=(60,), activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


Optimizer: Adam

Loss Function: Binary Crossentropy

Metric: Accuracy

Epochs: 100

Batch Size: 8

Results:
Metric	Value
Training Accuracy	~100%
Test Accuracy	~81%
Precision (Class 0)	0.88
Recall (Class 1)	0.80

⚠️ The model shows signs of overfitting (training accuracy 100%, test accuracy lower).

🧪 Model 2 – With Dropout Regularization

To reduce overfitting, dropout layers were introduced after each Dense layer:

model = keras.Sequential([
    keras.layers.Dense(60, input_shape=(60,), activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

Results:

Reduced overfitting

More stable training

Lower accuracy than overfitted model, but better generalization

📊 Evaluation Metrics
1. Accuracy

Measures the overall correctness of the model.

2. Precision

Out of all samples predicted as “Mine” or “Rock”, how many were actually correct?

3. Recall

Out of all actual “Mine” or “Rock” samples, how many were correctly identified?

4. Confusion Matrix

Used to visualize model performance on test data.

🧾 Classification Report (Model 1 Example)
              precision    recall  f1-score   support

           0       0.88      0.81      0.85        27
           1       0.71      0.80      0.75        15

    accuracy                           0.81        42
   macro avg       0.79      0.81      0.80        42
weighted avg       0.82      0.81      0.81        42

💡 Key Learnings

Precision and Recall help understand misclassifications better than accuracy alone.

Dropout layers help reduce overfitting by randomly deactivating neurons during training.

The model can be further improved with:

Batch Normalization

Early Stopping

Learning Rate Scheduling

🧰 Tools & Environment

Language: Python 3.12

Libraries: TensorFlow, Keras, NumPy, Pandas, Scikit-learn

Hardware: Intel Iris Xe (CPU-based training)
⚙️ Note: Training on CPU took longer; using GPU would significantly reduce training time.

📈 Future Improvements

Implement cross-validation for more robust accuracy.

Use TensorBoard for performance visualization.

Experiment with different activation functions and optimizers.

Add batch normalization to improve training stability.

✨ Author

Namita Narang
📚 Exploring Deep Learning and Neural Networks
📊 Data Analysis | 🧠 Machine Learning | 🧩 AI Enthusiast
