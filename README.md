**Titanic Survival Prediction Model**

This project focuses on building machine learning models to predict which passengers survived the Titanic disaster. The dataset used is from the popular Kaggle competition: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic).

---

### **Project Overview:**
The goal of this project is to apply different machine learning algorithms to predict passenger survival based on features such as age, gender, passenger class, fare, and more.

### **Models Implemented:**
- **Support Vector Machines (SVM)**
- **Logistic Regression**
- **Random Forest Classifier**
- **Gaussian Naive Bayes**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**
- **Stochastic Gradient Descent (SGD)**

### **Performance Metrics:**
| Model                        | Accuracy (%) | Cross-Validation Score (%) |
|-----------------------------|--------------|-----------------------------|
| Support Vector Machines      | 83.0         | 82.34                       |
| Logistic Regression          | 80.0         | 79.76                       |
| Random Forest                | 87.0         | 79.75                       |
| Gaussian Naive Bayes         | 80.0         | 79.64                       |
| K-Nearest Neighbors (KNN)    | 84.0         | 79.19                       |
| Decision Tree                | 84.0         | 78.85                       |
| Stochastic Gradient Descent  | 79.0         | 75.71                       |

---

### **Key Insights:**
- **Random Forest** achieved the highest accuracy but showed signs of overfitting.
- **Support Vector Machines** demonstrated the best balance between accuracy and cross-validation score, making it the most reliable model.
- Logistic Regression and Gaussian Naive Bayes provided consistent and stable results.

### **Project Structure:**
- `data/`: Contains training and test datasets.
- `notebooks/`: Jupyter notebooks with data exploration, preprocessing, and model training.
- `models/`: Saved trained models.
- `scripts/`: Python scripts for data processing and model evaluation.

---

### **Technologies Used:**
- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)

---

### **Next Steps:**
- Hyperparameter tuning to improve model performance.
- Implementing ensemble techniques.
- Adding data visualizations and more feature engineering.

---

**Feel free to explore the project and contribute!** 😊

