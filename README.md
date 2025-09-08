# 🧠 Stroke Prediction Model in R

This project builds and deploys a **stroke prediction model** using R.  
The workflow includes data preprocessing, exploratory data analysis (EDA), model training, model evaluation, and deployment.  
The final trained model is saved and can be reused for making predictions.

---

## 📊 Dataset
The dataset comes from [Kaggle – Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).  

It includes patient information such as:
- Gender, age, marital status, work type, residence type  
- Health factors (hypertension, heart disease, average glucose level, BMI)  
- Lifestyle factors (smoking status)  
- Stroke outcome (target variable: 1 = stroke, 0 = no stroke)

---

## ⚙️ Features
- Data cleaning & preprocessing (handling missing values, factors, scaling)  
- Exploratory Data Analysis (EDA) with visualizations  
- Machine learning models:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
- Model evaluation using:
  - Accuracy  
  - ROC-AUC  
  - Sensitivity & Specificity  
  - F1 Score  
- Deployment-ready model saved as `.rds`  

---

## 📂 Repository Structure


├── stroke-report.Rmd # Main R code/report
├── healthcare-dataset-stroke-data.csv # Dataset (if included)
├── best_stroke_model.rds # Saved best model
├── training_levels.rds # Factor levels for categorical variables
├── docs/
│ ├── index.html # Rendered HTML report (for GitHub Pages)
│ └── stroke-report.pdf # PDF report (optional)
└── README.md # Project documentation
