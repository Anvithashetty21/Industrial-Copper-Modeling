
# 🧠 Industrial Copper Modeling – Regression & Classification ML App

A complete end-to-end **Machine Learning solution** for the **copper manufacturing industry**, built to **predict selling prices** and **lead conversion outcomes** using real-world business data.

🔗 **Live Demo (Streamlit Cloud)**: [*Add your deployment link here*]  
📽️ **Demo Video**: [*Add LinkedIn or YouTube video link here*]

---

## 📌 Problem Statement

In the copper industry, pricing decisions and lead conversions are often hampered by:
- Skewed and noisy data
- Inconsistent feature patterns
- Manual prediction efforts

This project solves these challenges by:
- Building a **regression model** to predict `Selling_Price`
- Building a **classification model** to predict `Status` (WON / LOST)
- Delivering predictions via an **interactive Streamlit web app`

---

## 🏭 Domain
**Manufacturing / Industrial Sales Analytics**

---

## 🛠️ Tools & Technologies Used

| Tool         | Purpose                            |
|--------------|------------------------------------|
| Python       | Core programming                   |
| Pandas, NumPy| Data preprocessing, manipulation   |
| Seaborn, Matplotlib | EDA & visualization        |
| Sklearn      | Preprocessing, modeling utils      |
| XGBoost      | Regression & Classification models |
| Streamlit    | Web UI for predictions             |
| Joblib       | Model serialization (Pickle)       |

---

## 📂 Project Structure

```
├── Copper_Set.xlsx              # Raw Dataset
├── industrial_copper_modeling_final.py  # Training & model building script
├── app.py                       # Streamlit app for prediction
├── reg_model.pkl                # Trained XGBoost Regressor
├── clf_model.pkl                # Trained XGBoost Classifier
├── scaler.pkl                   # StandardScaler object
├── expected_columns.txt         # Input column order for models
├── cleaned_copper_data.csv      # Preprocessed dataset
└── README.md                    # Project documentation
```

---

## 🔍 Feature Overview

### 1. **Regression Task**
Predict the continuous value of `Selling_Price` using multiple business variables like:
- Quantity
- Product Reference
- Country
- Thickness, Width
- Lead Time, Application, etc.

### 2. **Classification Task**
Predict whether a quote will result in a **WON** or **LOST** status using the same features (excluding price).

---

## 🧪 ML Pipeline Overview

1. **Data Cleaning & Parsing**
2. **Outlier Removal** (IQR)
3. **Feature Engineering**
4. **Skewness Handling**
5. **Modeling** (XGBoost)
6. **Scaling** (StandardScaler)

---

## 🎯 Prediction App (Streamlit UI)

The user-friendly Streamlit app allows users to:
- Select regression or classification task
- Input relevant order details via dropdowns and sliders
- Get real-time predictions of:
  - ✅ **Selling Price**
  - 🏆 **Win/Loss Status**

---

## 🧾 Sample Inputs

### 🔹 Quote Status Labels

| Label              | Description                                        |
|--------------------|----------------------------------------------------|
| Not lost for AM    | Quote still active, not lost by Account Manager    |
| Offerable          | Ready to be sent to customer                       |
| Offered            | Officially shared with customer                    |
| Revised            | Modified as per feedback                           |
| To be approved     | Awaiting internal approval                         |
| Wonderful          | Indicates an exceptional or highly favorable quote |

### 🔹 Item Types

| Code   | Description            |
|--------|------------------------|
| PL     | Plates                 |
| S      | Strips                 |
| W      | Wires                  |
| WI     | Wire Insulated         |
| Others | Uncategorized items    |

### 🌍 Country Mapping

| Country       | Code |
|---------------|------|
| India         | 25   |
| Germany       | 28   |
| Italy         | 30   |
| France        | 32   |
| Spain         | 38   |
| Poland        | 26   |
| USA           | 27   |
| China         | 39   |
| UK            | 40   |
| Netherlands   | 77   |
| Belgium       | 78   |
| Sweden        | 79   |
| Switzerland   | 80   |
| Turkey        | 84   |
| Austria       | 89   |
| Norway        | 107  |
| Finland       | 113  |

---

## ✅ Project Highlights

- ✔️ Built with modular code and reusable components
- ✔️ Handles skewed, noisy, and real-world business data
- ✔️ Regression + Classification integrated in one UI
- ✔️ Fully deployable on **Streamlit Cloud**
- ✔️ GitHub-ready: clean, documented, and functional

---


## 📌 Learning Outcomes

- Real-time deployment of ML models
- Skewness handling, outlier removal, date feature engineering
- Regression + Classification use cases in manufacturing
- Streamlit app building with live prediction interface
- GitHub portfolio readiness for data roles

---

## 🙋‍♀️ Developed By
**Anvitha Shetty**  
Associate Consultant | Data Science Enthusiast  
[LinkedIn Profile](https://www.linkedin.com/in/shettyanvitha/)  
[GitHub](https://github.com/Anvithashetty21)
