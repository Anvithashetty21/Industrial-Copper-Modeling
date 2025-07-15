# industrial_copper_modeling_final.py

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import skew

# Step 1: Load and Clean Data
df = pd.read_excel('data/Copper_Set.xlsx', sheet_name='Result 1')
df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
df = df[df['id'].notna()]
df['customer'].fillna(df['customer'].mode()[0], inplace=True)
df['status'].fillna(df['status'].mode()[0], inplace=True)
df['item_date'] = pd.to_datetime(df['item_date'], errors='coerce', format='%Y%m%d')
df['delivery date'] = pd.to_datetime(df['delivery date'], errors='coerce', format='%Y%m%d')
df['selling_price'].fillna(df['selling_price'].median(), inplace=True)
df['thickness'].fillna(df['thickness'].median(), inplace=True)
df['application'].fillna(df['application'].mode()[0], inplace=True)
df['country'].fillna(df['country'].mode()[0], inplace=True)
df['quantity tons'].fillna(df['quantity tons'].median(), inplace=True)
df['item_date'].fillna(df['item_date'].mode()[0], inplace=True)
df['delivery date'].fillna(df['delivery date'].mode()[0], inplace=True)

# Step 2: Outlier Removal (IQR)
for col in ['selling_price', 'quantity tons', 'thickness']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# Step 3: Feature Engineering
df['item_year'] = df['item_date'].dt.year
df['item_month'] = df['item_date'].dt.month
df['item_dayofweek'] = df['item_date'].dt.dayofweek
df['delivery_year'] = df['delivery date'].dt.year
df['delivery_month'] = df['delivery date'].dt.month
df['delivery_dayofweek'] = df['delivery date'].dt.dayofweek
df['delivery_lead_time'] = (df['delivery date'] - df['item_date']).dt.days
df.drop(columns=['item_date', 'delivery date', 'id', 'material_ref'], inplace=True)
df = pd.get_dummies(df, columns=['status', 'item type'], drop_first=True)

# Step 4: Skewness Treatment
skewed_feats = df.select_dtypes(include=['float64', 'int64']).apply(lambda x: skew(x.dropna()))
high_skew = skewed_feats[skewed_feats > 1].index
for col in high_skew:
    if (df[col] <= 0).any():
        df[col] = np.log1p(df[col] - df[col].min() + 1)
    else:
        df[col] = np.log1p(df[col])

# Step 5: Prepare Regression Data
X_reg = df.drop(columns=['selling_price', 'status_Won'])
y_reg = df['selling_price']

# Step 6: Prepare Classification Data (Only WON/LOST)
clf_df = df[df['status_Won'].isin([0, 1])].copy()
X_clf = clf_df.drop(columns=['selling_price', 'status_Won'])
y_clf = clf_df['status_Won']

# Step 7: Scaling
scaler = StandardScaler()
scaler.fit(X_reg)  # Fit on regression data
X_reg_scaled = pd.DataFrame(scaler.transform(X_reg), columns=X_reg.columns)
X_clf_scaled = pd.DataFrame(scaler.transform(X_clf), columns=X_clf.columns)

# Save expected column order
with open("expected_columns.txt", "w") as f:
    for col in X_reg.columns:
        f.write(col + "\n")

# Step 8: Model Training and Saving
reg_model = XGBRegressor(random_state=42)
reg_model.fit(X_reg_scaled, y_reg)
joblib.dump(reg_model, 'reg_model.pkl')

clf_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
clf_model.fit(X_clf_scaled, y_clf)
joblib.dump(clf_model, 'clf_model.pkl')

joblib.dump(scaler, 'scaler.pkl')
df.to_csv("cleaned_copper_data.csv", index=False)

print("✅ Models, scaler, and cleaned data saved successfully.")
