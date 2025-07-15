import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Industrial Copper Modeling", layout="wide")

# Load models and scaler
reg_model = joblib.load("reg_model.pkl")
clf_model = joblib.load("clf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Country name to code mapping (Full List from Dataset)
country_map = {
    "India": 25, "Germany": 28, "Italy": 30, "France": 32, "Spain": 38,
    "Poland": 26, "USA": 27, "China": 39, "UK": 40, "Netherlands": 77,
    "Belgium": 78, "Sweden": 79, "Switzerland": 80, "Turkey": 84,
    "Austria": 89, "Norway": 107, "Finland": 113
}

status_options = [
    "Not lost for AM", "Offerable", "Offered", "Revised", "To be approved", "Wonderful"
]
item_type_options = ["Others", "PL", "S", "W", "WI"]

st.title("Industrial Copper Modeling App")
st.markdown("""
Predict Selling Price and Quote Outcome for Industrial Copper Orders using Machine Learning
""")

st.header("Enter Order Details")

month_names = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}

col1, col2, col3 = st.columns(3)
with col1:
    quantity_tons = st.number_input("Quantity (tons)", min_value=0.0, value=1.0)
    customer = st.selectbox("Customer ID (numeric)", list(range(10000, 10020)))
    country_name = st.selectbox("Country", list(country_map.keys()))
    country = country_map[country_name]
    application = st.number_input("Application Code", min_value=0, value=3)

with col2:
    thickness = st.number_input("Thickness (mm)", min_value=0.0, value=2.5)
    width = st.number_input("Width (mm)", min_value=0.0, value=1000.0)
    product_ref = st.selectbox("Product Reference", list(range(150000, 150020)))
    item_year = st.selectbox("Item Year", list(range(2020, 2026)))
    item_month = st.selectbox("Item Month", list(month_names.keys()))

with col3:
    item_dayofweek = st.selectbox("Item Day of Week (0=Mon, 6=Sun)", list(range(7)))
    delivery_year = st.selectbox("Delivery Year", list(range(2020, 2026)))
    delivery_month = st.selectbox("Delivery Month", list(month_names.keys()))
    delivery_dayofweek = st.selectbox("Delivery Day of Week (0=Mon, 6=Sun)", list(range(7)))
    delivery_lead_time = st.number_input("Delivery Lead Time (days)", min_value=0, value=10)

st.markdown("---")
st.subheader("Status")
status_selected = st.radio("Current Quote Status", status_options, horizontal=True)

st.subheader("Item Type")
item_type_selected = st.radio("Product Type", item_type_options, horizontal=True)

# Build input dictionary
input_dict = {
    'quantity tons': quantity_tons,
    'customer': customer,
    'country': country,
    'application': application,
    'thickness': thickness,
    'width': width,
    'product_ref': product_ref,
    'item_year': item_year,
    'item_month': month_names[item_month],
    'item_dayofweek': item_dayofweek,
    'delivery_year': delivery_year,
    'delivery_month': month_names[delivery_month],
    'delivery_dayofweek': delivery_dayofweek,
    'delivery_lead_time': delivery_lead_time
}

# One-hot encode status
for status in status_options:
    input_dict[f'status_{status}'] = 1 if status == status_selected else 0

# One-hot encode item type
for t in item_type_options:
    input_dict[f'item type_{t}'] = 1 if t == item_type_selected else 0

input_df = pd.DataFrame([input_dict])
scaled_input = scaler.transform(input_df)

# Make predictions
predicted_price = reg_model.predict(scaled_input)[0]
predicted_status = clf_model.predict(scaled_input)[0]

# Display Results
st.markdown("---")
st.header("Prediction Results")
st.success(f"Predicted Selling Price: ₹ {predicted_price:,.2f}")
st.info(f"Predicted Status (Win/Not-Win): {'Win' if predicted_status == 1 else 'Not Win'}")

# Expandable input guide
with st.expander("Input Feature Guide"):
    st.markdown("""
    ### Status Labels (Sales/Quote Stage)
    | Status Label | Meaning |
    |--------------|---------|
    | Not lost for AM | Quote still active, not lost by Account Manager |
    | Offerable | Quote ready to be sent |
    | Offered | Quote officially sent to customer |
    | Revised | Quote modified as per feedback |
    | To be approved | Awaiting internal approval |
    | Wonderful | Internally recognized as high-priority or strategic quote |

    ### Item Type Labels (Product Classification)
    | Item Type | Meaning |
    |-----------|---------|
    | Others | Uncategorized items |
    | PL | Plates |
    | S | Strips |
    | W | Wires |
    | WI | Wire Insulated or Intermediate |

    ### Country Codes
    | Country | Code |
    |---------|------|
    | India | 25 |
    | Germany | 28 |
    | Italy | 30 |
    | France | 32 |
    | Spain | 38 |
    | Poland | 26 |
    | USA | 27 |
    | China | 39 |
    | UK | 40 |
    | Netherlands | 77 |
    | Belgium | 78 |
    | Sweden | 79 |
    | Switzerland | 80 |
    | Turkey | 84 |
    | Austria | 89 |
    | Norway | 107 |
    | Finland | 113 |

    ### Notes:
    - **Item/Delivery Month** uses full month names.
    - **Day of Week**: 0 = Monday, ..., 6 = Sunday.
    """)
