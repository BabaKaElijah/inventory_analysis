import pandas as pd
import streamlit as st

from predict_pipeline import explain, predict


st.set_page_config(page_title="Inventory Demand Forecast", layout="centered")

st.title("Inventory Demand Forecast")
st.write("Enter inputs to predict daily units sold.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        store_id = st.text_input("Store ID", value="S001")
        product_id = st.text_input("Product ID", value="P001")
        category = st.selectbox(
            "Category",
            ["Electronics", "Clothing", "Home", "Groceries", "Toys"],
        )
        region = st.selectbox("Region", ["North", "South", "East", "West"])
        inventory_level = st.number_input("Inventory Level", min_value=0, value=120)
        units_ordered = st.number_input("Units Ordered", min_value=0, value=80)
        demand_forecast = st.number_input("Demand Forecast", min_value=0.0, value=140.5)
        price = st.number_input("Price", min_value=0.0, value=49.99)

    with col2:
        discount = st.number_input("Discount (%)", min_value=0, value=10)
        weather = st.selectbox(
            "Weather Condition",
            ["Sunny", "Rainy", "Cloudy", "Snowy"],
        )
        holiday = st.selectbox("Holiday/Promotion", [0, 1])
        competitor_pricing = st.number_input(
            "Competitor Pricing",
            min_value=0.0,
            value=52.0,
        )
        seasonality = st.selectbox(
            "Seasonality",
            ["Spring", "Summer", "Autumn", "Winter"],
        )
        day_of_week = st.number_input("Day of Week (0=Mon)", min_value=0, max_value=6, value=2)
        month = st.number_input("Month", min_value=1, max_value=12, value=7)
        day = st.number_input("Day", min_value=1, max_value=31, value=15)
        is_weekend = st.selectbox("Is Weekend", [0, 1])

    submitted = st.form_submit_button("Predict Units Sold")

if submitted:
    try:
        payload = {
            "Store ID": store_id,
            "Product ID": product_id,
            "Category": category,
            "Region": region,
            "Inventory Level": int(inventory_level),
            "Units Ordered": int(units_ordered),
            "Demand Forecast": float(demand_forecast),
            "Price": float(price),
            "Discount": int(discount),
            "Weather Condition": weather,
            "Holiday/Promotion": int(holiday),
            "Competitor Pricing": float(competitor_pricing),
            "Seasonality": seasonality,
            "day_of_week": int(day_of_week),
            "month": int(month),
            "day": int(day),
            "is_weekend": int(is_weekend),
        }

        df = pd.DataFrame([payload])
        prediction = float(predict(df)[0])
        st.success(f"Predicted Units Sold: {prediction:.2f}")

        st.subheader("Top Feature Contributions")
        contributions = explain(df, top_n=10)
        st.dataframe(contributions, use_container_width=True)
    except FileNotFoundError:
        st.error("Model not found. Train the model first.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
