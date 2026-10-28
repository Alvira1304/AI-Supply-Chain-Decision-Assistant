
import os
import math
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Load the API key
PROJECT_FOLDER = Path(__file__).resolve().parent
load_dotenv(PROJECT_FOLDER / ".env")

st.set_page_config(
    page_title="AI Supply Chain Decision Assistant",
    page_icon="📦",
    layout="wide"
)

st.title("📦 AI Supply Chain Decision Assistant")
st.write("Monitor inventory, identify risks, and get AI recommendations.")

# Load inventory data
try:
    df = pd.read_csv(PROJECT_FOLDER / "inventory.csv")
except FileNotFoundError:
    st.error("inventory.csv not found in the project folder.")
    st.stop()

# Check required columns
required = [
    "product_id", "product_name", "stock_quantity",
    "reorder_level", "unit_price", "monthly_sales",
    "supplier", "delivery_days"
]

if any(column not in df.columns for column in required):
    st.error("Your CSV is missing one or more required columns.")
    st.stop()

# Convert numeric columns and validate data
numeric_columns = [
    "stock_quantity", "reorder_level", "unit_price",
    "monthly_sales", "delivery_days"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

if df[numeric_columns].isna().any().any():
    st.error("Please check the numeric values in inventory.csv.")
    st.stop()

if (
    (df[["stock_quantity", "reorder_level", "monthly_sales",
         "delivery_days"]] < 0).any().any()
):
    st.error("Stock, sales, reorder levels, and delivery days cannot be negative.")
    st.stop()

# 1. Calculate average daily sales
df["daily_sales"] = df["monthly_sales"] / 30

# 2. Estimate how many days current stock will last
df["days_of_stock"] = df.apply(
    lambda row: (
        row["stock_quantity"] / row["daily_sales"]
        if row["daily_sales"] > 0
        else float("inf")
    ),
    axis=1
)

# 3. Estimate stockout risk
def identify_risk(row):
    if row["daily_sales"] == 0:
        if row["stock_quantity"] <= row["reorder_level"]:
            return "Review Stock"
        return "No Recent Demand"

    if row["days_of_stock"] <= row["delivery_days"]:
        return "Critical"

    if row["stock_quantity"] <= row["reorder_level"]:
        return "Reorder Now"

    if row["days_of_stock"] <= row["delivery_days"] + 7:
        return "Watch"

    return "Healthy"


df["risk_status"] = df.apply(identify_risk, axis=1)

# 4. Estimate a reorder quantity
# Target stock covers supplier lead time plus 15 safety-stock days.
def calculate_order_quantity(row):
    if row["daily_sales"] == 0:
        return 0

    target_stock = row["daily_sales"] * (
        row["delivery_days"] + 15
    )

    return max(
        0,
        math.ceil(target_stock - row["stock_quantity"])
    )


df["suggested_order_quantity"] = df.apply(
    calculate_order_quantity, axis=1
)

# Sort products by urgency
risk_order = {
    "Critical": 0,
    "Reorder Now": 1,
    "Review Stock": 2,
    "Watch": 3,
    "Healthy": 4,
    "No Recent Demand": 5
}

df["risk_rank"] = df["risk_status"].map(risk_order)
df = df.sort_values(
    ["risk_rank", "days_of_stock"],
    ascending=[True, True]
)

# Dashboard summary
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Products", len(df))
col2.metric(
    "Critical Products",
    int((df["risk_status"] == "Critical").sum())
)
col3.metric(
    "Products to Reorder",
    int(df["suggested_order_quantity"].gt(0).sum())
)
col4.metric(
    "Total Inventory Units",
    int(df["stock_quantity"].sum())
)

# Display inventory analysis
st.subheader("📋 Inventory Risk Analysis")

display_columns = [
    "product_name",
    "stock_quantity",
    "monthly_sales",
    "delivery_days",
    "days_of_stock",
    "risk_status",
    "suggested_order_quantity"
]

display_df = df[display_columns].copy()
display_df["days_of_stock"] = display_df["days_of_stock"].replace(
    float("inf"), "No recent demand"
)

st.dataframe(display_df, use_container_width=True)

st.caption(
    "Estimates assume 30-day months and stable sales. "
    "Suggested orders cover supplier lead time plus 15 safety-stock days."
)

# Show recommended actions
st.subheader("🚨 Highest-Priority Products")

urgent = df[df["risk_status"].isin(["Critical", "Reorder Now"])]

if urgent.empty:
    st.success("No urgent reorders identified.")
else:
    for _, row in urgent.iterrows():
        st.write(
            f"**{row['product_name']}** — {row['risk_status']}. "
            f"Current stock: {row['stock_quantity']:.0f} units. "
            f"Estimated stock coverage: {row['days_of_stock']:.1f} days. "
            f"Suggested order: {row['suggested_order_quantity']} units."
        )

st.subheader("📊 Monthly Sales")
st.bar_chart(df.set_index("product_name")["monthly_sales"])

# Groq-powered AI assistant
st.divider()
st.subheader("🤖 Ask the AI Supply Chain Assistant")

st.write(
    "Ask questions about stock, delivery risks, and suggested orders."
)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("Groq API key not found. Check your .env file.")
else:
    question = st.text_input(
        "Your question",
        placeholder="Which product should we reorder first?"
    )

    if st.button("Get AI Recommendation"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing supply-chain data..."):
                try:
                    client = Groq(api_key=api_key)

                    # Send calculated results, not just raw inventory
                    ai_data = df[display_columns].copy()
                    ai_data["days_of_stock"] = ai_data[
                        "days_of_stock"
                    ].replace(float("inf"), None)

                    response = client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a supply-chain decision assistant. "
                                    "Use the supplied inventory data and calculated "
                                    "risk statuses to answer the user's question. "
                                    "Prioritize Critical products, explain why, "
                                    "and use simple English. Do not invent facts. "
                                    "Suggested order quantities are estimates, not "
                                    "guaranteed optimal orders. Mention missing "
                                    "information when relevant."
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Inventory analysis:\n"
                                    f"{ai_data.to_csv(index=False)}\n\n"
                                    f"Question: {question}"
                                )
                            }
                        ],
                        temperature=0.2
                    )

                    st.markdown("### 💡 AI Recommendation")
                    st.write(response.choices[0].message.content)

                except Exception as error:
                    st.error(f"AI request failed: {error}")

st.caption(
    "Prototype using fictional data. Verify recommendations before ordering."
)
