
from pathlib import Path
import os

import chromadb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "inventory.csv"
DB_DIR = BASE_DIR / "chroma_db"

load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="AI Supply Chain Decision Assistant",
    page_icon="📦",
    layout="wide"
)

# --------------------------------------------------
# 2. LOAD GROQ CLIENT
# --------------------------------------------------

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error(
        "GROQ_API_KEY was not found. "
        "Check that your .env file contains your API key."
    )
    st.stop()

client = Groq(api_key=api_key)
MODEL_NAME = "openai/gpt-oss-20b"

# --------------------------------------------------
# 3. LOAD RAG KNOWLEDGE BASE
# --------------------------------------------------

@st.cache_resource
def load_rag_components():
    if not DB_DIR.exists():
        raise FileNotFoundError(
            "The ChromaDB database was not found. "
            "Run: python build_knowledge_base.py"
        )

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    chroma_client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    collection = chroma_client.get_collection(
        name="supply_chain_policy"
    )

    return embedding_model, collection


try:
    embedding_model, collection = load_rag_components()
    rag_ready = True
except Exception as exc:
    embedding_model = None
    collection = None
    rag_ready = False
    rag_error = str(exc)

# --------------------------------------------------
# 4. LOAD AND ANALYZE INVENTORY
# --------------------------------------------------

@st.cache_data
def load_inventory():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Inventory file not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "product_id",
        "product_name",
        "stock_quantity",
        "reorder_level",
        "unit_price",
        "monthly_sales",
        "supplier",
        "delivery_days"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing inventory columns: {', '.join(missing)}"
        )

    numeric_columns = [
        "stock_quantity",
        "reorder_level",
        "unit_price",
        "monthly_sales",
        "delivery_days"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_columns] = df[numeric_columns].fillna(0)

    df["daily_sales"] = df["monthly_sales"] / 30

    df["days_of_stock"] = (
        df["stock_quantity"] /
        df["daily_sales"].replace(0, float("nan"))
    )

    def determine_risk(row):
        if row["monthly_sales"] <= 0:
            if row["stock_quantity"] <= row["reorder_level"]:
                return "Review Stock"
            return "No Recent Demand"

        if row["days_of_stock"] <= row["delivery_days"]:
            return "Critical"

        if row["stock_quantity"] <= row["reorder_level"]:
            return "Reorder Now"

        if row["days_of_stock"] <= row["delivery_days"] + 7:
            return "Review Stock"

        if row["stock_quantity"] <= row["reorder_level"] * 1.25:
            return "Watch"

        return "Healthy"

    df["risk_status"] = df.apply(determine_risk, axis=1)

    # Sample target: cover supplier lead time plus 15 safety-stock days.
    target_days = df["delivery_days"] + 15

    df["suggested_order_quantity"] = (
        (df["daily_sales"] * target_days)
        - df["stock_quantity"]
    ).clip(lower=0).round().astype(int)

    return df


try:
    inventory = load_inventory()
except Exception as exc:
    st.error(f"Could not load inventory: {exc}")
    st.stop()

# --------------------------------------------------
# 5. RETRIEVE RELEVANT POLICY (RAG)
# --------------------------------------------------

def retrieve_policy(question, number_of_results=3):
    if not rag_ready:
        return (
            "Policy retrieval is currently unavailable. "
            "Do not invent policy details."
        )

    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(number_of_results, collection.count())
    )

    documents = results.get("documents", [[]])[0]

    if not documents:
        return "No relevant policy information was found."

    return "\n\n".join(
        f"Policy passage {i + 1}: {doc}"
        for i, doc in enumerate(documents)
    )

# --------------------------------------------------
# 6. ASK GROQ USING RETRIEVED CONTEXT
# --------------------------------------------------

def ask_ai(question, inventory_context):
    policy_context = retrieve_policy(question)

    system_prompt = """
You are an AI Supply Chain Decision Assistant.

Answer in simple, clear English.

Use the supplied inventory data for inventory-related questions.
Use the retrieved policy passages for company-policy questions.

Rules:
1. Do not invent stock figures, suppliers, policies, or facts.
2. If the provided information does not answer the question,
   clearly say what information is missing.
3. Explain critical stock using estimated stock days and supplier
   delivery time when those figures are available.
4. Treat order quantities as suggestions, not automatic orders.
5. All AI-generated purchase suggestions require approval
   from an authorized employee.
6. Clearly distinguish supplied facts from recommendations.
"""

    user_prompt = f"""
USER QUESTION:
{question}

RETRIEVED PURCHASING POLICY:
{policy_context}

CURRENT INVENTORY DATA:
{inventory_context}

Answer the user's question using the relevant information above.
If the question is unrelated to the policy, use the inventory data
when appropriate. Do not make up missing information.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=1000
    )

    return response.choices[0].message.content

# --------------------------------------------------
# 7. APP HEADER
# --------------------------------------------------

st.title("📦 AI Supply Chain Decision Assistant")

st.write(
    "Analyze inventory risks, identify products that may need "
    "reordering, and ask questions using your purchasing policy."
)

if rag_ready:
    st.success("RAG knowledge base connected.")
else:
    st.warning(
        f"Policy retrieval is unavailable: {rag_error}. "
        "The inventory dashboard can still be used."
    )

st.divider()

# --------------------------------------------------
# 8. INVENTORY DASHBOARD
# --------------------------------------------------

st.header("📊 Inventory Dashboard")

critical_count = int(
    (inventory["risk_status"] == "Critical").sum()
)

reorder_count = int(
    (inventory["risk_status"] == "Reorder Now").sum()
)

review_count = int(
    (inventory["risk_status"] == "Review Stock").sum()
)

total_products = len(inventory)

total_inventory_value = (
    inventory["stock_quantity"] * inventory["unit_price"]
).sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Products", total_products)
col2.metric("Critical Products", critical_count)
col3.metric("Reorder Now", reorder_count)
col4.metric("Inventory Value", f"₹{total_inventory_value:,.0f}")

st.subheader("Inventory Risk Overview")

status_filter = st.multiselect(
    "Filter by risk status",
    options=sorted(inventory["risk_status"].unique()),
    default=sorted(inventory["risk_status"].unique())
)

filtered_inventory = inventory[
    inventory["risk_status"].isin(status_filter)
].copy()

display_columns = [
    "product_id",
    "product_name",
    "stock_quantity",
    "reorder_level",
    "monthly_sales",
    "delivery_days",
    "days_of_stock",
    "risk_status",
    "suggested_order_quantity",
    "supplier"
]

display_inventory = filtered_inventory[display_columns].copy()

display_inventory["days_of_stock"] = (
    display_inventory["days_of_stock"].round(1)
)

display_inventory["days_of_stock"] = (
    display_inventory["days_of_stock"].fillna("Unknown")
)

st.dataframe(
    display_inventory,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Suggested order quantities use average monthly sales and a "
    "sample target of supplier lead time plus 15 safety-stock days. "
    "Review suggestions before placing orders."
)

st.download_button(
    label="Download Inventory Analysis CSV",
    data=filtered_inventory.to_csv(index=False).encode("utf-8"),
    file_name="inventory_analysis.csv",
    mime="text/csv"
)

st.divider()

# --------------------------------------------------
# 9. AI SUPPLY CHAIN ASSISTANT
# --------------------------------------------------

st.header("🤖 Ask the AI Assistant")

st.write(
    "Ask about stock risks, reorder suggestions, supplier lead times, "
    "or the purchasing policy."
)

if not rag_ready:
    st.info(
        "You can still ask inventory questions, but policy-based "
        "answers may be limited until the knowledge base is available."
    )

example_questions = [
    "Which products are critical?",
    "What should we do when stock is critical?",
    "Which products may need reordering?",
    "Explain the purchasing policy for safety stock.",
    "Why should AI-generated order suggestions be reviewed?"
]

selected_question = st.selectbox(
    "Choose an example question (optional)",
    ["Write my own question"] + example_questions
)

default_question = (
    "" if selected_question == "Write my own question"
    else selected_question
)

with st.form("ai_question_form"):
    question = st.text_area(
        "Your question",
        value=default_question,
        placeholder="Example: Which products should we reorder first?",
        height=100
    )

    submitted = st.form_submit_button("Ask AI", type="primary")

if submitted:
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        inventory_context = inventory.to_csv(index=False)

        with st.spinner(
            "Searching the policy and generating an answer..."
        ):
            try:
                answer = ask_ai(
                    question.strip(),
                    inventory_context
                )

                st.subheader("AI Answer")
                st.markdown(answer)

                with st.expander("View retrieved policy context"):
                    st.write(retrieve_policy(question.strip()))

            except Exception as exc:
                st.error(
                    "The AI request failed. Check your internet "
                    "connection, Groq API configuration, and model "
                    "availability."
                )
                st.caption(f"Technical details: {exc}")

st.divider()

st.caption(
    "Prototype for decision support only. Verify inventory data and "
    "all recommendations before making purchasing decisions."
)
