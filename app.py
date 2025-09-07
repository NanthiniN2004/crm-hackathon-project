import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import google.generativeai as genai
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="AI CRM Insights Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- API Key Configuration (Secure Method) ---
# This is the secure way to handle API keys in Streamlit.
# It checks for the key in st.secrets, which is best for deployment.
# It then checks for an environment variable, good for local development.
GEMINI_API_KEY = None
try:
    # Ideal for deployed Streamlit apps
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    # Fallback for local development if secrets.toml is not used
    st.info("GEMINI_API_KEY not found in st.secrets. Trying environment variable...")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Final check to ensure the key is available
if not GEMINI_API_KEY:
    st.error("🚨 Gemini API Key is not configured!")
    st.markdown(
        "Please configure your API key either in a `.streamlit/secrets.toml` file "
        "or as an environment variable `GEMINI_API_KEY`."
    )
    st.stop() # Halts the app if no key is found

genai.configure(api_key=GEMINI_API_KEY)


# --- Caching and Model Loading ---
@st.cache_resource
def load_models_and_data():
    """Loads all models and data, caches for performance."""
    try:
        churn_model = joblib.load('churn_model.joblib')
        segmentation_model = joblib.load('segmentation_model.joblib')
        data = pd.read_csv('mock_crm_data.csv')
        return churn_model, segmentation_model, data
    except FileNotFoundError as e:
        st.error(f"❌ Error loading files: {e}. Please run `generate_data.py` and `train.py` first.")
        st.stop()

churn_model, segmentation_model, df = load_models_and_data()

# --- Predictions and Insights ---
# Segmentation
segmentation_features = ['TotalPurchases', 'EngagementScore']
df['Segment'] = segmentation_model.predict(df[segmentation_features])
segment_map = {0: 'High-Value', 1: 'Loyal', 2: 'At-Risk', 3: 'New'}
df['Segment'] = df['Segment'].map(segment_map)

# Churn Prediction
churn_features = ['TotalPurchases', 'LastInteractionDaysAgo', 'EngagementScore', 'Industry']
df['ChurnProbability'] = churn_model.predict_proba(df[churn_features])[:, 1]


# --- Main Application UI ---
st.title("🧠 AI-Powered CRM Insights Generator")
st.markdown("An interactive dashboard to segment customers, predict churn, and get actionable insights.")

# --- Key Metrics ---
st.header("Overall Business Health")
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", f"{len(df)}")
col2.metric("Predicted Churn Rate", f"{df['ChurnProbability'].mean():.2%}")
col3.metric("Average Engagement", f"{df['EngagementScore'].mean():.2f}")

# --- Visualization Tabs ---
tab1, tab2 = st.tabs(["📊 Customer Segmentation", "🔥 Churn Prediction"])

with tab1:
    st.header("Customer Segmentation")
    fig = px.scatter(
        df,
        x="EngagementScore",
        y="TotalPurchases",
        color="Segment",
        hover_name="CompanyName",
        hover_data=['Industry', 'ChurnProbability'],
        title="Customer Segments Based on Engagement and Purchases"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Top Customers at Risk of Churn")
    churn_risk_df = df.sort_values(by="ChurnProbability", ascending=False).head(10)
    st.dataframe(
        churn_risk_df[['CompanyName', 'Industry', 'ChurnProbability', 'Segment']],
        use_container_width=True,
        column_config={
            "ChurnProbability": st.column_config.ProgressColumn(
                "Churn Probability",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
        }
    )

# --- Chatbot Interface ---
st.header("🤖 Get Actionable Insights with AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about high-value customers, churn risks, or upsell opportunities..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare data context for the model
    data_context = df.head(10).to_string() # Provide a sample of the data

    # Construct the full prompt for Gemini
    full_prompt = f"""
    You are a helpful CRM analyst. Analyze the following CRM data and answer the user's question.
    Provide concise, actionable business insights.

    Here is a sample of the customer data:
    {data_context}

    User's Question:
    {prompt}
    """
    
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."): # --- UX IMPROVEMENT: ADDED SPINNER ---
            try:
                model = genai.GenerativeModel('gemini-1.5-flash-latest') # Using a newer, faster model
                response = model.generate_content(full_prompt)
                full_response = response.text
                st.markdown(full_response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                full_response = "Sorry, I couldn't process that request."
                st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
