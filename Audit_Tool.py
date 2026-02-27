import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from scipy.stats import zscore


# --- CONFIGURATION ---
GEN_API_KEY = "AIzaSyAapz_PBv4YaitmeipPOXzWSRviavEYmoY"
genai.configure(api_key=GEN_API_KEY)


#Function to get Gemini response
def get_gemini_response(input_prompt):
    model = genai.GenerativeModel('gemini-2.5-flash')
    content = [input_prompt]

# --- ALGORITHMS ---


def benfords_law_check(data):
    """Checks if the first digits of the 'Amount' column follow Benford's Law."""
    first_digits = data.astype(str).str[0].astype(int)
    # Filter out zeros
    first_digits = first_digits[first_digits > 0]
    observed = first_digits.value_counts(normalize=True).sort_index()
    
    # Benford's Expected Distribution
    expected = pd.Series({d: np.log10(1 + 1/d) for d in range(1, 10)})
    
    # Simple alert: if '1' appears < 25% or '9' appears > 10%
    is_suspicious = (observed.get(1, 0) < 0.25) or (observed.get(9, 0) > 0.08)
    return is_suspicious, observed


def find_zscore_anomalies(df):
    """Identifies transactions with an amount > 3 standard deviations from mean."""
    df['z_score'] = zscore(df['Amount'])
    anomalies = df[df['z_score'].abs() > 3]
    return anomalies


# --- INTERFACE ---


st.set_page_config(page_title="AuditGPT", page_icon="🕵️")
st.title("🛡️ AI Fraud Audit & Forensic Tool")


# Sidebar for Setup
with st.sidebar:
    st.header("1. Data Ingestion")
    uploaded_file = st.file_uploader("Upload Ledger (CSV/XLSX)", type=['csv', 'xlsx'])


# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome Auditor. Please upload your transaction file, then type 'Analyze' to begin fraud detection."}]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# Main Logic
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)


    if uploaded_file:
        # Load Data
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        
        if "analyze" in prompt.lower() or "audit" in prompt.lower():
            with st.spinner("Executing Forensic Algorithms..."):
                # 1. Run Stats
                is_benford_suspicious, dist = benfords_law_check(df['Amount'])
                anomalies = find_zscore_anomalies(df)
                
                # 2. Prepare Context for Gemini
                context = f"""
                AUDIT LOG SUMMARY:
                - Total Transactions: {len(df)}
                - Total Value: ${df['Amount'].sum():,.2f}
                - Benford's Law Deviation: {'High Risk' if is_benford_suspicious else 'Low Risk'}
                - Number of Outliers (Z-score > 3): {len(anomalies)}
                - Outlier Samples: {anomalies[['Amount', 'Date', 'Vendor']].head(5).to_dict()}
                """
                
                # 3. Gemini Report Generation
                ai_prompt = f"Act as a Forensic Accountant. Based on this data: {context}, write a professional Fraud Audit Report. Highlight specific risks and suggest investigation steps."
                response = get_gemini_response(prompt)
                
                # 4. Display Results
                st.chat_message("assistant").write(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
                # Show charts for visual proof
                st.bar_chart(dist)
                st.write("First Digit Distribution vs Benford's Law")
        else:
            # Handle general chat about the data
            ai_chat = model.generate_content(f"The user says: {prompt}. Data summary: {df.describe().to_string()}")
            st.chat_message("assistant").write(ai_chat.text)
            st.session_state.messages.append({"role": "assistant", "content": ai_chat.text})
    else:
        st.error("Please upload a file in the sidebar first!")