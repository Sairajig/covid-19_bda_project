import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
from fpdf import FPDF
import re
import time
import plotly.express as px
import random

# Configure the API
genai.configure(api_key="AIzaSyAkQmZm6ayyy0AajZEh1FN7Ms5IrdbOVbQ")

def extract_patient_data(text):
    # Regular expressions to extract each field
    patient_id_pattern = r'"patient_id":\s*"([^"]*)"'
    age_pattern = r'"age":\s*(\d+)'
    gender_pattern = r'"gender":\s*"([^"]*)"'
    symptoms_pattern = r'"symptoms":\s*\[(.*?)\]'
    test_result_pattern = r'"test_result":\s*"([^"]*)"'
    hospitalized_pattern = r'"hospitalized":\s*(true|false)'

    # Extract data
    patient_ids = re.findall(patient_id_pattern, text)
    ages = re.findall(age_pattern, text)
    genders = re.findall(gender_pattern, text)
    symptoms = re.findall(symptoms_pattern, text)
    test_results = re.findall(test_result_pattern, text)
    hospitalized = re.findall(hospitalized_pattern, text)

    # Process symptoms
    symptoms = [s.replace('"', '').split(',') for s in symptoms]

    # Combine data
    data = []
    for i in range(min(len(patient_ids), len(ages), len(genders), len(symptoms), len(test_results), len(hospitalized))):
        data.append({
            "patient_id": patient_ids[i],
            "age": int(ages[i]),
            "gender": genders[i],
            "symptoms": symptoms[i],
            "test_result": test_results[i],
            "hospitalized": hospitalized[i] == "true"
        })

    return data

def get_patient_data(query, num_batches=10):
    model = genai.GenerativeModel('gemini-pro')
    all_data = []

    for _ in range(num_batches):
        prompt = f"""Generate a JSON array of patient data for the following query: {query}
        Each object should have the structure:
        {{
            "patient_id": "unique identifier",
            "age": number,
            "gender": "male or female",
            "symptoms": ["symptom1", "symptom2", ...],
            "test_result": "positive or negative",
            "hospitalized": true or false
        }}
        Provide data for 50 unique patients. Ensure the output is valid JSON."""

        try:
            response = model.generate_content(prompt)
            generated_text = response.text

            try:
                data = json.loads(generated_text)
                all_data.extend(data)
            except json.JSONDecodeError:
                json_like_structures = re.findall(r'\{[^{}]*\}', generated_text)
                for structure in json_like_structures:
                    try:
                        all_data.append(json.loads(structure))
                    except json.JSONDecodeError:
                        continue

            time.sleep(1)  # To avoid hitting rate limits
        except Exception as err:
            st.error(f"Error occurred: {err}")

    return all_data

def create_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=8)  # Smaller font size to fit more data
    
    # Calculate column widths based on content
    col_widths = [pdf.get_string_width(col) + 6 for col in df.columns]
    for i, col in enumerate(df.columns):
        col_widths[i] = max(col_widths[i], max(pdf.get_string_width(str(val)) + 6 for val in df[col]))
    
    # Add headers
    for i, col in enumerate(df.columns):
        pdf.cell(col_widths[i], 10, str(col), 1)
    pdf.ln()
    
    # Add data
    for _, row in df.iterrows():
        for i, col in enumerate(df.columns):
            value = str(row[col])
            if col == 'symptoms':
                value = ', '.join(row[col]) if isinstance(row[col], list) else value
            pdf.cell(col_widths[i], 10, value[:20] + '...' if len(value) > 20 else value, 1)
        pdf.ln()
    
    return pdf.output(dest='S').encode('latin-1')

def create_power_bi_visual(df):
    # Create a bar chart of test results
    fig_test_results = px.bar(df['test_result'].value_counts(), title="Test Results Distribution")
    st.plotly_chart(fig_test_results)

    # Create a pie chart of gender distribution
    fig_gender = px.pie(df, names='gender', title="Gender Distribution")
    st.plotly_chart(fig_gender)

    # Create a histogram of age distribution
    fig_age = px.histogram(df, x='age', title="Age Distribution")
    st.plotly_chart(fig_age)

    # Create a bar chart of top symptoms
    symptoms = [symptom for symptoms_list in df['symptoms'] for symptom in symptoms_list]
    symptom_counts = pd.Series(symptoms).value_counts().head(10)
    fig_symptoms = px.bar(symptom_counts, title="Top 10 Symptoms")
    st.plotly_chart(fig_symptoms)

# Streamlit UI
st.title("COVID-19 Patient Data Viewer")
query = st.text_input("Enter a query (e.g., COVID-19 data for pregnant women):")
num_batches = st.slider("Number of data batches to generate (each batch ~50 patients)", 1, 20, 10)

# Button to retrieve data
if st.button("Retrieve Data"):
    data = get_patient_data(query, num_batches)
    
    if data:
        df = pd.json_normalize(data)
        
        if 'symptoms' in df.columns:
            df['symptoms'] = df['symptoms'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        
        st.dataframe(df)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name="patient_data.csv", mime="text/csv")
        
        pdf = create_pdf(df)
        st.download_button(label="Download PDF", data=pdf, file_name="patient_data.pdf", mime="application/pdf")

        st.subheader("Power BI-like Visualizations")
        create_power_bi_visual(df)
    else:
        st.warning("No data generated. Please try again or refine your query.")