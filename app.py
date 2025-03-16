import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
from fpdf import FPDF
import re
import time
import plotly.express as px

# Configure the API
genai.configure(api_key="AIzaSyAMg-g3H6F-0QSBvuxUAhyfa7uQojKAuT4")

# Function to generate a COVID-19 article
def generate_covid_article(tone, word_limit, specific_topic):
    if not specific_topic:
        st.warning("Please enter a specific topic.")
        return ""
    
    prompt = f"Write a {tone.lower()} article about {specific_topic} related to COVID-19. The article should be approximately {word_limit} words long."
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            max_output_tokens=word_limit * 4,  # Approximate tokens to words ratio
            temperature=0.7,
            top_p=0.8,
            top_k=40
        ))
        return response.text
    except Exception as e:
        st.error(f"Error occurred while generating the article: {e}")
        return ""

# Function to retrieve patient data (COVID-19)
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
        Provide data for 50 unique patients."""
        
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

            time.sleep(1)
        except Exception as err:
            st.error(f"Error occurred: {err}")

    return all_data

# Function to create PDF from patient data
def create_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=8)
    
    col_widths = [pdf.get_string_width(col) + 6 for col in df.columns]
    for i, col in enumerate(df.columns):
        col_widths[i] = max(col_widths[i], max(pdf.get_string_width(str(val)) + 6 for val in df[col]))
    
    for i, col in enumerate(df.columns):
        pdf.cell(col_widths[i], 10, str(col), 1)
    pdf.ln()
    
    for _, row in df.iterrows():
        for i, col in enumerate(df.columns):
            value = str(row[col])
            if col == 'symptoms':
                value = ', '.join(row[col]) if isinstance(row[col], list) else value
            pdf.cell(col_widths[i], 10, value[:20] + '...' if len(value) > 20 else value, 1)
        pdf.ln()
    
    return pdf.output(dest='S').encode('latin-1')

# Function to create visualizations (Power BI-like)
def create_power_bi_visual(df):
    fig_test_results = px.bar(df['test_result'].value_counts(), title="Test Results Distribution")
    st.plotly_chart(fig_test_results)

    fig_gender = px.pie(df, names='gender', title="Gender Distribution")
    st.plotly_chart(fig_gender)

    fig_age = px.histogram(df, x='age', title="Age Distribution")
    st.plotly_chart(fig_age)

    symptoms = [symptom for symptoms_list in df['symptoms'] for symptom in symptoms_list]
    symptom_counts = pd.Series(symptoms).value_counts().head(10)
    fig_symptoms = px.bar(symptom_counts, title="Top 10 Symptoms")
    st.plotly_chart(fig_symptoms)

# Streamlit UI
st.title("COVID-19 Data Viewer & Article Generator")

# Tab structure for the two functionalities
tab1, tab2 = st.tabs(["COVID-19 Visualization", "COVID-19 Article Generation"])

# COVID-19 Data Visualization
with tab1:
    st.header("COVID-19 Patient Data Viewer")
    
    query = st.text_input("Enter a query for COVID-19 data (e.g., COVID-19 data for pregnant women):")
    num_batches = st.slider("Number of data batches to generate (each batch ~50 patients)", 1, 20, 10)

    if st.button("Retrieve Patient Data"):
        data = get_patient_data(query, num_batches)
        
        if data:
            df = pd.json_normalize(data)
            
            if 'symptoms' in df.columns:
                df['symptoms'] = df['symptoms'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
            
            st.dataframe(df)
            
            # CSV Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download CSV", data=csv, file_name="patient_data.csv", mime="text/csv")
            
            # PDF Download
            pdf = create_pdf(df)
            st.download_button(label="Download PDF", data=pdf, file_name="patient_data.pdf", mime="application/pdf")
            
            st.subheader("Power BI-like Visualizations")
            create_power_bi_visual(df)
        else:
            st.warning("No data generated. Please try again or refine your query.")

# COVID-19 Article Generation
with tab2:
    st.header("COVID-19 Article Generator")

    # Article Tone Selection
    tones = ["Informative", "Creative", "Formal", "Conversational", "Persuasive"]
    selected_tone = st.selectbox("Choose a tone for the article:", tones)

    # Word Limit
    word_limits = [500, 800, 1000, 1500, 2000]
    selected_word_limit = st.selectbox("Select a word limit for the article:", word_limits)

    # Specific Topic Input
    specific_topic = st.text_input("Enter a specific COVID-19 topic (e.g., COVID-19 vaccination, pandemic impact):")

    # Generate Article Button
    if st.button("Generate COVID-19 Article"):
        with st.spinner("Generating article..."):
            article = generate_covid_article(selected_tone, selected_word_limit, specific_topic)
            if article:
                st.success("COVID-19 article generated successfully!")
                st.write(article)

    # Refresh Button for Article Inputs
    if st.button("Refresh Article Inputs"):
        st.experimental_rerun()
