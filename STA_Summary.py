import streamlit as st
import pandas as pd
import openai
import time
from io import StringIO, BytesIO
from PyPDF2 import PdfReader
import docx
import os
from dotenv import load_dotenv
from openai.error import APIError, RateLimitError

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def read_uploaded_file(uploaded_file):
    """
    Read the uploaded file and return its content as text.
    Supports .txt, .pdf, .docx, and .xlsx formats.
    """
    content = ""
    if uploaded_file.name.endswith(".txt"):
        content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        content = " ".join([page.extract_text() for page in pdf_reader.pages])
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        content = " ".join([p.text for p in doc.paragraphs])
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        content = df.to_string(index=False)
    else:
        st.error("Unsupported file format!")
    return content

def chunk_text(text, chunk_size=5000):
    """
    Splits text into smaller chunks to stay within GPT-4 token limits.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def analyze_text_with_gpt4(text):
    """
    Uses GPT-4 to analyze text for errors while handling token limits.
    """
    chunks = chunk_text(text)
    analysis_reports = []

    for i, chunk in enumerate(chunks):
        prompt = f"""
        Analyze the following text for typographical errors, name inconsistencies, date inconsistencies, 
        and domain-specific mistakes. Provide a detailed, categorized report:
        {chunk}
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant helping with document analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,  # Reduced token limit
                temperature=0.7
            )

            analysis_reports.append(response.choices[0]['message']['content'].strip())
            time.sleep(2)  # Adding delay to prevent rate limits

        except openai.error.RateLimitError:
            st.error("Rate limit exceeded! Please wait or try again later.")
        except AttributeError:
            st.error("Unexpected response format from GPT-4!")

    return "\n\n".join(analysis_reports)



def main():
    st.title("Insurance Document Error Detector")
    st.write("Upload your insurance documents to detect errors like typographical issues, name inconsistencies, date errors, and more.")

    uploaded_files = st.file_uploader(
        "Upload files (txt, pdf, docx, xlsx)",
        type=["txt", "pdf", "docx", "xlsx"],
        accept_multiple_files=True
    )

    error_summary_placeholder = st.empty()

    if st.button("Detect Errors"):
        if not uploaded_files:
            st.error("Please upload at least one file!")
        else:
            all_errors = []
            for uploaded_file in uploaded_files:
                file_content = read_uploaded_file(uploaded_file)
                if not file_content:
                    continue

                st.write(f"Analyzing {uploaded_file.name}...")
                analysis_report = analyze_text_with_gpt4(file_content)

                error_summary_placeholder.write(f"### Errors in {uploaded_file.name}")
                error_summary_placeholder.write(analysis_report)

if __name__ == "__main__":
    main()
