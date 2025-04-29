import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
from io import StringIO, BytesIO
from PyPDF2 import PdfReader
import docx
import json
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def read_uploaded_file(uploaded_file):
    """Reads the uploaded file and extracts its content with line references."""
    content = ""
    if uploaded_file.name.endswith(".txt"):
        content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        content = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        content = " ".join([p.text for p in doc.paragraphs])
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        content_lines = []
        for row_idx, row in df.iterrows():
            for col_idx, col_name in enumerate(df.columns):
                cell_value = row[col_name]
                col_letter = chr(65 + col_idx)  # A, B, C, etc.
                cell_ref = f"{col_letter}{row_idx + 2}"
                content_lines.append(f"[{cell_ref}] {cell_value}")
        content = "\n".join(content_lines)
    else:
        st.error("Unsupported file format!")
    return content

def chunk_text(text, chunk_size=8000):
    """Splits text into smaller chunks to stay within token limits."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def analyze_text_with_gemini(text, retries=3, delay=5):
    """Uses Google Gemini 1.5 Pro to extract errors from document content with retry handling."""
    chunks = chunk_text(text)
    analysis_reports = []

    for i, chunk in enumerate(chunks):
        prompt = f"""
**You are an expert insurance document reviewer powered by advanced AI capabilities. Your task is to carefully analyze insurance-related documents and detect a wide range of possible errors, including typographical mistakes, inconsistencies, and domain-specific issues.**

Please perform the following checks on the document:

1. **Typographical Errors**
2. **Name Inconsistencies**
3. **Date Inconsistencies**
4. **Domain-Specific Mistakes**

**Output Requirements:**
- For each detected error, produce:
  - `Line_Number`
  - `Page_Number`
  - `Error_Type`
  - `Error_description`
  - `Suggestions`

- Always output a **single JSON object** with one key `"errors"` containing a **list of error objects**.
- No extra explanation or commentary — only valid JSON.
- No parsing errors in the JSON.

    - **Important Constraints:**  
    - Always output a **single JSON object** with one key `"errors"` containing a **list of error objects** — even if there is only one error.  
    - Do not add any explanation, text, or commentary — **only output valid JSON**.  
    - Ensure there are no parsing errors in the JSON structure.  
    
    **Example format (even for a single error):**
    ```json
    {{
    "errors": [
        {{
            "Page_Number": 1,
            "Line_Number": 25,
            "Error_Type": "Typographical Error",
            "Error_Description": "Misspelled word: 'insurence' instead of 'insurance'.",
            "Suggestions": "Correct 'insurence' to 'insurance'."
        }},
        {{
            "Page_Number": 1,
            "Line_Number": 12,
            "Error_Type": "Typographical Error",
            "Error_Description": "Misspelled word: 'premim' instead of 'premium'.",
            "Suggestions": "Correct 'premim' to 'premium'."
        }},
        {{
            "Page_Number": 1,
            "Line_Number": 30,
            "Error_Type": "Typographical Error",
            "Error_Description": "Misspelled word: 'polisy' instead of 'policy'.",
            "Suggestions": "Correct 'polisy' to 'policy'."
        }},
        {{
            "Page_Number": 1,
            "Line_Number": 45,
            "Error_Type": "Typographical Error",
            "Error_Description": "Misspelled word: 'benifit' instead of 'benefit'.",
            "Suggestions": "Correct 'benifit' to 'benefit'."
        }},
        {{
            "Page_Number": 2,
            "Line_Number": 15,
            "Error_Type": "Name Inconsistency",
            "Error_Description": "Variations of the name detected within the document: 'John Smith,' 'John A. Smith,' and 'J. Smith.'",
            "Suggestions": "Ensure consistent representation of the name in all relevant insurance documents."
        }},
        {{
            "Page_Number": 2,
            "Line_Number": 10,
            "Error_Type": "Name Inconsistency",
            "Error_Description": "Inconsistent representation: 'Jane Doe' in one section, 'J. Doe' in another.",
            "Suggestions": "Standardize the name across the document as 'Jane Doe' to maintain uniformity."
        }},
        {{
            "Page_Number": 3,
            "Line_Number": 20,
            "Error_Type": "Date Inconsistency",
            "Error_Description": "Start date (31-12-2025) occurs after the end date (01-01-2025).",
            "Suggestions": "Correct the dates to ensure chronological accuracy (e.g., adjust the start date to occur before the end date)."
        }},
        {{
            "Page_Number": 3,
            "Line_Number": 25,
            "Error_Type": "Date Inconsistency",
            "Error_Description": "Policy expiration date is mentioned as '01-01-2026', but a related claim references the expiration date as '31-12-2025'.",
            "Suggestions": "Verify the correct policy expiration date and update the documents accordingly."
        }},
        {{
            "Page_Number": 3,
            "Line_Number": 35,
            "Error_Type": "Date Inconsistency",
            "Error_Description": "Inconsistent date formats detected: 'MM/DD/YYYY' in one instance and 'YYYY-MM-DD' in another.",
            "Suggestions": "Standardize the date format across the insurance document (e.g., 'YYYY-MM-DD')."
        }},
        {{
            "Page_Number": 4,
            "Line_Number": 40,
            "Error_Type": "Policy Number Error",
            "Error_Description": "Invalid policy number format detected: 'AB12345' instead of 'POL-123456'.",
            "Suggestions": "Update the policy number to align with the specified format ('POL-123456')."
        }},
        {{
            "Page_Number": 4,
            "Line_Number": 50,
            "Error_Type": "Coverage Amount Error",
            "Error_Description": "Coverage amount ($10,000,000) is unrealistic for a basic auto insurance policy.",
            "Suggestions": "Verify the coverage amount and adjust it to match typical auto insurance policy standards."
        }},
        {{
            "Page_Number": 5,
            "Line_Number": 33,
            "Error_Type": "Coverage Detail Error",
            "Error_Description": "Coverage detail lists 'fire damage' in a life insurance policy, which is irrelevant.",
            "Suggestions": "Ensure coverage details align with the policy type; remove 'fire damage' from a life insurance policy."
        }},
        {{
            "Page_Number": 5,
            "Line_Number": 80,
            "Error_Type": "Missing Information",
            "Error_Description": "Policy document lacks a required start date.",
            "Suggestions": "Include the policy start date to ensure completeness of the document."
        }},
        {{
            "Page_Number": 6,
            "Line_Number": 65,
            "Error_Type": "Terminology Error",
            "Error_Description": "Incorrect terminology used: 'insured person' instead of 'policyholder'.",
            "Suggestions": "Replace 'insured person' with 'policyholder' to reflect precise insurance terminology."
        }}
    ]
}}
    ```

**Input document:**

{chunk}
        """

        for attempt in range(retries):
            try:
                model = genai.GenerativeModel('gemini-1.5-pro')
                response = model.generate_content(prompt)

                raw_content = response.text.strip()

                if raw_content.startswith("```json"):
                    raw_content = raw_content.replace("```json", "").replace("```", "").strip()

                parsed = json.loads(raw_content)
                errors = parsed.get("errors", [])
                analysis_reports.extend(errors)
                break  # Success

            except json.JSONDecodeError:
                st.error("❌ Failed to parse JSON from Gemini response.")
                st.code(raw_content)
                with open("gemini_raw_output_error.json", "w", encoding="utf-8") as f:
                    f.write(raw_content)
                break

            except Exception as e:
                st.warning(f"⚠️ Error on attempt {attempt + 1}: {e}")
                time.sleep(delay)
                continue

        time.sleep(1)  # Optional: small pause between chunks

    return analysis_reports

def export_errors_to_excel(errors, file_name="Analysis_Report.xlsx"):
    """Saves detected errors in an Excel file."""
    df = pd.DataFrame(errors)
    df_exploded = df.explode('Error Description').reset_index(drop=True)
    error_expanded = pd.json_normalize(df_exploded['Error Description'])
    df_final = pd.concat([df_exploded.drop(columns=['Error Description']), error_expanded], axis=1)
    output = BytesIO()
    df_final.to_excel(output, index=False, engine="openpyxl")
    return output.getvalue()

def main():
    st.title("🛡 Insurance Document Error Detector (Google Gemini)")
    st.write("Upload your insurance documents to detect errors like typographical issues, name inconsistencies, date errors, and more.")

    uploaded_files = st.file_uploader(
        "📎 Upload files (txt, pdf, docx, xlsx)",
        type=["txt", "pdf", "docx", "xlsx"],
        accept_multiple_files=True
    )

    error_summary_placeholder = st.empty()

    if st.button("🚀 Detect Errors"):
        if not uploaded_files:
            st.error("Please upload at least one file!")
        else:
            all_errors = []
            for uploaded_file in uploaded_files:
                file_content = read_uploaded_file(uploaded_file)
                if not file_content:
                    continue

                st.write(f"🔍 Analyzing **{uploaded_file.name}**...")
                analysis_report = analyze_text_with_gemini(file_content)

                error_summary_placeholder.write(f"### ❗ Errors in `{uploaded_file.name}`")
                error_summary_placeholder.write(analysis_report)

                all_errors.append({
                    "Document Name": uploaded_file.name,
                    "Error Description": analysis_report,
                })

            if all_errors:
                st.success("✅ Analysis completed!")
                excel_file = export_errors_to_excel(all_errors)
                st.download_button(
                    label="📥 Download Error Report",
                    data=excel_file,
                    file_name="Analysis_Report_Google.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
