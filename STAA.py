import streamlit as st
import pandas as pd
import openai
import time
from io import StringIO, BytesIO
from PyPDF2 import PdfReader
import docx
import json
import os
from dotenv import load_dotenv
from openai.error import APIError, RateLimitError

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
        content = df.to_string(index=False)
    else:
        st.error("Unsupported file format!")
    return content

def chunk_text(text, chunk_size=8000):
    """Splits text into smaller chunks to stay within token limits."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def analyze_text_with_gpt(text, retries=3, delay=5):
    """Uses GPT-4o Mini to extract errors from document content with retry handling."""
    chunks = chunk_text(text)
    analysis_reports = []

    for i, chunk in enumerate(chunks):
        prompt = f"""
**You are an expert insurance document reviewer powered by advanced AI capabilities. Your task is to carefully analyze insurance-related documents and detect a wide range of possible errors, including typographical mistakes, inconsistencies, and domain-specific issues.**

Please perform the following checks on the document:

1. **Typographical Errors:**
   - Detect spelling mistakes.
   - Identify grammatical errors (e.g., subject-verb agreement, incorrect tense usage).
   - Spot punctuation mistakes (e.g., missing commas, misuse of semicolons).

2. **Name Inconsistencies:**
   - Flag variations in the representation of names within the same document.
   - Detect discrepancies in names across related documents if applicable.

3. **Date Inconsistencies:**
   - Identify inconsistent date formats within a document.
   - Detect illogical or impossible date sequences (e.g., start date after end date).

4. **Domain-Specific Mistakes:**
   - Detect invalid or incorrectly formatted policy numbers (assume valid formats like "POL-1234567" or "INS-9876543-2025").
   - Identify unrealistic coverage amounts relative to insurance type.
   - Flag incorrect insurance-specific terminology (suggest the correct term).
   - Detect missing critical information (e.g., missing policy start date, missing policy number).

**Output Requirements:**
- For each detected error, produce an object with the following keys:
  - `Line_Number`: Line number of the error. This should be more precise.
  - `Page_Number`: Give the page number of the error.
  - `Error_Type`: Type of error.
  - `Error_description`: Clear, detailed description of the issue.
  - `Suggestions`: Recommended correction or improvement.

- Always output a **single JSON object** with one key `"errors"` containing a **list of error objects** — even if there is only one error.
- Do not add any explanation or commentary — **only output valid JSON**.
- Ensure there are no parsing errors in the JSON structure.

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
      "Page_Number": 3,
      "Line_Number": 12,
      "Error_Type": "Typographical Error",
      "Error_Description": "Subject-verb disagreement: The policyholder don't agree with the terms' instead of 'The policyholder doesn't agree with the terms'.",
      "Suggestions": "Change 'don't' to 'doesn't' for proper agreement."",
      "Suggestions": "Change 'don't' to 'doesn't' for proper agreement."
    }},
    {{
      "Page_Number": 5,
      "Line_Number": 8,
      "Error_Type": "Typographical Error",
      "Error_Description": "Incorrect use of semicolon: 'The policy includes coverage for accidents; theft, and damage.'",
      "Suggestions": "Replace the semicolon with a comma to correctly list the items: 'The policy includes coverage for accidents, theft, and damage.'"
    }},
    {{
      "Page_Number": 1,
      "Line_Number": 12,
      "Error_Type": "Typographical Error",
      "Error_Description": "Incorrectly omitted semicolon: 'The policy covers the following; fire damage, theft, and personal liability.'",
      "Suggestions": "Ensure proper semicolon usage when separating clauses or lists containing commas: 'The policy covers the following; fire damage, theft, and personal liability.'"
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
      "Error_Type": "Name Inconsistencies",
      "Error_Description": "The name is represented as 'John A. Smith' here, while elsewhere in the document it appears as 'John Smith' and 'J. Smith'.",
      "Suggestions": "Standardize the name representation to 'John A. Smith' throughout the document for consistency."
    }},
    {{
      "Page_Number": 3,
      "Line_Number": 10,
      "Error_Type": "Name Inconsistencies",
      "Error_Description": "The name appears as 'Jonathan Smith' here, whereas in the associated policy document it is 'John A. Smith'.",
      "Suggestions": "Verify the correct name and update the document to reflect consistent and accurate representation."
    }},
    {{
      "Page_Number": 1,
      "Line_Number": 5,
      "Error_Type": "Name Inconsistencies",
      "Error_Description": "The name is abbreviated as 'J. Smith' in this section, whereas the full name 'John Smith' is used elsewhere.",
      "Suggestions": "Avoid abbreviations to ensure clarity and uniformity. Use the full name 'John Smith'."
    }},
    {{
      "Page_Number": 2,
      "Line_Number": 15,
      "Error_Type": "Name Inconsistencies",
      "Error_Description": "Variations of the name detected within the document: 'John Smith,' 'John A. Smith,' and 'J. Smith.'",
      "Suggestions": "Ensure consistent representation of the name in all relevant insurance documents."
    }},
    {{
      "Page_Number": 2,
      "Line_Number": 10,
      "Error_Type": "Name Inconsistencies",
      "Error_Description": "Inconsistent representation: 'Jane Doe' in one section, 'J. Doe' in another.",
      "Suggestions": "Standardize the name across the document as 'Jane Doe' to maintain uniformity."
    }},
    {{
      "Page_Number": 4,
      "Line_Number": 17,
      "Error_Type": "Date Inconsistencies",
      "Error_Description": "The date is formatted as 'MM/DD/YYYY' in this section, while elsewhere in the document it uses 'YYYY-MM-DD' and 'DD-MMM-YYYY'.",
      "Suggestions": "Standardize all date formats in the document to 'YYYY-MM-DD' for clarity and uniformity."
    }},
    {{
      "Page_Number": 7,
      "Line_Number": 22,
      "Error_Type": "Date Inconsistencies",
      "Error_Description": "The policy start date is '2025-04-01', which is after the policy end date '2025-03-31'.",
      "Suggestions": "Correct the dates so that the start date is earlier than the end date."
    }},
    {{
      "Page_Number": 2,
      "Line_Number": 8,
      "Error_Type": "Date Inconsistencies",
      "Error_Description": "The date '28-APR-2025' is inconsistent with the format used elsewhere in the document, 'DD/MM/YYYY'.",
      "Suggestions": "Change '28-APR-2025' to '28/04/2025' for consistent formatting."
    }},
    {{
      "Page_Number": 3,
      "Line_Number": 12,
      "Error_Type": "Policy Number Error",
      "Error_Description": "Invalid policy number format: 'POL123XYZ' does not match the expected format 'POL-XXXXX-YYYY'.",
      "Suggestions": "Correct the policy number to follow the standard format, e.g., 'POL-12345-2025'."
    }},
    {{
      "Page_Number": 5,
      "Line_Number": 18,
      "Error_Type": "Coverage Amount Error",
      "Error_Description": "Unrealistic coverage amount: '$10,000,000' for a basic auto insurance policy.",
      "Suggestions": "Reassess the coverage amount and update it to reflect a realistic range for basic auto insurance, e.g., '$50,000 to $100,000'."
    }},
    {{
      "Page_Number": 7,
      "Line_Number": 25,
      "Error_Type": "Terminology Error",
      "Error_Description": "Incorrect terminology: 'insured person' used instead of 'policyholder'.",
      "Suggestions": "Replace 'insured person' with 'policyholder' for proper insurance terminology."
    }},
    {{
      "Page_Number": 9,
      "Line_Number": 30,
      "Error_Type": "Missing Information",
      "Error_Description": "Required field missing: The policy start date is not provided.",
      "Suggestions": "Add the policy start date to complete the document and ensure accuracy."
    }},
    {{
      "Page_Number": 2,
      "Line_Number": 14,
      "Error_Type": "Policy Number Error",
      "Error_Description": "Policy number 'P-1234' is incomplete and does not match the expected format 'POL-XXXXX-YYYY'.",
      "Suggestions": "Expand the policy number to fit the standard format, e.g., 'POL-12345-2023'."
    }},
    {{
      "Page_Number": 6,
      "Line_Number": 22,
      "Error_Type": "Coverage Amount Error",
      "Error_Description": "Coverage amount '$500' is unrealistically low for a comprehensive homeowner insurance policy.",
      "Suggestions": "Adjust the coverage amount to reflect a reasonable range for homeowner insurance, e.g., '$100,000 to $500,000'."
    }},
    {{
      "Page_Number": 8,
      "Line_Number": 10,
      "Error_Type": "Terminology Error",
      "Error_Description": "Incorrect term 'beneficiary' used instead of 'covered party' in the context of the policyholder's coverage.",
      "Suggestions": "Replace 'beneficiary' with 'covered party' for accurate terminology."
    }},
    {{
      "Page_Number": 11,
      "Line_Number": 27,
      "Error_Type": "Missing Information",
      "Error_Description": "The document lacks critical information: no expiration date for the policy is provided.",
      "Suggestions": "Add the expiration date to ensure the document is complete and compliant."
    }},
    {{
      "Page_Number": 4,
      "Line_Number": 18,
      "Error_Type": "Policy Number Error",
      "Error_Description": "The policy number contains unsupported special characters: '#12345*2023'.",
      "Suggestions": "Remove special characters and standardize the policy number to 'POL-12345-2023'."
    }},
    {{
      "Page_Number": 7,
      "Line_Number": 19,
      "Error_Type": "Coverage Amount Error",
      "Error_Description": "Coverage amount '$15,000,000' is overly high for a typical auto insurance policy.",
      "Suggestions": "Lower the coverage amount to reflect realistic values for auto insurance, e.g., '$100,000 to $300,000'."
    }},
    {{
      "Page_Number": 9,
      "Line_Number": 8,
      "Error_Type": "Terminology Error",
      "Error_Description": "Term 'indemnified party' used instead of the more precise 'policyholder' in this context.",
      "Suggestions": "Change 'indemnified party' to 'policyholder' for clear and accurate communication."
    }},
    {{
      "Page_Number": 12,
      "Line_Number": 5,
      "Error_Type": "Missing Information",
      "Error_Description": "The document does not include the policyholder's contact details.",
      "Suggestions": "Add contact details for the policyholder to ensure completeness."
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
    }}
  ]
}}
    ```


**Input document:**

{chunk}
        """

        for attempt in range(retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a highly advanced AI designed to analyze insurance-related documents and detect errors. Your primary task is to identify and categorize errors, then generate a detailed report"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=8000,
                    temperature=0.2
                )

                raw_content = response.choices[0]['message']['content']

                if raw_content.startswith("```json"):
                    raw_content = raw_content.replace("```json", "").replace("```", "").strip()

                parsed = json.loads(raw_content)
                errors = parsed.get("errors", [])
                analysis_reports.extend(errors)
                break  # Success

            except json.JSONDecodeError:
                st.error("❌ Failed to parse JSON from GPT response.")
                st.code(raw_content)
                with open("gpt_raw_output_error.json", "w", encoding="utf-8") as f:
                    f.write(raw_content)
                break

            except APIError as e:
                st.warning(f"⚠️ API Error on attempt {attempt + 1}: {e}")
                time.sleep(delay)

            except RateLimitError:
                st.warning("🚫 Rate limit exceeded. Retrying after delay...")
                time.sleep(delay)
                continue

            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                break

        time.sleep(1)  # Optional: small pause between chunks

    return analysis_reports

def output_preprocessing(row: dict):
    return row["Line_Number"], row["Error_Type"], row["Error_description"], row["Suggestions"]

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
    st.title("🛡 Insurance Document Error Detector")
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
                analysis_report = analyze_text_with_gpt(file_content)

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
                    file_name="Analysis_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
