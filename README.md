**Overview** :
This project leverages advanced AI models and tools to efficiently analyze and summarize insurance documents. It employs different strategies to handle and process data, offering robust solutions tailored to various requirements.

**Components** :

**STA** :

* Utilizes GPT-4 O Mini without Chunking.

* Best suited for smaller data sets or documents requiring linear analysis.

**STAA** :

* Utilizes GPT-4 O Mini with Chunking.

* Ideal for processing large and complex documents by breaking them into smaller, manageable sections.

**STAG** :

* Employs Gemini 1.5 Pro with Chunking.

* Optimized for advanced analysis and chunk-based processing with enhanced efficiency and accuracy.

**STA_Summary** :

* Designed specifically for summarizing insurance documents.

* Provides concise and clear summaries, saving time and effort in understanding lengthy data.

**User Interface** :

The project incorporates Streamlit, offering a simple yet interactive interface for users to access and visualize the processed data effectively.

**Getting Started** : 

Install required dependencies using: **pip install streamlit**

Run the application: **streamlit run STAA.py**

Follow the prompts on the interface to upload documents and choose the processing mode.

**Applications**: This tool is particularly suited for:

* Insurance document analysis.

* Exporting processed tables and summaries.

* Efficient handling of large-scale textual data.
