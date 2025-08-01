# 🩺 Medical Chatbot (Gemini + PubMed + Pinecone)

A powerful AI-driven medical chatbot that supports:

- 🤖 **General Q&A** using Google's **Gemini 2.5 Pro**
- 📚 **Scientific Search** with **PubMed + LangChain RAG**
- 🧠 **Document embedding** using **Pinecone**
- 💬 Clean chat interface with **downloadable history**

---

## 🚀 Features

- **Gemini-based chat**: Ask medical questions like symptoms, treatments, or biology.
- **PubMed RAG mode**: Search scientific literature and get AI-generated answers backed by real studies.
- **Chat history**: Stored per session with download option.
- **Modern UI**: Styled with custom colors and layout using Streamlit + HTML/CSS.

---
## 🛠️ Setup Instructions

### 1. Clone the Repository

-git clone https://github.com/Parneet22Singh/medical-chatbot.git
-cd medical-chatbot

### Install Dependencies
- pip install -r requirements.txt

### Configure API Keys
Create a .streamlit/secrets.toml file with the following:

- gemini_api_key = "your_google_gemini_api_key"
- pinecone_api_key = "your_pinecone_api_key"
- pinecone_environment = "us-west1-gcp"  # adjust if needed

### Run the app
- streamlit run app.py

### Details
- Chat_engine.py deals with the vectoriazation of the incoming data.
- llm_setup.py performs the function of the chatbot in the PubMed RAG mode.
- gemini_qa.py consists of the code for the gemini powered chatbot in General-Q&A mode.
- fetch_medical_docs.py file contains the code that deals with the fetching of relevant document from PubMed using NIH API.

### Note
- The PubMed Mode is more tuned for research and scientific purposes, for general purpose General Q&A can be used.
- If there are connectivity issues or API request issues, simply restart (relaunch it in localhost) the session (try out reloading the website first).
