import os
import streamlit as st
from fetch_medical_docs import search_pubmed, fetch_abstracts, embed_articles_to_pinecone
from chat_engine import build_chat_chain
import google.generativeai as genai

# ───────────────────────────────
# 🔑 Load Gemini API key from environment or secrets
API_KEY = st.secrets.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ Gemini API key not found. Please set GEMINI_API_KEY in environment or .streamlit/secrets.toml")
    st.stop()

genai.configure(api_key=API_KEY)
MODEL_NAME = "models/gemini-2.5-pro"

# ───────────────────────────────
# 🧠 Initialize model and chat session
if "chat_model" not in st.session_state:
    st.session_state.chat_model = genai.GenerativeModel(MODEL_NAME)
if "chat_session" not in st.session_state:
    st.session_state.chat_session = st.session_state.chat_model.start_chat()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ───────────────────────────────
# 🎨 Custom CSS Styling and scrollable chat container
st.markdown("""
    <style>
    body {
        background-color: white !important;
        color: black;
    }
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding-right: 10px;
        margin-bottom: 1rem;
        border: 1px solid #ccc;
        border-radius: 8px;
    }
    .question {
        font-weight: bold;
        font-size: 1.1rem;
        color: white !important;
        background-color: #3366cc;
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 6px;
        max-width: 90%;
        word-wrap: break-word;
    }
    .answer {
        background-color: #e6f0ff;
        padding: 16px;
        border-left: 5px solid #3366cc;
        border-radius: 6px;
        font-size: 1.05rem;
        color: #000000;
        margin-bottom: 20px;
        max-width: 90%;
        word-wrap: break-word;
    }
    .history-header {
        margin-top: 2rem;
        font-size: 1.25rem;
        font-weight: 600;
        border-bottom: 2px solid #ccc;
        padding-bottom: 4px;
        margin-bottom: 10px;
        color: #1a1a1a;
    }
    </style>
""", unsafe_allow_html=True)

# ───────────────────────────────
# 🎯 Streamlit UI
st.title("🩺 Medical Chatbot (Pinecone + Gemini)")
mode = st.radio("Choose interaction mode:", ["PubMed Search", "General Q&A"], index=0)
query = st.text_input("What would you like to know?")

# ───────────────────────────────
# 💬 General Q&A Mode
def handle_general_qa(user_query):
    try:
        st.session_state.chat_history.append(("You", user_query))
        response = st.session_state.chat_session.send_message(user_query)
        st.session_state.chat_history.append(("Gemini", response.text))
    except Exception as e:
        st.error(f"❌ Error from Gemini API: {e}")

# ───────────────────────────────
# 🔍 PubMed RAG Mode
def handle_pubmed_search(user_query):
    with st.spinner("🔍 Searching PubMed and embedding..."):
        pmids = search_pubmed(user_query, max_results=5)
        articles = fetch_abstracts(pmids)
        embed_articles_to_pinecone(articles)

    with st.spinner("🤖 Answering using retrieved docs..."):
        qa_chain = build_chat_chain()
        result = qa_chain.invoke({"query": user_query})

    st.subheader("💡 Answer:")
    st.write(result['result'])

    st.subheader("📄 Source Documents:")
    seen = set()
    for doc in result['source_documents']:
        title = doc.metadata.get('title', 'Untitled')
        if title not in seen:
            st.markdown(f"**{title}**")
            st.write(doc.page_content)
            seen.add(title)

# ───────────────────────────────
# 🧠 Main Logic
if query:
    if mode == "General Q&A":
        handle_general_qa(query)
    elif mode == "PubMed Search":
        handle_pubmed_search(query)

# ───────────────────────────────
# 🧵 Chat History Display
if mode == "General Q&A" and st.session_state.chat_history:
    st.subheader("🧠 Gemini Q&A")
    chat_html = '<div class="chat-container">'
    for role, msg in st.session_state.chat_history:
        if role == "You":
            chat_html += f'<div class="question"><strong>{role}:</strong> {msg}</div>'
        else:
            chat_html += f'<div class="answer">{msg}</div>'
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    if st.download_button("📥 Download Q&A Log",
                          data="\n\n".join(f"{r}: {m}" for r, m in st.session_state.chat_history),
                          file_name="medical_chat_history.txt"):
        st.success("✅ Chat log downloaded.")

# ───────────────────────────────
# Auto-scroll chat container to bottom (works on most browsers)
st.markdown("""
<script>
const chatContainer = window.parent.document.querySelector('.chat-container');
if(chatContainer){
    chatContainer.scrollTop = chatContainer.scrollHeight;
}
</script>
""", unsafe_allow_html=True)
