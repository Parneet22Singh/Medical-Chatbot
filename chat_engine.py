import streamlit as st
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from fetch_medical_docs import search_pubmed, fetch_abstracts
from llm_setup import load_light_llm


def run_pubmed_chain(user_query):
    with st.spinner("🔍 Searching PubMed and embedding..."):
        pmids = search_pubmed(user_query, max_results=5)
        st.write(f"🔎 PubMed returned {len(pmids)} article IDs.")

        if not pmids:
            return {"result": "⚠️ No PubMed articles found for this query.", "source_documents": []}

        articles = fetch_abstracts(pmids)
        st.write(f"📄 Fetched {len(articles)} articles with abstracts.")

        docs = [
            Document(page_content=a["abstract"], metadata={"title": a["title"]})
            for a in articles if a["title"] and a["abstract"]
        ]
        st.write(f"🗂 Created {len(docs)} documents after filtering.")

        if not docs:
            return {"result": "⚠️ No relevant documents found with abstracts and titles.", "source_documents": []}

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = load_light_llm()

        system_instruction = (
            "You are a helpful and knowledgeable medical assistant. "
            "Use the provided documents as context. "
            "If they do not fully answer the question, provide accurate and up-to-date information from your own medical knowledge. "
            "Always prioritize relevance and clarity in your answer."
        )

        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(f"{system_instruction}\n\n"
                      "Context:\n{context}\n\n"
                      "Question: {question}\n"
                      "Answer:")
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={"prompt": custom_prompt}
        )

        with st.spinner("🤖 Answering using retrieved docs and reasoning..."):
            result = qa_chain.invoke({"query": user_query})

        return result
        
