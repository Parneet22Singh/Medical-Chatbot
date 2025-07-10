from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from llm_setup import load_light_llm
from pinecone_setup import get_index

def build_chat_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = get_index()
    vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)

    # Configure retriever to fetch top 3 results from Pinecone index
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = load_light_llm()

    # Custom system prompt (same as in llm_setup.py)
    system_instruction = (
        "You are a helpful and knowledgeable medical assistant. "
        "Use the provided documents as context, but if they lack specific treatment details or drug names, "
        "you may provide accurate and up-to-date medical information from your own knowledge. "
        "Focus on answering the user's question clearly and directly, including treatment names or drug recommendations when appropriate."
    )

    # Custom prompt template
    custom_template = PromptTemplate(
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
        chain_type_kwargs={"prompt": custom_template}
    )

    return qa_chain
